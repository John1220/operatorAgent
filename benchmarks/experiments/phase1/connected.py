"""Serial entry integration with a parent-owned deadline and independent final."""
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

from operator_agent.app.autonomous import _atomic_json
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.task_clock import monotonic
from . import connected_plan as plans, plan as files
from .connection import credential
from .runner import _supervise, _cancelled, _offline_environment


def _resources():
    """当前进程的 fd 用量与软/硬限制；读不到就省键，不编数字。"""
    value={}
    try:
        import resource
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        value.update(nofile_soft=soft if soft!=resource.RLIM_INFINITY else None,
                     nofile_hard=hard if hard!=resource.RLIM_INFINITY else None)
    except Exception:
        pass
    try:
        value['open_fds']=len(os.listdir('/dev/fd'))
    except OSError:
        pass
    value['pid']=os.getpid()
    return value


def configuration(plan,task,slot,cell,started):
    connection=dict(plan['connection'])
    value=dict(schema='phase1-connected-cell-v1',**slot,plan_hash=plan['hash'],
        run_dir=str(cell/'work'),declaration=task['declaration'],model=plan['model'],
        backend=plan['backend'],connection=connection,gpu=plan['gpu'],entry_b=plan['entry_b'],
        budget=plan['budget'],max_rounds=plan['max_rounds'],execution_context=plan['execution_context'],
        shared_ledger=str(cell/'budget.sqlite'),
        shared_task_id='p1c-'+files.digest([plan['hash'],slot['cell_id']])[:24],
        started_monotonic=started,deadline_monotonic=started+plan['budget']['wall_clock_s'])
    for key in ('entry_a','sampling','study'):
        if key in plan:value[key]=plan[key]
    return value


def _final(plan,task,slot,cell,selected,config,root,invocation):
    raw=Path(selected['selected_source'])
    source=raw.resolve()
    if raw.is_symlink() or cell/'work' not in source.parents:
        raise ValueError('candidate must be a regular file inside its cell')
    data=source.read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if digest!=selected['source_hash'] or selected['contract_hash']!=task['contract_hash']:
        raise ValueError('selected source/contract binding mismatch')
    pinned=cell/'selected.py'
    with pinned.open('xb') as f:f.write(data)
    final=dict(config,run_dir=str(cell/'final'),source=str(pinned),source_hash=digest,
        contract_hash=task['contract_hash'])
    files.write_once(cell/'final-config.json',final)
    started=monotonic()
    outcome=_supervise([sys.executable,'-B','-m','benchmarks.experiments.phase1.connected_final',
        str(cell/'final-config.json')],cell,started+plan['final_wall_clock_s'],root,invocation)
    receipt=cell/'final/final-result.json'
    outcome.update(source_hash=digest,contract_hash=task['contract_hash'],max_verifications=1,
                   receipt=str(receipt) if receipt.is_file() else None)
    if receipt.is_file() and files.read(receipt).get('finished_monotonic',float('inf'))>outcome['deadline_monotonic']:
        outcome['stopped_reason']='budget_wall_clock'
    return outcome


def _cell(plan,task,slot,root,invocation):
    cell=root/'cells'/slot['cell_id']
    started=monotonic()
    cfg=configuration(plan,task,slot,cell,started)
    if 'study' in cfg:
        from .study_instance import check
        files.write_once(cell/'instance-check.json',check(cfg))
    # 诊断面：建账本前记一次进程资源事实。2026-11-04 连跑两轮，第二格起全部
    # `BudgetError`（create 失败会留下 0 字节 sqlite），而单独调用 create 又成功 ——
    # 得先有 fd / 软限制这些现场数字，才谈得上判断是环境还是缺陷。
    _atomic_json(cell/'resources.json',_resources())
    shared.create(cell/'budget.sqlite',task_id=cfg['shared_task_id'],
        max_calls=plan['budget']['max_llm_calls'],max_output_tokens=40960,
        total_output_tokens=40960*plan['budget']['max_llm_calls'],wall_clock_s=plan['budget']['wall_clock_s'])
    files.write_once(cell/'cell-config.json',cfg)
    outcome=_supervise([sys.executable,'-B','-m',plan['entries'][slot['arm']],str(cell/'cell-config.json')],
        cell,cfg['deadline_monotonic'],root,invocation)
    record=dict(schema='phase1-connected-record-v1',**slot,plan_hash=plan['hash'],phase='failed',
        reason=outcome['stopped_reason'] or 'worker_missing_terminal_receipt',development=outcome,
        execution_context=plan['execution_context'],final_evaluation=None)
    _atomic_json(cell/'record.json',record)
    path=cell/'work/connected-result.json'
    if path.is_file():
        selected=files.read(path)
        if (selected.get('cell_id')!=slot['cell_id'] or selected.get('plan_hash')!=plan['hash'] or
                selected.get('contract_hash')!=task['contract_hash'] or selected.get('execution_context')!=plan['execution_context']):
            raise ValueError('worker receipt identity mismatch')
        record.update(phase=selected['phase'],worker_result=str(path),reason=selected.get('reason'),
                      worker_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        _atomic_json(cell/'record.json',record)
        if outcome['returncode']!=0:record.update(phase='failed',reason='worker_failed')
        # 宿主非零退出**默认**不许触发独立终评（那条守卫是真事故换来的：见
        # test_development_nonzero_exit_cannot_trigger_independent_final）。唯一例外是
        # 工作进程自己声明了「按预算停止/宿主在通过之后结束」—— 此时候选是可信验证
        # 服务判过的，宿主死活与它的有效性无关（native.last_passing_submission）。
        salvaged = selected.get('reason') in ('budget_calls_exhausted', 'host_ended_after_pass')
        if (selected.get('selected_source') and not outcome['stopped_reason']
                and (outcome['returncode']==0 or salvaged)):
            record['final_evaluation']=_final(plan,task,slot,cell,selected,cfg,root,invocation)
            if record['final_evaluation']['stopped_reason']:
                record.update(phase='cancelled' if record['final_evaluation']['stopped_reason']=='user_cancelled' else 'failed',
                    reason=record['final_evaluation']['stopped_reason'])
            elif record['final_evaluation']['returncode']!=0 or not record['final_evaluation']['receipt']:
                record.update(phase='failed',reason='final_worker_failed')
    if outcome['stopped_reason']:
        record.update(phase='cancelled' if outcome['stopped_reason']=='user_cancelled' else 'failed',reason=outcome['stopped_reason'])
    if record['phase']=='failed' and record.get('reason') in (
            'native_host_failed','host_returned','host_left_task_incomplete'):
        # 预算用尽是**正常的预算停止**，不是宿主故障 —— 读账本判，不猜文本。
        try:
            snap=shared.snapshot(cell/'budget.sqlite',expected_task_id=cfg['shared_task_id'])
            if snap.get('aborted') or snap.get('remaining_calls')==0:
                record.update(reason='budget_calls_exhausted',budget_stop=dict(
                    remaining_calls=snap.get('remaining_calls'),
                    aborted=bool(snap.get('aborted')),
                    reserved_calls=snap.get('reserved_calls')))
        except Exception:
            pass
    record['finished_at']=time.time()
    _atomic_json(cell/'record.json',record)
    return record


def run(plan_path,output):
    _offline_environment()
    plan=plans.load(plan_path)
    credential(plan['model'])
    if plan['entry_b']=='dsh' or plan.get('entry_a')=='dsh_native':
        from .host import preflight
        preflight()
    root=files.fresh_directory(output)
    files.write_once(root/'plan.json',plan)
    invocation=uuid.uuid4().hex
    execution=dict(schema='phase1-execution-v1',mode='connected',plan_hash=plan['hash'],
        invocation_id=invocation,phase='running',started_at=time.time(),
        execution_context=plan['execution_context'],formal_comparison_credit=False)
    _atomic_json(root/'execution.json',execution)
    for slot in plan['schedule']:
        (root/'cells'/slot['cell_id']).mkdir(parents=True)
        _atomic_json(root/'cells'/slot['cell_id']/'record.json',dict(**slot,phase='not_started',reason='not_dispatched'))
    tasks={task['task_id']:task for task in plan['tasks']}
    try:
        for slot in plan['schedule']:
            if _cancelled(root,invocation):
                execution['phase']='cancelled';break
            plans.validate(plan)
            try:
                record=_cell(plan,tasks[slot['task_id']],slot,root,invocation)
            except Exception as exc:
                path=root/'cells'/slot['cell_id']/'record.json'
                record=files.read(path)
                # 只记类型名会把「为什么」丢掉：2026-11-04 一轮八格里 7 格记成
                # `reason: BudgetError`，既没理由码也没栈，只能靠重跑猜。
                # 落一份异常详情（不删、不改判据），reason_detail 里带理由码。
                detail=dict(schema='phase1-connected-exception-v1',cell_id=slot['cell_id'],
                    type=type(exc).__name__,message=str(exc)[:500],
                    traceback=traceback.format_exc()[-4000:])
                _atomic_json(root/'cells'/slot['cell_id']/'exception.json',detail)
                record.update(phase='failed',reason=type(exc).__name__,
                    reason_detail=str(exc)[:200] or None,exception='exception.json')
                _atomic_json(path,record)
            if record['phase']=='cancelled':execution['phase']='cancelled';break
        else:execution['phase']='done'
    except (KeyboardInterrupt,ValueError) as exc:
        execution.update(phase='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',reason=type(exc).__name__)
    finally:
        execution['finished_at']=time.time()
        _atomic_json(root/'execution.json',execution)
    from .connected_summary import summarize
    result=summarize(root)
    _atomic_json(root/'summary.json',result)
    return result
