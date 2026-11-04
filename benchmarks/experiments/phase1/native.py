"""Native DSH baseline with ordinary tools and supervisor-owned verification.

The public stdlib check.py client submits source to a per-cell loopback service.
Only the trusted service sees the backend/SSH configuration and core verifier.
There is no operator plugin, core CLI, generation delegation or final evaluator
on the participant surface. The parent remains the deadline/cleanup owner.
"""
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import sys
import subprocess
import threading
import time

from benchmarks.experiments.s5 import s5_arm_env, s5_plugin_env
from benchmarks.experiments.s5.s5_development_service import _HTTP, _Handler
from operator_agent.app import autonomous, hitl_tools as ht
from operator_agent.execute.budget import BudgetLedger, BudgetPlan
from operator_agent.execute.task_budget import TaskBudgetConfig
from operator_agent.execute.task_clock import monotonic
from operator_agent.generate.repair_loop import _full_l2_passed

from .baseline import _persist_verification
from .connection import options
from .delegate import source_ref
from .plan import REPO, read, write_once
from .summary import _verdicts


class VerificationService:
    """Serialize full ordinary verification and reserve its budget first."""
    def __init__(self,config,contract_hash):
        self.config=config
        self.work=Path(config['run_dir'])
        self.contract_hash=contract_hash
        self.policy=TaskBudgetConfig.from_dict(config['budget'])
        self.ledger=BudgetLedger(BudgetPlan(oracle_total=self.policy.oracle_total,
            wall_clock_s=self.policy.wall_clock_s))
        self.lock=threading.Lock()
        self.token=secrets.token_urlsafe(32)
        self.timeout=max(1,min(1810,int(self.policy.wall_clock_s)+5))
        self.records=[]
        self.events=[]
        self.selected_source=None
        self.selected_hash=None
        self.levels={}
        self.used=0
        self.closed=False
        self.no_gain_streak=0
        self.stop_reason=None
        self.server=None
        _persist_verification(self.work,self.ledger,self.policy.oracle_total)

    def __enter__(self):
        self.server=_HTTP(('127.0.0.1',0),_Handler)
        self.server.owner=self
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        return self

    def client_config(self):
        if self.server is None:raise RuntimeError('native verifier not started')
        # The unmodified public client accepts mock/ssh. For local fixture
        # checks mock is a transport label only; execution remains explicit
        # FakeBackend and every retained verdict keeps its test_only context.
        return dict(url='http://127.0.0.1:%d/development/verify'%self.server.server_address[1],
            token=self.token,contract=self.contract_hash,
            backend='ssh' if self.config['backend']=='ssh' else 'mock',timeout=self.timeout)

    def _save(self,index,code,response,*,admitted,error=None):
        response=dict(response,**self.budget_status())
        path=self.work/'native-verification'/('submission-%04d.json'%index)
        row=dict(index=index,timestamp=time.time(),source_hash=hashlib.sha256(code.encode()).hexdigest(),
            contract_hash=self.contract_hash,admitted=admitted,response=response,error=error)
        write_once(path,row)
        self.records.append(row)
        self.events.append(dict(id='native-verify-%d'%index,timestamp=row['timestamp'],stage='check',
            actor='agent',kind='tool_result',tool='native.check',action='candidate_verify',
            ok=response.get('passed') if response.get('ok') else False,source=source_ref(path)))
        autonomous._atomic_json(self.work/'events.json',self.events)
        return response

    def budget_status(self):
        """Expose the common ledger without introducing a second quota."""
        reserved=self.ledger.caps['confirm']
        return dict(development_used=self.used,
            max_development=self.policy.oracle_total-reserved,
            confirm_reserved=reserved,
            remaining=dict(non_confirm=max(0,self.policy.oracle_total-reserved-self.ledger.spent_all),
                           confirm=max(0,reserved-self.ledger.spent['confirm'])),
            oracle_unit='candidate_verification_dispatch_per_input_group')

    def _verify(self,index,candidate):
        """Run the ordinary verifier in a fresh interpreter's main thread.

        Candidate guards may fork. Forking from an HTTP request thread after
        NumPy/BLAS work can inherit locked library state, so HTTP threads only
        reserve/record and supervise this trusted verifier process.
        """
        request=self.work/'native-verification'/('input-%04d.json'%index)
        receipt=self.work/'native-verification'/('result-%04d.json'%index)
        write_once(request,dict(run_dir=str(self.work),source=str(candidate),
            source_hash=self.selected_hash,contract_hash=self.contract_hash,
            backend=self.config['backend'],connection=self.config.get('connection',{}),result=str(receipt)))
        with (self.work/'native-verification'/('verifier-%04d.log'%index)).open('x') as stream:
            process=subprocess.run([sys.executable,'-B','-m','benchmarks.experiments.phase1.native',
                '--verify',str(request)],cwd=str(REPO),stdout=stream,stderr=subprocess.STDOUT,
                timeout=max(.01,self.config['deadline_monotonic']-monotonic()))
        if process.returncode or not receipt.is_file():raise RuntimeError('native verifier did not publish a result')
        return read(receipt)

    def develop(self,code,contract_hash):
        if (not isinstance(code,str) or not code.strip() or len(code.encode('utf-8'))>65536
                or contract_hash!=self.contract_hash):
            raise ValueError('frozen contract and bounded source are required')
        with self.lock:
            index=len(self.records)+1
            candidate=self.work/'native-verification'/('candidate-%04d.py'%index)
            candidate.parent.mkdir(parents=True,exist_ok=True)
            with candidate.open('x',encoding='utf-8') as stream:stream.write(code)
            self.selected_source=candidate
            self.selected_hash=source_ref(candidate)['sha256']
            self.levels={}
            if self.closed or monotonic()>=self.config['deadline_monotonic']:
                return self._save(index,code,dict(ok=False,error='verification_closed',
                    stop_reason=self.stop_reason),admitted=False)
            phase='explore' if self.used==0 else 'refine'
            if not self.ledger.oracle_may(phase,1):
                self.ledger.oracle_refuse(phase)
                _persist_verification(self.work,self.ledger,self.policy.oracle_total)
                return self._save(index,code,dict(ok=False,error='development_budget_exhausted'),admitted=False)
            self.ledger.oracle_spend(phase,1)
            self.used+=1
            _persist_verification(self.work,self.ledger,self.policy.oracle_total)
            self.events.append(dict(id='native-verify-%d-call'%index,timestamp=time.time(),stage='check',
                actor='agent',kind='tool_call',tool='native.check',action='candidate_verify',
                source=source_ref(candidate)))
            autonomous._atomic_json(self.work/'events.json',self.events)
            try:
                result=self._verify(index,candidate)
                self.levels=result['levels']
                full_l2=_full_l2_passed(_verdicts(result['verdicts']))
                passed=(result.get('passed') is True or
                        self.config['backend']=='fake' and full_l2 is True)
                status='passed' if passed else 'failed' if result.get('ok') is True else 'unknown'
                reply=dict(schema='s5-development-feedback-v1',ok=True,phase='development',
                    source_sha256=self.selected_hash,contract_hash=self.contract_hash,
                    status=status,passed={'passed':True,'failed':False,'unknown':None}[status],
                    levels=result['levels'],verdicts=result['verdicts'],full_l2_passed=full_l2,
                    backend=self.config['backend'],verification_evidence=source_ref(result['evidence']),
                    execution_context=self.config['execution_context'])
                self._note_no_gain(reply)
                return self._save(index,code,reply,admitted=True)
            except Exception as exc:
                # Retain consumed reservation and candidate even when the
                # ordinary verifier cannot finish. Do not refund or guess L2.
                self.levels={}
                reply=dict(schema='s5-development-feedback-v1',ok=True,phase='development',
                    source_sha256=self.selected_hash,contract_hash=self.contract_hash,
                    status='unknown',passed=None,levels={},full_l2_passed=None,
                    backend=self.config['backend'],error=type(exc).__name__,
                    execution_context=self.config['execution_context'])
                return self._save(index,code,reply,admitted=True,error=type(exc).__name__)


    def _note_no_gain(self,reply):
        """连续「跑对但没更快」到阈值就关闭验证 —— A 臂此前唯一的限制是预算。
 
        为什么 A 臂需要这条：它不经过 Conductor，10 次提交里可以有 6 次都是
        「L1✓ L2✓ L3✗」，一路烧到模型调用用满（2026-11-04 实测 773 s / 30 次调用）。
        计数只吃 `l3_no_gain()` 的 True；无效测量与「没跑到 L3」都不计入。
        """
        from operator_agent.app.conductor import MAX_STAGNANT_ROUNDS
        counted=l3_no_gain(reply)
        if counted is None:
            return
        if counted is False:
            self.no_gain_streak=0
            return
        self.no_gain_streak+=1
        if self.no_gain_streak>=MAX_STAGNANT_ROUNDS:
            self.stop_reason='no_gain_%d_rounds'%MAX_STAGNANT_ROUNDS
            self.closed=True
            print('  [native] 连续 %d 次「跑对但没更快」，关闭验证（不再接受提交）'
                  % self.no_gain_streak)

    def snapshot(self):
        with self.lock:
            return dict(selected_source=str(self.selected_source) if self.selected_source else None,
                source_hash=self.selected_hash,levels=dict(self.levels),verification_count=self.used,
                last_submission_verified=bool(self.records and self.records[-1]['admitted']
                    and self.records[-1].get('error') is None),records=list(self.records),
                no_gain_streak=self.no_gain_streak,stop_reason=self.stop_reason,
                **self.budget_status())

    def __exit__(self,*exc):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        # Existing handler threads remain parent-owned until their verification
        # completes or the parent kills the cell at its hard deadline.
        with self.lock:self.closed=True


def run(config):
    if config.get('arm')!='A' or config.get('entry_a')!='dsh_native':
        raise ValueError('native adapter requires the explicit DSH baseline arm')
    if config['backend'] not in ('fake','ssh'):raise ValueError('explicit fake or SSH backend required')
    from .host import dsh_model_options, model_gateway, runtime_identity
    cell=Path(config['run_dir']).parent
    work=Path(config['run_dir']);work.mkdir(parents=True,exist_ok=True)
    prepared=ht.prepare(config['declaration'],work)
    runtime=s5_plugin_env.stage_runtime(cell/'native-runtime')
    write_once(cell/'host-runtime.json',dict(installed=runtime_identity(),public_client=runtime,
        installed_plugin_acceptance=False))
    service=VerificationService(config,prepared['contract_hash'])
    result=dict(cell_id=config['cell_id'],plan_hash=config['plan_hash'],arm='A',phase='failed',
        reason='native_host_incomplete',selected_source=None,source_hash=None,levels={},
        contract_hash=prepared['contract_hash'],execution_context=config['execution_context'],
        dsh_entry_verified=False,entry_a='dsh_native')
    try:
        with service,model_gateway(config,cell) as gateway:
            built=s5_plugin_env.build('A',runtime=runtime,development=service.client_config(),
                base_url=gateway.base_url,arm_dir=cell/'dsh',model=config['model']['id'],
                **dsh_model_options(config))
            write_once(cell/'host-surface.json',dict(surface=built['surface'],
                patch_sha256=built['patch_sha256'],env_keys=sorted(built['env']),
                provider_credential_in_child=False,source_plugin=False,ordinary_shell_edit=True))
            prompt=('Implement the frozen operator using the ordinary shell, read and edit tools. '
                'Write source yourself; no operator plugin, core CLI or generation delegation is provided. '
                'Submit each candidate with '+shlex.quote(str(Path(sys.executable).resolve()))+
                ' check.py --code-file candidate.py. '
                'This public client performs full verification through a supervisor-owned backend. '
                'Use its feedback to revise the source. The final candidate is the last source submitted '
                'to check.py, including a valid submission refused after budget exhaustion; previous best '
                'candidates will not be selected. Do not retry after verification_closed or '
                'development_budget_exhausted. Each oracle unit counts one candidate-verification dispatch '
                'per frozen input group, covering full L1/L2/L3; it is not a GPU launch, GPU seconds, '
                'or the number of internal correctness cases. The confirm allowance is protected and '
                'cannot be spent through this public development client. Independent final evaluation '
                'is supervised separately. Report the actual last result. Frozen common budget: '+
                json.dumps(config['budget'],ensure_ascii=False,sort_keys=True)+
                '. Current oracle allowance: '+json.dumps(service.budget_status(),sort_keys=True)+
                '. Effective wire sampling policy: '+json.dumps(config.get('sampling') or
                    dict(status='not_fixed; inspect retained actual requests'),ensure_ascii=False,sort_keys=True)+
                '. Frozen declaration: '+
                json.dumps(config['declaration'],ensure_ascii=False))
            response=s5_arm_env.start_headless(built,prompt,
                timeout=max(.01,config['deadline_monotonic']-monotonic()),capture_dir=cell/'host-output')
            write_once(cell/'host-response.json',response)
            state=service.snapshot()
            result.update({k:state[k] for k in ('selected_source','source_hash','levels',
                'verification_count','last_submission_verified')})
            exhausted = _model_budget_exhausted(config)
            # 验证服务自己关掉的（连续无增益）优先于「宿主怎么退的」—— 它更具体。
            stop=state.get('stop_reason')
            if response['returncode']==0 and state['selected_source'] and not stop:
                result.update(phase='done', reason='native_host_returned')
            else:
                # 宿主退出时不再丢掉已通过的候选：验证服务的判定与宿主生死无关。
                best = last_passing_submission(state.get('records'))
                if best is not None:
                    result.update(selected_source=best.get('selected_source') or
                                      str(service.work/'native-verification'/
                                          ('candidate-%04d.py'%best['index'])),
                                  source_hash=best.get('source_hash'),
                                  levels=(best.get('response') or {}).get('levels') or {},
                                  phase='done',
                                  reason=stop or ('budget_calls_exhausted' if exhausted
                                                  else 'host_ended_after_pass'))
                elif stop:
                    result.update(phase='failed', reason=stop,
                                  no_gain_streak=state.get('no_gain_streak'))
                elif exhausted:
                    result.update(phase='failed', reason='budget_calls_exhausted')
            result['host_response']=source_ref(cell/'host-response.json')
    except Exception as exc:
        result.update(phase='failed',reason=type(exc).__name__)
        raise
    finally:
        state=service.snapshot()
        result.update({k:state[k] for k in ('selected_source','source_hash','levels',
            'verification_count','last_submission_verified')})
        write_once(work/'native-service.json',dict(contract_hash=prepared['contract_hash'],**state))
        if not (work/'events.json').exists():write_once(work/'events.json',[])
        _persist_verification(work,service.ledger,service.policy.oracle_total)
        write_once(work/'connected-result.json',result)
    return result


def _model_budget_exhausted(config):
    """模型调用额度是否已经用满（或已中止）。读账本，不猜文本。"""
    try:
        from operator_agent.execute import s5_shared_budget as shared
        snap=shared.snapshot(config['shared_ledger'],expected_task_id=config['shared_task_id'])
        return bool(snap.get('aborted')) or snap.get('remaining_calls')==0
    except Exception:
        return False


def l3_no_gain(response):
    """这一次提交是否算「跑对了但没更快」——停滞计数的唯一输入。

    返回 True(计一次)/False(有改善)/None(不计数)。
    不计数的是：没跑到 L3（编译或数值问题，属于另一类失败），以及**测量无效**的
    L3（degraded/uninformative）—— 拿无效读数判停滞就是「提前停不可恢复」。
    """
    verdicts={v.get('level'):v for v in (response or {}).get('verdicts') or []
              if isinstance(v,dict)}
    l3=verdicts.get('L3')
    if not isinstance(l3,dict):
        return None
    if l3.get('passed') is True:
        return False
    metrics=l3.get('metrics') or {}
    if metrics.get('measurement_validity')!='valid':
        return None
    return True


def last_passing_submission(records):
    """最后一个 L1/L2/L3 全过的提交；没有就返回 None。

    为什么要这个：A 臂的候选是经**可信验证服务**判的，它的 verdict 不依赖宿主是否
    还活着。2026-11-04 真机：一格连着 verify-5/7/8 三次全过，随后模型调用额度用满、
    网关回 429、DSH 进程非零退出 —— 于是 `selected_source` 为空、整格记成
    `native_host_failed`，**三次真实通过全部作废**。候选的选择规则本来就是「最后一次
    提交」，这里只是把它落到「最后一次**通过**的提交」，不挑最好的那个。
    """
    for row in reversed(records or []):
        response = (row or {}).get('response') or {}
        levels = response.get('levels') or {}
        if getattr(response.get('passed'), 'real', response.get('passed')) is True or (
                levels and all(levels.get(k) is True for k in ('L1', 'L2', 'L3'))):
            return row
    return None


def verify_worker(request):
    """Trusted worker entry; not exposed to the DSH participant."""
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():raise ValueError('native verifier refuses inherited public delegation')
    source=Path(request['source'])
    data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!=request['source_hash']:
        raise ValueError('native source binding mismatch')
    if read(Path(request['run_dir'])/'contract.json').get('hash')!=request['contract_hash']:
        raise ValueError('native contract binding mismatch')
    result=ht.verify(request['run_dir'],data.decode('utf-8'),backend=request['backend'],
        candidate_origin='outer',**options(request))
    write_once(request['result'],result)


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--verify':raise SystemExit('trusted --verify request required')
    verify_worker(read(sys.argv[2]))
