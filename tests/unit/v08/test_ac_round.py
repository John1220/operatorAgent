"""P0-4 ③ 的可验证部分：两份 prompt 的口径，与一格的记录链（不含模型）。"""
import json

import pytest

from benchmarks.experiments.v08 import ac_plan
from benchmarks.experiments.v08.ac_driver import SurfaceError
from benchmarks.experiments.v08.ac_round import a_cell, arm_prompt

BUDGET = dict(cells=8, max_model_calls=30, max_output_tokens_per_call=8192,
              total_output_token_reservations=245760, max_verifications=8,
              wall_clock_s=2400, final_reserve=1)
DECL = dict(request='fp16 relu 256x256', shapes=[[256, 256]], hardware='mock_gpu',
            dsl='python', precision='fp16', min_speedup=1.0)
GOOD = "import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n"


def test_the_two_prompts_say_different_things_on_purpose():
    a = arm_prompt('A', declaration=DECL, budget=BUDGET)
    c = arm_prompt('C', declaration=DECL, budget=BUDGET)
    # A：普通工具 + check.py，且**不出现**插件词汇
    assert 'check.py' in a and 'tool' in a.lower()
    for word in ('operator_prepare', 'operator_context', 'operator_verify', 'mcp'):
        assert word not in a, f'A 臂 prompt 不该出现 {word}'
    # C：插件工作流写死（这是归因口径的一部分，不是可选提示）
    for step in ('operator_prepare', 'operator_context', 'operator_verify',
                 'verification_purpose="final"'):
        assert step in c
    # 两臂都看到同一份冻结预算与声明，逐字一致
    for prompt in (a, c):
        assert json.dumps(BUDGET, ensure_ascii=False, sort_keys=True) in prompt
        assert json.dumps(DECL, ensure_ascii=False, sort_keys=True) in prompt


def test_the_prompt_budget_matches_the_frozen_plan():
    plan = ac_plan.design()
    budget = plan['budget'] if plan['budget'].get('cells') else dict(BUDGET)
    text = arm_prompt('C', declaration=DECL, budget=budget)
    assert json.dumps(budget, ensure_ascii=False, sort_keys=True) in text


def test_an_unknown_arm_has_no_prompt():
    with pytest.raises(SurfaceError):
        arm_prompt('B', declaration=DECL, budget=BUDGET)


def test_the_a_cell_record_chain_runs_prepare_submit_final(tmp_path):
    cell = a_cell('relu-smoke', DECL, GOOD, host_budget=dict(
        wall_clock_s=600, max_verifications=3, max_model_calls=3,
        max_output_tokens_per_call=1024, total_output_token_reservations=3072),
        backend='fake', run_dir=tmp_path / 'run', model_dispatches=7, verifications=2)
    assert cell['state'] == 'completed' and cell['outcome'] == 'delivered'
    assert cell['final_receipt']['verification_purpose'] == 'final'
    assert cell['final_receipt']['levels']['L2'] is True
    assert cell['model_dispatches'] == 7 and cell['verifications'] == 2
    from pathlib import Path
    assert Path(cell['source']).is_file()          # ac_compare 要按文件重算哈希
    assert Path(cell['source']).read_text().strip() == GOOD.strip()


def test_a_cell_that_never_passes_l2_is_failed_not_delivered(tmp_path):
    broken = "import numpy as np\ndef kernel(X):\n    return X + 1\n"
    cell = a_cell('relu-smoke', DECL, broken, host_budget=dict(
        wall_clock_s=600, max_verifications=3, max_model_calls=3,
        max_output_tokens_per_call=1024, total_output_token_reservations=3072),
        backend='fake', run_dir=tmp_path / 'run')
    assert cell['outcome'] == 'failed' and cell['final_receipt'] is None
    assert cell['levels']['L2'] is False


# ---------------------------------------------------------------------------
# P0-5 的一跳：scripted upstream 把 A 臂会话真跑起来（需要装好的 DSH）
# ---------------------------------------------------------------------------

def _dsh_ready():
    from pathlib import Path
    from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE
    return bool(NODE) and (Path(DSH_ROOT) / 'dsh' / 'lib' / 'bin.js').is_file()


@pytest.mark.skipif(not _dsh_ready(), reason='需要装好的 DSH headless 入口')
def test_the_a_arm_session_reaches_a_final_receipt_through_the_real_process(tmp_path):
    """DSH 子进程 → bash → check.py → supervisor 服务 → 选中源码 → 独立终评 → 一格回执。

    scripted upstream 只发两条消息：一条 bash 命令、一条收尾文本。它不评价模型能力，
    只证明**记录链在真进程里通**（模型调用数因此不是真模型用量，回执里带 scripted_fixture）。
    """
    from benchmarks.experiments.v08.ac_fixture import run_a_cell

    budget = dict(wall_clock_s=600, max_model_calls=30, max_output_tokens_per_call=8192,
                  total_output_token_reservations=245760, max_verifications=4, final_reserve=1)
    record = run_a_cell(DECL, task_id='relu-fixture',
                        candidate_code="import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n",
                        root=tmp_path / 'cell', budget=budget, backend='fake', timeout=180)
    assert record['state'] == 'completed' and record['outcome'] == 'delivered'
    assert record['final_receipt']['verification_purpose'] == 'final'
    assert record['final_receipt']['levels']['L2'] is True
    assert record['final_receipt']['host_task']['remaining_final'] == 0
    assert record['scripted_fixture'] is True and record['fixture_errors'] == []
    from pathlib import Path
    assert Path(record['source']).is_file()


@pytest.mark.skipif(not _dsh_ready(), reason='需要装好的 DSH headless 入口')
def test_the_c_arm_session_delivers_through_the_plugin_and_the_same_final(tmp_path):
    """C 臂走插件的 MCP 工具提交，终评仍由 supervisor 用**同一套** host task 产出。

    方案 1（2026-11-08）：A/C 的终评同源。所以会话里只 prepare + verify，final 不在这里。
    """
    from benchmarks.experiments.v08.ac_fixture import run_c_cell

    budget = dict(wall_clock_s=600, max_model_calls=30, max_output_tokens_per_call=8192,
                  total_output_token_reservations=245760, max_verifications=4, final_reserve=1)
    record = run_c_cell(DECL, task_id='relu-fixture',
                        candidate_code="import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n",
                        root=tmp_path / 'cell', budget=budget, backend='fake', timeout=240)
    assert record['state'] == 'completed' and record['outcome'] == 'delivered'
    receipt = record['final_receipt']
    assert receipt['verification_purpose'] == 'final'      # 与 A 臂同一个读取器
    assert receipt['levels']['L2'] is True
    assert receipt['host_task']['remaining_final'] == 0
    assert record['fixture_errors'] == []


def test_the_prompt_carries_the_same_abi_the_plugin_publishes():
    """A 臂的 ABI 必须**同源**：与插件 `authoritative.interface` 是同一个函数产出的。

    2026-11-08：A 臂写了 169 行正确的 Triton，死在 `Missing 'kernel' function` ——
    prompt 里只有声明、没有调用约定。补 ABI 不是给 A 加能力，是把两臂拉回同一起跑线。
    """
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.generate.generation_context import wrapper_calling_convention as same

    prompt = arm_prompt('A', declaration=DECL, budget=BUDGET)
    contract = freeze_from_task(build_task(DECL))
    expected = same([{t.name: dict(shape=list(t.shape), dtype=t.dtype)
                      for t in contract.cases[0].tensors}])
    import json as _json
    assert _json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=1) in prompt
    assert 'kernel' in prompt and 'do not rename' in prompt


@pytest.mark.skipif(not _dsh_ready(), reason='需要装好的 DSH headless 入口')
def test_the_c_arm_runs_inside_a_repo_blind_sandbox(tmp_path):
    """C 臂与 A 臂必须在**同一个沙箱**里：仓库不可读、插件从 runtime 拷贝加载。

    实测踩出来的四件事（都只在这个形态下出现）：runtime 根目录多一个文件会让
    `verify_runtime` 报 "public runtime drift"；插件行要连 `name`（路径）一起换；
    任务目录要显式放进可写白名单；包按仓库布局找 `remote_scripts/run_kernel.py`。
    """
    from benchmarks.experiments.v08.ac_fixture import run_c_cell

    budget = dict(wall_clock_s=600, max_model_calls=30, max_output_tokens_per_call=8192,
                  total_output_token_reservations=245760, max_verifications=4, final_reserve=1)
    record = run_c_cell(DECL, task_id='relu-fixture',
                        candidate_code="import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n",
                        root=tmp_path / 'cell', budget=budget, backend='fake', timeout=300,
                        sandbox=True)
    assert record['state'] == 'completed' and record['outcome'] == 'delivered'
    # 沙箱真的生效：插件与 operator_agent 都从 runtime 拷贝来的，而不是仓库
    sandbox = tmp_path / 'cell' / 'sandbox' / 'core'
    assert (sandbox / 'operator_agent' / '__init__.py').is_file()
    assert (sandbox / 'remote_scripts' / 'run_kernel.py').is_file()


def test_the_prompt_abi_includes_scalars_like_the_plugin_does():
    """attention 的 `causal` 是按值传的：只按 tensors 生成签名会让裸臂写出 TypeError 的核。

    2026-11-09 实测：裸臂照着我给的 `def kernel(K=None, Q=None, V=None)` 写，
    而调用方发的是 `kernel(K=..., Q=..., V=..., causal=False)` ⇒ L1 直接挂。
    """
    from benchmarks.experiments.v08.ac_round import calling_convention

    attention = dict(request='fp16 attention prefill 1x16x4096x128', shapes=[[1, 16, 4096, 128]],
                     hardware='nvidia_gpu', dsl='triton', precision='fp16', min_speedup=1.0)
    conv = calling_convention(attention)
    assert conv['signature'] == 'def kernel(K, Q, V, causal):'
    assert 'causal' in conv['parameter_names']
    assert conv['keyword_sets'] == [['K', 'Q', 'V', 'causal']]


def test_both_arm_prompts_carry_the_same_anti_cheat_rule():
    from operator_agent.decide.anticheat import ANTICHEAT_POLICY

    for arm in ('A', 'C'):
        prompt = arm_prompt(arm, declaration=DECL, budget=BUDGET)
        assert ANTICHEAT_POLICY in prompt


def test_both_arms_read_the_same_wire_sampling_policy():
    """A/C 是两个独立进程：只在 A 的路径里改策略，C 就会用默认值。

    2026-11-10 实测：弱宿主那轮 A 用 `xhigh`、C 用 `medium`，配对差不可解释。
    """
    import os
    from benchmarks.experiments.phase1 import sampling as wire_sampling
    from benchmarks.experiments.v08.ac_live import _apply_wire_policy

    original = wire_sampling.POLICY['reasoning_effort']
    try:
        os.environ['AC_REASONING_EFFORT'] = 'xhigh'
        assert _apply_wire_policy()['reasoning_effort'] == 'xhigh'
        os.environ['AC_REASONING_EFFORT'] = 'low'
        assert _apply_wire_policy()['reasoning_effort'] == 'low'
    finally:
        wire_sampling.POLICY['reasoning_effort'] = original
        os.environ.pop('AC_REASONING_EFFORT', None)


def test_counts_come_from_the_records_not_from_hand_tallying(tmp_path):
    """`model_dispatches`/`verifications`/墙钟必须由量具产出（评审 P0）。"""
    import json as _json
    from benchmarks.experiments.v08.ac_fixture import _counts

    cell = tmp_path / 'cell'
    (cell / 'cell' / 'outer-requests').mkdir(parents=True)
    (cell / 'run').mkdir(parents=True)
    for index in range(3):
        (cell / 'cell' / 'outer-requests' / f'request-{index:04d}.json').write_text('{}')
    (cell / 'run' / 'verify-1.json').write_text(_json.dumps({'levels': {'L1': True}}))
    (cell / 'run' / 'verify-2.json').write_text(
        _json.dumps({'levels': {'L1': True, 'L2': True}, 'created_at': 10.0}))
    counts = _counts(cell)
    assert counts['dispatches'] == 3 and counts['dispatch_source'] == 'measured'
    assert counts['verifications'] == 2
    assert counts['wall_clock_source'] in (None,
        'recomputed_from_receipt_and_request_timestamps')


def test_failed_cell_early_return_still_meters_dispatches(tmp_path, monkeypatch):
    """早退路径（无提交通过 L2）也必须带计量 —— v6r1-attention-C 曾在此漏计成 None。

    只 mock 重外部件（headless 宿主/网关/监督服务/沙箱），`ht.prepare`、`_counts`、
    verify-*.json 扫描都走真实代码，证明这条早退入口真的过计量。
    """
    import json as _json
    from contextlib import contextmanager
    from types import SimpleNamespace

    from benchmarks.experiments.phase1 import host as p1host, native as p1native
    from benchmarks.experiments.s5 import s5_arm_env, s5_plugin_env
    from benchmarks.experiments.v08 import ac_fixture, ac_live

    class _FakeService:
        def __init__(self, config, contract_hash):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def snapshot(self):
            return dict(selected_source=None, verification_count=1)

    @contextmanager
    def _fake_gateway(config, cell_dir):
        yield SimpleNamespace(base_url='http://127.0.0.1:1/v1')

    root = tmp_path / 'cell'
    (root / 'cell' / 'outer-requests').mkdir(parents=True)
    for index in range(2):
        (root / 'cell' / 'outer-requests' / f'request-{index:04d}.json').write_text('{}')

    monkeypatch.setattr(p1native, 'VerificationService', _FakeService)
    monkeypatch.setattr(p1host, 'model_gateway', _fake_gateway)
    monkeypatch.setattr(s5_plugin_env, 'stage_runtime', lambda p: dict(root=str(p)))
    monkeypatch.setattr(s5_arm_env, 'build', lambda *a, **k: dict(
        work=str(root / 'dsh' / 'work'), home=str(root / 'dsh' / 'home'), env={}, patch=[]))
    monkeypatch.setattr(s5_arm_env, 'start_headless',
                        lambda *a, **k: dict(returncode=0, stderr=''))
    monkeypatch.setattr(ac_fixture, '_stage_sandbox_runtime', lambda runtime, r: 'py:path')
    monkeypatch.setattr(ac_fixture, '_sandbox_c', lambda *a, **k: None)
    monkeypatch.setattr(ac_live, 'select_gpu', lambda *a, **k: {})
    monkeypatch.setattr(ac_live, '_assert_tool_surface', lambda *a, **k: None)

    declaration = dict(request='relu fp32', shapes=[[4, 8]], precision='fp32', dsl='python')
    budget = dict(max_model_calls=20, max_output_tokens_per_call=40960,
                  total_output_token_reservations=819200, max_verifications=8,
                  wall_clock_s=2400, final_reserve=1)
    # prepare 真实跑完后，放一份 L2 未过的提交回执：无可选源码 ⇒ 走早退。
    from operator_agent.app import hitl_tools as ht
    ht.prepare(declaration, root / 'run', host_budget=dict(budget))
    (root / 'run' / 'verify-1.json').write_text(_json.dumps(
        dict(levels={'L1': True, 'L2': False}, source=None)))

    record = ac_live.run_c_live(declaration, task_id='relu-test', root=root, budget=budget,
                                connection=dict(host='h', work_dir='/tmp/w'))
    assert record['outcome'] == 'failed'
    assert record['reason'] == 'no submission passed L2'
    assert record['model_dispatches'] == 2
    assert record['verifications'] == 1
    assert record['metering']['dispatches'] == 'measured'


def test_the_runner_pins_the_declared_gpu_on_both_arms():
    """自动选卡会在卡之间跳（2026-11-10 实测 2/3 两张卡都用过），实验期必须钉住。"""
    import os
    from benchmarks.experiments.v08.ac_live import GPU_PIN, _pin_gpu

    original = {key: os.environ.get(key) for key in
                ('OPERATOR_AGENT_REQUIRED_GPU', 'OPERATOR_AGENT_REQUIRED_GPU_UUID',
                 'OPERATOR_AGENT_REQUIRED_GPU_NAME')}
    try:
        built = dict(patch=[dict(insert=[dict(id='mcp-operator', config=dict(env={}))])])
        _pin_gpu(built)
        assert os.environ['OPERATOR_AGENT_REQUIRED_GPU'] == str(GPU_PIN['index'])
        # C 臂的 MCP 子进程不继承 built['env']，必须写进它自己的条目
        env = built['patch'][0]['insert'][0]['config']['env']
        assert env['OPERATOR_AGENT_REQUIRED_GPU'] == str(GPU_PIN['index'])
        assert env['OPERATOR_AGENT_REQUIRED_GPU_UUID'] == str(GPU_PIN['uuid'])
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_a_busy_gpu_aborts_the_cell_instead_of_running_on_it(monkeypatch):
    """共享机器上"显存空着"≠"卡空着"：GPU 7 上 83 GB 空闲却有别人的 vLLM 占 93%。"""
    from benchmarks.experiments.v08 import ac_live

    busy = dict(index=3, util=93, free_mb=83000, processes=1)
    monkeypatch.setattr(ac_live, 'gpu_state', lambda *a, **k: busy)
    with pytest.raises(SystemExit) as error:
        ac_live.require_idle_gpu(dict(host='h'), None)
    assert '不空闲' in str(error.value) and 'util 93%' in str(error.value)

    monkeypatch.setattr(ac_live, 'gpu_state',
                        lambda *a, **k: dict(index=3, util=0, free_mb=90000, processes=0))
    assert ac_live.require_idle_gpu(dict(host='h'), None)['util'] == 0


def test_gpu_selection_uses_util_and_free_memory_not_process_count(monkeypatch):
    """共享机器上零进程的卡等不到：休眠的 vLLM 不是干扰，util 才是（用户 2026-11-10 选 A）。"""
    from benchmarks.experiments.v08 import ac_live

    idle_with_sleepers = dict(index=2, util=0, free_mb=30000, processes=8, neighbours=[])
    monkeypatch.setattr(ac_live, 'gpu_state', lambda *a, **k: idle_with_sleepers)
    state = ac_live.select_gpu(dict(host='h'))
    assert state['index'] == 2 and state['processes'] == 8

    busy = dict(index=7, util=99, free_mb=83000, processes=1, neighbours=[])
    monkeypatch.setattr(ac_live, 'gpu_state', lambda *a, **k: busy)
    with pytest.raises(SystemExit) as error:
        ac_live.select_gpu(dict(host='h'))
    assert 'util 99%' in str(error.value)


def test_the_stream_deadline_never_exceeds_the_transport_limit(monkeypatch):
    """65536 输出 × 30 s/1k = 1966 s，而 UpstreamTransport 只认 ≤1800 s。

    2026-11-10 实测：不做这个夹取，12 格全部在开跑前 `invalid upstream timeout`。
    """
    import os
    from benchmarks.experiments.phase1 import host
    from benchmarks.experiments.v08.ac_live import _widen_stream_deadline

    monkeypatch.setenv('AC_UPSTREAM_S_PER_1K', '30')
    _widen_stream_deadline()
    deadline = host.upstream_deadline_s(dict(budget=dict(wall_clock_s=2400),
                                             sampling=dict(max_tokens=65536)))
    assert deadline <= 1800


def test_instrument_failures_are_not_recorded_as_candidate_failures():
    """钉卡失败/OOM 是**我们**没跑起来，不是候选写错（2026-11-10 实测两格如此）。"""
    from benchmarks.experiments.v08.ac_fixture import _instrument_failure

    pin = {'final': {'verdicts': [{'message': '... (required GPU selection previously failed)'}]}}
    oom = {'final': {'verdicts': [{'message': 'OutOfMemoryError: CUDA out of memory.'}]}}
    real = {'final': {'verdicts': [{'message': '功能验证失败: 误差 1.16e-03'}]}}
    assert _instrument_failure(pin) and _instrument_failure(oom)
    assert _instrument_failure(real) is None
