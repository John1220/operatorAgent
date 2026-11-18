"""P1-2：真模型跑一对里的一格（A 臂），预算与派发都过 supervisor 的网关。

与 `ac_fixture` 的区别只有三处：上游是真 provider、prompt 是真任务、验证在 H20 上跑。
其余一律复用——`VerificationService` 的预算/判据、`model_gateway` 的调用硬门与 arm log、
`_finish_cell` 的终评。这样"跑真机"与"跑 fixture"走的是同一条记录链。

凭据只从环境读（`ALIYUN_API_KEY`），不落盘、不写进任何回执。
"""
from __future__ import annotations

import json
import os
from subprocess import TimeoutExpired
import sys
from pathlib import Path

#: 臂用的模型。默认走 ssh 隧道到本机回环的 vLLM（H20 上的 Qwen3.8-27B-FP8，不限量），
#: 需要换回云端时设 `AC_MODEL_BASE_URL/AC_MODEL_ID/AC_MODEL_KEY_ENV` 覆盖。
A_MODEL = dict(base_url=os.environ.get('AC_MODEL_BASE_URL',
                                       'http://127.0.0.1:18003/v1'),
               id=os.environ.get('AC_MODEL_ID', 'Qwen3.8-27B'),
               api_key_env=os.environ.get('AC_MODEL_KEY_ENV', 'AC_MODEL_KEY'))


def _disable_plan_mode(built):
    """关掉 DSH 的 plan 模式。

    2026-11-09 实测：C 臂 20 次调用里 **17 次**在 `exit_plan_mode` 上打转，一次都没提交。
    plan 模式要求"先出计划再执行"，而这一轮的任务是"直接写核并提交"。
    """
    if not any(row.get('id') == 'plan-mode' for row in built['patch']):
        built['patch'].append(dict(id='plan-mode', disabled=True))
    return built


#: 钉卡：`SSHExecutor` 在没人指定时会按"最空闲"自动选，实验期间会在卡之间跳。
#: 2026-11-10 用户指定固定在 **GPU 7**（当时唯一空闲、且不是 vLLM 占用的那张）。
GPU_PIN = dict(index=os.environ.get('AC_GPU_INDEX', '3'),
               uuid=os.environ.get('AC_GPU_UUID',
                                   'GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2'),
               name=os.environ.get('AC_GPU_NAME', 'NVIDIA H20'))

#: 预检门槛（2026-11-10 用户选 A）：**util 才是真实干扰信号**。
#: GPU 7 就是 83 GB 空闲却 99% 利用；而"进程数"会把一堆休眠的 vLLM 服务算成干扰，
#: 共享机器上零进程的卡基本等不到。所以判据是 `util <= max_util` 且 `free >= min_free_mb`，
#: 进程数只记录、不否决。
GPU_IDLE = dict(max_util=int(os.environ.get('AC_GPU_MAX_UTIL', '5')),
                # 8192 是用户给的底线；但 4096³ fp32 的峰值（torch baseline + 候选 +
                # 参考各一份）比这个大，取 16 GB 更稳，避免跑一半 OOM。
                min_free_mb=int(os.environ.get('AC_GPU_MIN_FREE_MB', '16384')))

#: 候选卡：按顺序取第一张通过预检的，并把**实际用哪张**写进回执。
GPU_CANDIDATES = [int(item) for item in
                  os.environ.get('AC_GPU_CANDIDATES', '3,2,0').split(',') if item.strip()]


def _pin_gpu(built=None):
    """把钉卡写进环境；C 臂还要写进 MCP 子进程自己的 env（它不继承 built['env']）。"""
    values = {'OPERATOR_AGENT_REQUIRED_GPU': str(GPU_PIN['index']),
              'OPERATOR_AGENT_REQUIRED_GPU_UUID': str(GPU_PIN['uuid']),
              'OPERATOR_AGENT_REQUIRED_GPU_NAME': str(GPU_PIN['name'])}
    os.environ.update(values)
    if built:
        for row in built.get('patch') or ():
            for entry in row.get('insert') or ():
                if entry.get('id') == 'mcp-operator':
                    config = entry.setdefault('config', {})
                    config['env'] = dict(config.get('env') or {}, **values)
    return values


def gpu_state(connection, ssh=None):
    """读一次钉卡的实际状态：(util%, 计算进程数, 空闲 MiB)。读不到就抛错，不猜。"""
    import subprocess

    ssh = dict(ssh or {})
    host = str(ssh.get('host') or connection.get('host'))
    port = ssh.get('port')
    user = ssh.get('user')
    key = ssh.get('key')
    # 有别名（如 `h20`）就让 ssh config 去补端口/用户/密钥；只有显式给了 user 才拼 `user@host`。
    target = f'{user}@{host}' if user else host
    index = str(GPU_PIN['index'])
    command = ('nvidia-smi -i ' + index +
               ' --query-gpu=utilization.gpu,memory.free,uuid,name '
               '--format=csv,noheader,nounits;'
               ' echo ---;'
               ' nvidia-smi -i ' + index +
               ' --query-compute-apps=pid,process_name,used_memory --format=csv,noheader')
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=10']
    if port:
        argv += ['-p', str(port)]
    if key:
        argv += ['-i', str(Path(key).expanduser()), '-o', 'IdentitiesOnly=yes']
    argv += [target, command]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise SystemExit(f'GPU 预检读不到状态: {done.stderr.strip()[:200]}')
    head, _, tail = done.stdout.partition('---')
    parts = [part.strip() for part in head.strip().split(',')]
    try:
        util, free = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        raise SystemExit(f'GPU 预检输出无法解析: {head.strip()[:120]}') from None
    neighbours = [line.strip() for line in tail.splitlines() if line.strip()]
    return dict(index=int(index), util=util, free_mb=free,
                uuid=parts[2] if len(parts) > 2 else None,
                name=parts[3] if len(parts) > 3 else None,
                processes=len(neighbours), neighbours=neighbours[:6])


def _complain(state):
    problems = []
    if state['util'] > GPU_IDLE['max_util']:
        problems.append(f'util {state["util"]}% > {GPU_IDLE["max_util"]}%')
    if state['free_mb'] < GPU_IDLE['min_free_mb']:
        problems.append(f'空闲显存 {state["free_mb"]} MiB < {GPU_IDLE["min_free_mb"]} MiB')
    return problems


def select_gpu(connection, ssh=None):
    """按候选顺序取第一张通过预检的卡；一张都不行就中止（不硬上）。"""
    rejected = []
    for index in GPU_CANDIDATES:
        GPU_PIN['index'] = str(index)
        state = gpu_state(connection, ssh)
        problems = _complain(state)
        if not problems:
            GPU_PIN['uuid'] = state.get('uuid') or GPU_PIN['uuid']
            GPU_PIN['name'] = state.get('name') or GPU_PIN['name']
            state['candidates'] = list(GPU_CANDIDATES)
            return state
        rejected.append(f'GPU{index}: ' + '; '.join(problems))
    raise SystemExit('没有可用 GPU，中止本格: ' + ' | '.join(rejected))


def use_pinned_gpu(connection, pinned, ssh=None):
    """用冻结件里钉死的 GPU；不做候选自动选择，UUID 不符直接中止。

    v6 的 `select_gpu` 会按 GPU_CANDIDATES 取第一张空闲卡 —— 那让"同预算同卡"只
    在运行时才碰巧成立。融合 A/C 轮把 index/uuid/name 冻在计划里，实际回执必须对上。
    """
    if not isinstance(pinned, dict) or not pinned.get('uuid'):
        raise SystemExit('冻结计划没有可用的 GPU 身份（execution.gpu）')
    GPU_PIN['index'] = str(pinned.get('index'))
    GPU_PIN['uuid'] = str(pinned['uuid'])
    GPU_PIN['name'] = str(pinned.get('name') or 'NVIDIA H20')
    state = gpu_state(connection, ssh)
    if state.get('uuid') != pinned['uuid']:
        raise SystemExit(f"冻结 GPU UUID 不符: plan={pinned['uuid']} actual={state.get('uuid')}")
    problems = _complain(state)
    if problems:
        raise SystemExit('冻结 GPU 当前不可用: ' + '; '.join(problems))
    return state


def require_idle_gpu(connection, ssh=None):
    """兼容旧调用：检查当前钉住的那张卡。"""
    state = gpu_state(connection, ssh)
    problems = _complain(state)
    if problems:
        raise SystemExit(f'GPU {state["index"]} 不空闲，中止本格: ' + '; '.join(problems))
    return state


def _assert_budget_consistent(budget):
    """线上输出上限必须 ≤ 账本每次预留，否则网关会在**第一步**就把请求拒掉。

    2026-11-08 的坑：线上策略 `max_tokens` 被压到 8192（账本也 8192 才"自洽"），
    而 harness 的口径是 32768/40960 —— 那一步把每个模型的输出截了一半，
    C 臂第一轮 thinking 就把 8192 吃满（`reason=max-tokens`、0 提交）。
    """
    from benchmarks.experiments.phase1 import sampling

    wire = int(sampling.POLICY['max_tokens'])
    reserve = int(budget['max_output_tokens_per_call'])
    if wire > reserve:
        raise SystemExit(f'线上 max_tokens={wire} 超过账本每次预留 {reserve}：网关会拒请求')
    if int(budget['total_output_token_reservations']) < int(budget['max_model_calls']) * reserve:
        raise SystemExit('累计预留小于 调用数 × 每次预留：预算不自洽')


def repair_workflow(arm, *, run_dir, ssh, budget, seed_receipt):
    """修复任务的**工作流**：两边都是「先原样复现 → 再看诊断 → 再改」。

    2026-11-18 的 C 臂修复试跑仍在 `operator_context(phase="generate")`：那一步发的是
    「从零写一个核」的材料（生成规约、技能、配方），而 repair 轮需要的材料是
    「这份源码为什么挂」。所以这里把修复阶段的入口写死成：
    先提交**未修改**的 `start_kernel.py`（复现，产生 round 1 的失败回执），
    再 `operator_context(phase="repair", verification_round=1)` 取绑定该源码+结果的诊断。
    """
    facts = _receipt_facts(seed_receipt)
    head = ("## 这是修复任务\n"
            "工作目录里的 `start_kernel.py` 就是当前**失败**的实现——不要重写结构，先修它。\n"
            "验收条件（权威）：修好的源码必须在**同一份冻结契约与 32 条套件**上完整 L2 通过，"
            "并再通过一次独立终评（`purpose=final`）的 L2；**不比性能、不要求融合结构**。"
            "契约、套件、容差都不许改。\n"
            "已知故障（来自同一冻结契约的 L2 回执）：\n" + _json_dump(facts) + "\n")
    if arm == 'C':
        body = ("工作流（按顺序）：\n"
                "1. 先**原样不改**提交 `start_kernel.py` 复现失败："
                f'operator_verify(run_dir="{run_dir}", code=<start_kernel.py 的内容>, '
                'candidate_origin="outer", backend="ssh", '
                f'host="{ssh.get("host")}", port={ssh.get("port") or 22}, '
                f'user="{ssh.get("user") or "root"}", '
                f'key_path="{Path(ssh.get("key") or "").expanduser()}", stop_after="L2")；'
                "这一步只为拿到 round 1 的失败回执。\n"
                f'2. 调 operator_context(run_dir="{run_dir}", phase="repair", '
                'verification_round=1, backend="ssh") 取绑定这份源码与这次失败的诊断材料；'
                "它是**假设**，不是指令。\n"
                "3. 改源码，再 `operator_verify(...)` 提交；按回包继续改，"
                f"最多 {budget['max_model_calls']} 次模型调用，不要超。\n"
                "4. 终评由 supervisor 做，你不用调 final。\n")
    else:
        body = ("工作流：读 `start_kernel.py`，改好后用 `check.py --code-file <你的文件>` 提交；"
                "按回包继续改，最多 " + str(budget['max_model_calls']) + " 次模型调用。\n")
    tail = ("不要重写整个实现、不要在思考里整段复述源码（会烧光输出预算），"
            "只回复必要的说明。\n")
    return head + body + tail


#: 会话因**设施**而中断的标记：这些不是候选结果，必须记 unknown。
#: 2026-11-18 修复轮：C 臂两次都是 `Stream ended without finish_reason`（上游流时限），
#: 却被记成 `failed / no submission passed L2` —— 那是把仪器故障算成候选失败。
_SESSION_INSTRUMENT_MARKERS = ('Stream ended without finish_reason',
                               'stream idle timeout',
                               'invalid upstream timeout',
                               'TRANSPORT:',
                               'ECONNRESET',
                               'socket hang up')


def session_instrument_failure(response):
    """这一次会话是被设施打断的吗？是就返回原因串，否则 None。"""
    if not isinstance(response, dict):
        return None
    text = str(response.get('stderr') or '')
    for event in response.get('events') or ():
        if isinstance(event, dict) and event.get('type') in ('final', 'status', 'error'):
            text += ' ' + json.dumps(event, ensure_ascii=False)
    for marker in _SESSION_INSTRUMENT_MARKERS:
        if marker in text:
            return marker
    return None


def _receipt_facts(seed_receipt):
    """从原始失败回执里抽出**这条样本**的故障事实（具名用例 + 误差）。"""
    if not seed_receipt:
        return {}
    import json as _json
    receipt = _json.load(open(seed_receipt, encoding="utf-8"))
    for verdict in receipt.get("verdicts") or []:
        if verdict.get("level") != "L2":
            continue
        cases = ((verdict.get("metrics") or {}).get("correctness") or {}).get("cases") or []
        named = []
        for row in cases:
            metrics = row.get("metrics") or {}
            if row.get("verdict") == "failed":
                named.append(dict(case_id=row.get("case_id"), verdict="failed",
                                  max_error=metrics.get("max_error"),
                                  matched_ratio=metrics.get("matched_ratio"),
                                  reason=str(row.get("reason"))[:300]))
            elif row.get("verdict") == "passed":
                named.append(dict(case_id=row.get("case_id"), verdict="passed"))
        return dict(contract_hash=receipt.get("contract_hash"),
                    summary=str(verdict.get("message"))[:600], cases=named)
    return {}


def _json_dump(value):
    import json as _json
    return _json.dumps(value, ensure_ascii=False, indent=1)


def _assert_tool_surface(root, arm, expected):
    """会话后核对**第一次请求实际拿到的工具面**；缺关键工具就判失败，不许静默。

    2026-11-09：C 臂的工具列表里一个 `mcp__operator__*` 都没有，它跑成了第二个裸 DSH，
    而整轮被当成 A/C 对照记了下来。这一条就是为了让那种情况当场红掉。
    """
    import glob as _glob
    import json as _json

    paths = sorted(_glob.glob(str(Path(root) / 'cell' / 'outer-requests' / 'request-*.json')))
    if not paths:
        raise SystemExit(f'{arm}: 没有任何模型请求记录，工具面无法核对')
    names = {row['function']['name'] for row in
             (_json.load(open(paths[0])).get('tools') or [])}
    missing = [name for name in expected if name not in names]
    if missing:
        raise SystemExit(f'{arm}: 工具面缺少 {missing}；实际只有 {sorted(names)[:10]}')


def _apply_wire_policy():
    """两臂**必须**走同一份线上采样策略。

    2026-11-10 实测：`reasoning_effort` 的补丁原先只写在 `run_a_live` 里，
    而 A/C 是两个独立进程 ⇒ C 那一格用的是默认 `medium`，A 用的是 `xhigh`。
    两臂采样身份不同，那一轮的配对差就不可解释了。现在两臂都从这里过一遍。
    """
    from benchmarks.experiments.phase1 import sampling as wire_sampling

    requested = os.environ.get('AC_REASONING_EFFORT')
    if requested:
        wire_sampling.POLICY['reasoning_effort'] = requested
    # 输出上限也要可配：27B 在 xhigh 下会把整个 kernel 写进思考，32768 直接撞顶
    # （2026-11-10 实测：attention/A 第 6 次调用 654 s 写满 32768 后 `max-tokens` 收场）。
    cap = os.environ.get('AC_WIRE_MAX_TOKENS')
    if cap:
        wire_sampling.POLICY['max_tokens'] = int(cap)
    return dict(wire_sampling.policy())


def _widen_stream_deadline():
    """把流式截止从 16 s/1k tokens 放宽到可配（默认 24 s/1k）。

    本机 vLLM 实测 ~53 tok/s，而 16 s/1k 隐含要求 ≥62.5 tok/s ⇒ 只要模型一路写到
    32k 上限就必被掐（2026-11-09 实测：单次流 524 s / 27.8k tokens 后
    `Stream ended without finish_reason`）。仍受墙钟预算约束。

    **两个臂都要调**：第一次只加在 A 上，C 那次仍然死在 524 s。
    """
    from benchmarks.experiments.phase1 import host as _host
    from benchmarks.experiments.phase1 import sampling as _sampling

    # 24 s/1k 在 32768 上限下只有 786 s；35 s/1k ≈1147 s。2026-11-18 修复轮实测：
    # A 臂第 3 次调用正好在 786.4 s 被掐、C 臂在 1146.9 s 被掐（都已流出 1.6/4.9 MB，
    # 仍在写）。本机 vLLM 在长上下文下明显慢于 53 tok/s。
    # 取 50 s/1k ≈ 1638 s，仍在传输层 1800 s 硬上限之内，且最坏 1 次长调用后
    # 还剩约 760 s 墙钟给其余步骤。
    per_1k = float(os.environ.get('AC_UPSTREAM_S_PER_1K', '50'))

    # `UpstreamTransport` 硬性拒绝 > 1800 s 的 timeout（MAX_WALL_SECONDS）；
    # 65536 输出 × 30 s/1k = 1966 s 会直接 `invalid upstream timeout`（2026-11-10 实测）。
    from benchmarks.experiments.s5.s5_gateway_transport import MAX_WALL_SECONDS

    def _deadline(config):
        cap = _sampling.output_cap(config)
        return min(max(120.0, per_1k * cap / 1000.0), config['budget']['wall_clock_s'],
                   float(MAX_WALL_SECONDS))

    _host.upstream_deadline_s = _deadline


def run_a_live(declaration, *, task_id, root, budget, connection, model=None, wall_clock_s=None,
               arm='A', timeout_s=None, pinned_gpu=None, seed_source=None, seed_receipt=None,
               acceptance='delivery'):
    from benchmarks.experiments.phase1.host import model_gateway
    from benchmarks.experiments.phase1.native import VerificationService
    from benchmarks.experiments.s5 import s5_arm_env
    from operator_agent.app import hitl_tools as ht
    from operator_agent.execute import s5_shared_budget as shared
    from operator_agent.execute.task_clock import monotonic

    from .ac_fixture import _finish_cell
    from .ac_round import arm_prompt

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / 'run').resolve()
    wall = int(wall_clock_s or budget['wall_clock_s'])
    model = dict(model or A_MODEL)
    _assert_budget_consistent(budget)
    gpu = (use_pinned_gpu(connection, pinned_gpu) if pinned_gpu
           else select_gpu(connection))     # A 臂没有显式 ssh 参数，按 connection 读
    _pin_gpu()

    prepared = ht.prepare(declaration, run_dir, host_budget=dict(budget))
    ledger = (root / 'budget.sqlite').resolve()
    shared.create(ledger, task_id='ac-' + prepared['contract_hash'][:16],
                  max_calls=int(budget['max_model_calls']),
                  max_output_tokens=int(budget['max_output_tokens_per_call']),
                  total_output_tokens=int(budget['total_output_token_reservations']),
                  wall_clock_s=wall)
    service = VerificationService(
        dict(run_dir=str(run_dir), backend='ssh', connection=dict(connection),
             declaration=declaration,
             budget=dict(wall_clock_s=wall, max_llm_calls=int(budget['max_model_calls']),
                         oracle_total=int(budget['max_verifications'])),
             deadline_monotonic=monotonic() + wall),
        prepared['contract_hash'])
    # 线上 wire 策略是 phase1 的固定常量（`sampling.POLICY`，max_tokens=32768）。本轮计划
    # 冻的是**每次输出 8192**，两者冲突时不能各说各话：把 max_tokens 改成计划的 8192，
    # 其余几格（temperature/n/thinking/reasoning_effort）保持 phase1 取值，两臂相同。
    from benchmarks.experiments.phase1 import sampling as wire_sampling

    assert wire_sampling.POLICY['max_tokens'] <= int(budget['max_output_tokens_per_call']), (
        '线上 max_tokens 不能超过账本每次预留')
    # 本机 vLLM（Qwen3.8-27B）实测：`reasoning_effort` 的 `none` 会**关掉思考**
    # （0.6s/25 tok vs medium 的 2.9s/151 tok），`high` 直接 400。
    # 它是 harness 本来就允许的字段，所以不需要放开 `chat_template_kwargs` 那条禁令。

    wire_policy = _apply_wire_policy()      # **必须在建 gateway_config 之前**：
    gateway_config = dict(model=model,     # 否则 config['sampling'] 会是打补丁前的旧副本
                          
                          budget=dict(max_llm_calls=int(budget['max_model_calls']),
                                      wall_clock_s=wall),
                          shared_ledger=str(ledger),
                          shared_task_id='ac-' + prepared['contract_hash'][:16],
                          run_dir=str(run_dir), backend='ssh',
                          sampling=wire_policy)

    from benchmarks.experiments.s5 import s5_plugin_env, s5_seatbelt

    runtime = s5_plugin_env.stage_runtime(root / 'runtime')
    _widen_stream_deadline()
    with service, model_gateway(gateway_config, root / 'cell') as gateway:
        built = s5_arm_env.build(
            arm, base_url=gateway.base_url, model=model['id'], tools=True,
            max_tokens=int(budget['max_output_tokens_per_call']), context_window=65536,
            development=service.client_config(), arm_dir=root / 'dsh')
        for folder in (built['work'], built['home']):
            Path(folder).mkdir(parents=True, exist_ok=True)
        if arm == 'C':
            # 只有 C 臂的 MCP 子进程要 import operator_agent；A 臂的 check.py 是 stdlib 客户端，
            # 给 A 设 PYTHONPATH 等于把仓库路径写进它的环境（2026-11-08 实测：模型顺着它去读本仓）。
            built['env'] = dict(built['env'],
                                PYTHONPATH=str(Path(__file__).resolve().parents[3]))
        _disable_plan_mode(built)
        _pin_gpu(built)
        # 沙箱由下面的 `_sandbox` 统一写（它允许额外读根：A 臂需要 `.venv` 里的解释器）。
        # 这里**不能**再调 `s5_seatbelt.configure` —— 两个都会写 isolation.sb，
        # 第二个 `open('x')` 直接 FileExistsError（实测）。
        # 沙箱里 PATH 只有系统目录，裸 `python` 不存在；`check.py` 是 stdlib 客户端，
        # 用 `/usr/bin/python3` 就够（venv 的 python 在仓库里，沙箱读不到）。
        import shutil as _shutil
        work = Path(built['work'])
        work.mkdir(parents=True, exist_ok=True)
        _shutil.copyfile(Path(runtime['root']) / 'check.py', work / 'check.py')
        if seed_source is not None:
            # 修复任务的起点：坏核原样放进工作目录，两臂拿同一份字节。
            _shutil.copyfile(seed_source, work / 'start_kernel.py')
        # `/usr/bin/python3` 在这台机器上是 xcode-select 的壳，DSH 的内层沙箱下直接报
        # `unable to read data link`。用仓库 venv 的 python（绝对路径），并把
        # **`.venv` 这一个子目录**放进读白名单——仓库源码仍然读不到。
        interpreter = str(Path(sys.executable).resolve())
        if seed_source is not None:
            # 修复格：不给“从零实现”的整段 prompt，只给修复工作流 + 故障事实 + 提交命令。
            prompt = (repair_workflow('A', run_dir=run_dir, ssh={}, budget=budget,
                                      seed_receipt=seed_receipt)
                      + '\n提交命令：`' + interpreter + ' check.py --code-file <你的文件>`。\n')
        else:
            prompt = arm_prompt(arm, declaration=declaration, budget=budget).replace(
                'python check.py', f'{interpreter} check.py')
        if True:  # A 臂同样进沙箱：只放行 `.venv` 与臂目录，仓库源码读不到
            from .ac_fixture import _sandbox_c as _sandbox
            _sandbox(built, runtime=runtime, root=root,
                     extra_roots=[Path(interpreter).parent.parent],
                     extra_files=[Path(interpreter)])
        (root / 'prompt.txt').write_text(prompt, encoding='utf-8')
        # 墙钟到点是**正常终止**（记 deadline），不是崩溃：2026-11-18 实测 A 臂在预算
        # 用尽后仍会循环重试 check.py，`start_headless` 抛 TimeoutExpired 会把整格丢成
        # crash/unknown —— 而预登记要求"用完墙钟仍未达标 ⇒ not_delivered（失败）"。
        try:
            response = s5_arm_env.start_headless(built, prompt, timeout=timeout_s or wall,
                                                 capture_dir=root / 'host-output')
            deadline_hit = False
        except TimeoutExpired:
            response = dict(returncode=None, stderr='wall_clock_exhausted', events=[],
                            output_capture={})
            deadline_hit = True
        state = service.snapshot()

    instrument = session_instrument_failure(response)
    selected = state.get('selected_source')
    _assert_tool_surface(root, arm, ('bash',))
    (root / 'session.json').write_text(json.dumps(
        dict(returncode=response.get('returncode'), stderr=(response.get('stderr') or '')[-4000:],
             selected=selected, verifications=state.get('verification_count')),
        ensure_ascii=False, indent=2), encoding='utf-8')
    if not selected:
        from .ac_fixture import _counts
        counted = _counts(root)
        # 优先级：设施中断 ⇒ unknown；正常用完墙钟 ⇒ failed（预登记口径）。
        state_name = ('crashed' if instrument else
                      'deadline' if deadline_hit else 'crashed')
        return dict(task_id=task_id, arm=arm, state=state_name,
                    final_receipt=None, source=None,
                    outcome='unknown' if (instrument or not deadline_hit) else 'failed',
                    instrument_failure=instrument,
                    reason=(f'instrument: {instrument}' if instrument else
                            ('wall clock exhausted without a full-L2 submission'
                             if deadline_hit else 'no submission reached the supervisor')),
                    scripted_fixture=False, model_dispatches=counted['dispatches'],
                    verifications=counted['verifications'],
                    metering=dict(dispatches=counted['dispatch_source'],
                                  verifications='measured' if counted['verifications'] else None,
                                  wall_clock=None))
    record = _finish_cell(task_id, arm, selected, root=root, declaration=declaration,
                          acceptance=acceptance,
                          budget=budget, backend='ssh', dispatches=None, errors=[],
                          returncode=response.get('returncode'), connection=connection)
    record['gpu'] = gpu          # 实际用的卡 + 卡上邻居：杜绝"以为在测 A 卡其实在 B 卡"
    return record


def run_c_live(declaration, *, task_id, root, budget, connection, model=None,
               wall_clock_s=None, timeout_s=None, ssh=None, pinned_gpu=None,
               seed_source=None, seed_receipt=None, acceptance='delivery'):
    """C 臂一格（真模型 + 沙箱 + 20 次预算）。

    与 A 臂共用同一套 supervisor 件：`prepare(host_budget)`、`VerificationService`、
    `model_gateway`、`_finish_cell`。差别只在宿主那侧用插件工具提交。
    """
    from benchmarks.experiments.phase1.host import model_gateway
    from benchmarks.experiments.phase1.native import VerificationService
    from benchmarks.experiments.s5 import s5_arm_env, s5_plugin_env
    from operator_agent.app import hitl_tools as ht
    from operator_agent.execute import s5_shared_budget as shared
    from operator_agent.execute.task_clock import monotonic

    from .ac_fixture import _finish_cell, _sandbox_c, _stage_sandbox_runtime
    from .ac_round import arm_prompt

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / 'run').resolve()
    wall = int(wall_clock_s or budget['wall_clock_s'])
    model = dict(model or A_MODEL)
    _assert_budget_consistent(budget)
    gpu = (use_pinned_gpu(connection, pinned_gpu, ssh) if pinned_gpu
           else select_gpu(connection, ssh))
    _pin_gpu()

    prepared = ht.prepare(declaration, run_dir, host_budget=dict(budget))
    ledger = (root / 'budget.sqlite').resolve()
    shared.create(ledger, task_id='ac-' + prepared['contract_hash'][:16],
                  max_calls=int(budget['max_model_calls']),
                  max_output_tokens=int(budget['max_output_tokens_per_call']),
                  total_output_tokens=int(budget['total_output_token_reservations']),
                  wall_clock_s=wall)
    service = VerificationService(
        dict(run_dir=str(run_dir), backend='ssh', connection=dict(connection),
             declaration=declaration,
             budget=dict(wall_clock_s=wall, max_llm_calls=int(budget['max_model_calls']),
                         oracle_total=int(budget['max_verifications'])),
             deadline_monotonic=monotonic() + wall),
        prepared['contract_hash'])
    # 线上策略保持 harness 口径（32768），**不要**再按预留值改小：
    # 预留着的是账本额度，不是"截断点"。
    wire_policy = _apply_wire_policy()      # 与 A 臂同一个入口、同一份策略
    gateway_config = dict(model=model,
                          budget=dict(max_llm_calls=int(budget['max_model_calls']),
                                      wall_clock_s=wall),
                          shared_ledger=str(ledger),
                          shared_task_id='ac-' + prepared['contract_hash'][:16],
                          run_dir=str(run_dir), backend='ssh',
                          sampling=wire_policy)

    runtime = s5_plugin_env.stage_runtime(root / 'runtime')
    _widen_stream_deadline()
    with service, model_gateway(gateway_config, root / 'cell') as gateway:
        # **不给 `development`**：那份 S5 开发客户端会让 MCP 服务器进入 supervised 模式，
        # 工具面里就没有 `operator_context`（实测 C4：只有 prepare/verify/capabilities）。
        # C 臂本来也不用 check.py —— 它的提交走插件工具，终评由 supervisor 做。
        # 归档那对把 `tool-bash`/`tool-fs` **关掉**了：插件臂一旦有 shell，就会绕开插件
        # 自己 ssh 上卡（C6 实测过）。C 臂的 treatment 就是"只有插件工具"，
        # 代码通过 `operator_verify(code=...)` 以内联字符串提交。
        built = s5_arm_env.build(
            'C', base_url=gateway.base_url, model=model['id'], tools=False,
            max_tokens=int(budget['max_output_tokens_per_call']), context_window=65536,
            arm_dir=root / 'dsh')
        for folder in (built['work'], built['home']):
            Path(folder).mkdir(parents=True, exist_ok=True)
        python_path = _stage_sandbox_runtime(runtime, root)
        # ssh **在臂自己的 MCP 子进程里发起**（与归档那对同一种机制）。沙箱把 HOME 指到
        # 臂目录，所以不能依赖 `~/.ssh/config` 的 `Host h20` 别名 —— 显式给
        # host/port/user/key，并让沙箱只放行那一把私钥。
        ssh = dict(ssh or {})
        _disable_plan_mode(built)
        built['env'] = dict(built['env'], PYTHONPATH=python_path,
                            OPERATOR_AGENT_WORKFLOW='host',
                            REMOTE_SSH_HOST=str(ssh.get('host') or connection.get('host')),
                            REMOTE_WORK_DIR=str(connection.get('work_dir')),
                            OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1')
        for source, target in (('port', 'REMOTE_SSH_PORT'), ('user', 'REMOTE_SSH_USER'),
                               ('key', 'REMOTE_SSH_KEY')):
            if ssh.get(source):
                built['env'][target] = str(ssh[source])
        for row in built['patch']:
            for entry in row.get('insert') or ():
                config = entry.get('config') or {}
                if isinstance(config.get('env'), dict):
                    config['env']['PYTHONPATH'] = python_path
                if entry.get('id') == 'mcp-operator':
                    # MCP 子进程的环境由这一格 config.env **自带**，不继承 built['env']：
                    # 只写 built['env'] 时 `operator_verify(backend="ssh")` 会报 host_required。
                    merged = dict(config.get('env') or {})
                    for source, target in (('port', 'REMOTE_SSH_PORT'), ('user', 'REMOTE_SSH_USER'),
                                           ('key', 'REMOTE_SSH_KEY')):
                        if ssh.get(source):
                            merged[target] = str(ssh[source])
                    merged['REMOTE_SSH_HOST'] = str(ssh.get('host') or connection.get('host'))
                    merged['REMOTE_WORK_DIR'] = str(connection.get('work_dir'))
                    merged['OPERATOR_AGENT_EXISTING_REMOTE_ONLY'] = '1'
                    merged['OPERATOR_AGENT_WORKFLOW'] = 'host'
                    config['env'] = merged
                    # **工具调用超时**：`s5_arm_env.build` 在没有 `development` 时给 60 s，
                    # 而 attention 的 L2 是 8 条 case（S=4096，H20 上数分钟）⇒ 一律
                    # `Error: Request timed out`，宿主永远拿不到 L2 裁决（2026-11-09 实测
                    # C 臂 7 次 verify 里 5 次这样死掉）。按墙钟预算给足。
                    config['toolCallTimeoutMs'] = int(max(600, wall) * 1000)
                if entry.get('id') == 'operator-tools':
                    entry['name'] = str(Path(runtime['root']) / 'plugin' / 'index.mjs')
                    config['command'] = str(Path(runtime['root']) / 'plugin' / 'index.mjs')
        _pin_gpu(built)
        key_file = ssh.get('key')
        _sandbox_c(built, runtime=runtime, root=root,
                   extra_roots=[Path(python_path.split(os.pathsep)[0]).parent,
                                Path(python_path.split(os.pathsep)[1])],
                   extra_files=[Path(key_file).expanduser()] if key_file else [])
        # 精简版：契约已由 supervisor 冻结，C 臂不需要再 prepare，也不需要把声明与预算
        # 的整段 JSON 塞进 prompt（那会让第一轮 thinking 直接吃掉 8192 输出预算，
        # 实测 `reason: max-tokens`、7 次调用 0 提交）。给它的只有：任务目录、ABI、
        # 提交命令、预算数字。
        abi_text = json.dumps(__import__('benchmarks.experiments.v08.ac_round',
                                         fromlist=['calling_convention'])
                              .calling_convention(declaration), ensure_ascii=False)
        prompt = (
            '你在做一个 Triton 算子任务。任务目录（契约已冻结，原样使用）：' + str(run_dir) + '\n'
            '调用约定（ABI，权威）：' + abi_text + '\n'
            '工作流：先 `operator_context(run_dir=..., phase="generate")` 取事实与材料；'
            '然后你自己写 kernel 源码，**按下面的完整参数**提交（host/port/user/key_path 必须显式给：'
            '工具入口不读环境变量，只写 backend 会报 host_required）：'
            f'operator_verify(run_dir="{run_dir}", code=<源码>, backend="ssh", '
            f'host="{ssh.get("host")}", port={ssh.get("port") or 22}, '
            f'user="{ssh.get("user") or "root"}", key_path="{Path(ssh.get("key") or "").expanduser()}", '
            'stop_after="L2")；失败就按回包改，最多 ' +
            str(budget['max_model_calls']) + ' 次模型调用，不要超。终评由 supervisor 做，你不用调 final。\n'
            '只回复必要的说明，不要长篇思考。')
        if seed_source is not None:
            # C 臂也拿到同一份修前源码（写进它的工作目录，沙箱里只读得到臂目录）。
            import shutil as _shutil
            work_c = Path(built['work'])
            work_c.mkdir(parents=True, exist_ok=True)
            _shutil.copyfile(seed_source, work_c / 'start_kernel.py')
            prompt = (repair_workflow('C', run_dir=run_dir, ssh=ssh,
                                      budget=budget, seed_receipt=seed_receipt)
                      + '\n调用约定（ABI，权威）：' + abi_text + '\n')
        (root / 'prompt.txt').write_text(prompt, encoding='utf-8')
        response = s5_arm_env.start_headless(built, prompt, timeout=timeout_s or wall,
                                             capture_dir=root / 'host-output')
        state = service.snapshot()

    # C 臂的**存在理由**就是那套插件工具；没有它们这一格不是 C 臂。
    _assert_tool_surface(root, 'C', ('mcp__operator__operator_context',
                                     'mcp__operator__operator_verify'))
    selected, last = None, None
    for path in sorted(run_dir.glob('verify-*.json')):
        value = json.loads(path.read_text())
        last = value
        if (value.get('levels') or {}).get('L2') is True and value.get('source'):
            selected = value['source']
    (root / 'session.json').write_text(json.dumps(
        dict(returncode=response.get('returncode'), stderr=(response.get('stderr') or '')[-4000:],
             selected=selected, verifications=state.get('verification_count'),
             last=last), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    if not selected:
        from .ac_fixture import _counts
        counted = _counts(root)
        instrument = session_instrument_failure(response)
        # 没提交 ≠ 候选失败：会话被设施掐断时记 unknown（预登记：只有“正常用完预算
        # 仍未达标”才是 not_delivered）。
        return dict(task_id=task_id, arm='C',
                    state='crashed' if instrument else 'completed',
                    final_receipt=None, source=None,
                    outcome='unknown' if instrument else 'failed',
                    instrument_failure=instrument,
                    reason=(f'instrument: {instrument}' if instrument
                            else 'no submission passed L2'),
                    scripted_fixture=False, last_receipt=last, model_dispatches=counted['dispatches'],
                    verifications=counted['verifications'],
                    metering=dict(dispatches=counted['dispatch_source'],
                                  verifications='measured' if counted['verifications'] else None,
                                  wall_clock=None))
    record = _finish_cell(task_id, 'C', selected, root=root, declaration=declaration,
                          acceptance=acceptance,
                          budget=budget, backend='ssh', dispatches=None, errors=[],
                          returncode=response.get('returncode'), connection=connection)
    record['gpu'] = gpu
    return record


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog='ac_live', description=__doc__.splitlines()[0])
    parser.add_argument('--root', required=True)
    parser.add_argument('--json-out', required=True)
    parser.add_argument('--arm', default='A', choices=['A', 'C'])
    parser.add_argument('--task-id', default='matmul-4096-fp32')
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--ssh-host', default=os.environ.get('AC_SSH_HOST'),
                        help='C 臂自己 ssh 时的主机（沙箱里没有 ~/.ssh/config，必须显式）')
    parser.add_argument('--ssh-port', default=os.environ.get('AC_SSH_PORT'))
    parser.add_argument('--ssh-user', default=os.environ.get('AC_SSH_USER'))
    parser.add_argument('--ssh-key', default=os.environ.get('AC_SSH_KEY'))
    parser.add_argument('--plan', default=None, help='冻结计划 JSON；给了就按其任务声明与预算跑')
    parser.add_argument('--seed-source', default=None,
                        help='修复任务：把这份修前源码放进臂的工作目录（start_kernel.py）')
    parser.add_argument('--seed-receipt', default=None,
                        help='修复任务：原始失败回执，作为权威故障事实发给模型')
    parser.add_argument('--acceptance', default='delivery', choices=['delivery', 'repair'],
                        help='delivery=达标交付（L2+valid+1.5x+单核）；repair=修好（完整 L2+独立终评 L2）')
    args = parser.parse_args(argv)

    # 凭据只从环境读。本地 vLLM 用占位 key（`EMPTY`），云端换 provider 时改这里即可。
    key_env = os.environ.get('AC_MODEL_KEY_ENV', 'AC_MODEL_KEY')
    if not os.environ.get(key_env):
        raise SystemExit(f'{key_env} 不在环境里：凭据只从环境读，不落盘')
    if args.plan:
        # 声明与预算一律从冻结计划取，避免"手写声明"与计划漂移。
        plan = json.loads(Path(args.plan).read_text())
        task = next((t for t in plan['tasks'] if t['task_id'] == args.task_id), None)
        if task is None:
            raise SystemExit(f'计划里没有任务 {args.task_id}')
        declaration = dict(task['declaration'])
        pinned_gpu = (plan.get('execution') or {}).get('gpu')
        budget = dict(wall_clock_s=plan['budget']['wall_clock_s'],
                      max_model_calls=plan['budget']['max_model_calls'],
                      max_output_tokens_per_call=plan['budget']['max_output_tokens_per_call'],
                      total_output_token_reservations=plan['budget']['total_output_token_reservations'],
                      max_verifications=plan['budget']['max_verifications'],
                      final_reserve=plan['budget']['final_reserve'])
    else:
        pinned_gpu = None
        declaration = dict(request='fp32 4096x4096x4096 matmul, triton',
                           shapes=[[4096, 4096, 4096]], hardware='nvidia_gpu', dsl='triton',
                           precision='fp32', min_speedup=1.0)
    # 缺省预算只在**没有** --plan 时生效；给了计划就必须用计划里的冻结值，
    # 否则"同预算"只是命令行巧合（2026-11-17 评审指出旧代码会覆盖计划预算）。
    if not args.plan:
        from benchmarks.experiments.v08 import ac_plan
        budget = dict(wall_clock_s=2400, max_model_calls=20,
                      max_output_tokens_per_call=ac_plan.OUTPUT_RESERVATION_PER_CALL,
                      total_output_token_reservations=20 * ac_plan.OUTPUT_RESERVATION_PER_CALL,
                      max_verifications=8, final_reserve=1)
    # 按 --arm 分派：这里曾经写死 `run_a_live`，于是 `--arm C` 跑出来的是 A 臂
    # （工具面只有 bash，没有 mcp__operator__*），而回执被记成了 C 臂。
    runner = run_c_live if args.arm == 'C' else run_a_live
    extra = dict(ssh=dict(host=args.ssh_host, port=args.ssh_port, user=args.ssh_user,
                          key=args.ssh_key)) if args.arm == 'C' else {}
    record = runner(declaration, task_id=args.task_id, root=args.root, budget=budget,
                    connection=dict(host=args.host, work_dir=args.work_dir, persistent=True),
                    wall_clock_s=budget['wall_clock_s'], timeout_s=args.timeout,
                    pinned_gpu=pinned_gpu, seed_source=args.seed_source,
                    seed_receipt=args.seed_receipt, acceptance=args.acceptance, **extra)
    # 每格必须自报用的哪个模型：跨格换模型时，交付数不可比，得能查出来。
    record['model'] = os.environ.get('AC_MODEL_ID', A_MODEL['id'])
    record['reasoning_effort'] = os.environ.get('AC_REASONING_EFFORT', 'none')
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + '\n',
                                   encoding='utf-8')
    print(json.dumps({k: record.get(k) for k in ('state', 'outcome', 'reason')}, ensure_ascii=False))
    return 0 if record.get('final_receipt') else 2


if __name__ == '__main__':
    raise SystemExit(main())