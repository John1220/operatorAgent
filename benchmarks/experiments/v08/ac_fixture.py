"""P0-5 的一跳：用**本地 scripted upstream** 把 A 臂会话真的跑起来（不花模型钱）。

它不评价模型能力，只回答一个问题：**记录链在真进程里通不通**——
DSH 子进程 → 普通 bash 工具 → `check.py` → supervisor 的验证服务 → 选中源码 → 独立终评
→ 一格回执。`installed_smoke.py` 已经用同样的手法验证过"装好的插件 + scripted 模型"，
这里换成 A 臂（无插件、有 bash/fs）。

边界：`scripted` 意味着这一格的 `model_dispatches` 不是真模型的用量，回执里必须带
`scripted_fixture=true`，且 `test_only` 由执行上下文承担——它不能当作 A 臂的任何成绩。
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHECKS = ("bash", "shell", "run_shell", "tool-bash")


class FixtureError(RuntimeError):
    pass


def _sse(chunks):
    return (''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
            + 'data: [DONE]\n\n').encode()


def _scripted_server(*, command, requests, errors, model_id):
    """第一轮回一个 bash 调用，之后回一段收尾文本。只做这一件事。"""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path not in ('/v1/chat/completions', '/chat/completions'):
                    raise FixtureError(f'unexpected path {self.path}')
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                index = len(requests)
                names = {row['function']['name'] for row in request.get('tools') or ()}
                tool_messages = [m for m in request['messages'] if m.get('role') == 'tool']
                if index > 1 and tool_messages:
                    delta = dict(content='Candidate submitted; stopping as instructed.')
                    finish = 'stop'
                else:
                    name = next((n for n in CHECKS if n in names), None)
                    if name is None:
                        raise FixtureError(f'A arm exposes no shell tool; saw {sorted(names)[:12]}')
                    delta = dict(tool_calls=[dict(index=0, id=f'fixture-{index}', type='function',
                                                  function=dict(name=name,
                                                                arguments=json.dumps(dict(
                                                                    command=command,
                                                                    description='Submit the candidate to check.py'))))])
                    finish = 'tool_calls'
                chunks = [dict(id=f'fixture-{index}', model=request.get('model') or model_id,
                               choices=[dict(index=0, delta=delta, finish_reason=None)]),
                          dict(id=f'fixture-{index}', model=request.get('model') or model_id,
                               choices=[dict(index=0, delta={}, finish_reason=finish)])]
                payload = _sse(chunks)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:                     # noqa: BLE001 — 回执要能读到原因
                errors.append(f'{type(error).__name__}: {error}')
                self.send_response(500)
                self.end_headers()
    return Handler


def run_a_cell(declaration, *, task_id, candidate_code, root, budget, backend='fake',
               connection=None, model_id='ac-fixture', timeout=180):
    """跑一格 A 臂（scripted 模型），返回 `ac_driver.cell` 形状的 dict。"""
    from benchmarks.experiments.phase1.native import VerificationService
    from benchmarks.experiments.s5 import s5_arm_env, s5_plugin_env
    from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE
    from operator_agent.app import hitl_tools as ht

    from .ac_driver import cell, final_evaluation

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / 'run'
    connection = dict(connection or {})
    prepared = ht.prepare(declaration, run_dir, host_budget=dict(budget))
    from operator_agent.execute.task_clock import monotonic
    wall = int(budget['wall_clock_s'])
    service_config = dict(run_dir=str(run_dir), backend=backend, declaration=declaration,
                          connection=connection,
                          execution_context=dict(cell_id=task_id, arm='A', fixture=True),
                          # `deadline_monotonic` 是 native 服务的必填项：缺了它 develop()
                          # 会 KeyError，服务回 400，check.py 只报
                          # `development_transport_failed`（2026-11-17 fixture 实测）。
                          deadline_monotonic=monotonic() + wall,
                          budget=dict(wall_clock_s=wall,
                                      max_llm_calls=int(budget['max_model_calls']),
                                      oracle_total=int(budget['max_verifications'])))
    service = VerificationService(service_config, prepared['contract_hash'])

    requests, errors = [], []
    candidate = root / 'candidate.py'
    candidate.write_text(candidate_code, encoding='utf-8')
    # 沙箱里的 PATH 只有 /usr/local/bin:/usr/bin:/bin —— 裸 `python` 不存在（实测 exit 127）。
    # phase1 的 A 臂 prompt 也是用解释器的**绝对路径**，这里照同一条规矩。
    command = f'{shlex.quote(str(Path(sys.executable).resolve()))} check.py --code-file {candidate.name}'

    with service:
        handler = _scripted_server(command=command, requests=requests, errors=errors,
                                   model_id=model_id)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runtime = s5_plugin_env.stage_runtime(root / 'runtime')
            built = s5_arm_env.build(
                'A', base_url=f'http://127.0.0.1:{server.server_port}/v1',
                model=model_id, tools=True, max_tokens=1024, context_window=65536,
                development=service.client_config(), arm_dir=root / 'dsh')
            # 本机 DSH 的 workspace-write 沙箱后端不可用（test_only fixture，不是公平性
            # 声明）；fixture 只用 danger-full-access 把 check.py 跑通。
            built['env']['DSH_PERMISSION_MODE'] = 'danger-full-access'
            _widen_tool_timeout(built, budget['wall_clock_s'])
            work = Path(built['work'])
            work.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(runtime['root']) / 'check.py', work / 'check.py')
            shutil.copyfile(candidate, work / candidate.name)
            response = s5_arm_env.start_headless(built, 'Run the submission command.',
                                                 timeout=timeout, capture_dir=root / 'host-output')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        state = service.snapshot()

    selected = state.get('selected_source')
    if not selected:
        return dict(task_id=task_id, arm='A', state='crashed', final_receipt=None,
                    source=None, outcome='unknown',
                    reason=f'no submission reached the supervisor: {errors or response.get("stderr")}',
                    scripted_fixture=True, model_dispatches=len(requests))
    # 与 C 臂同一套收尾：final-run 里 explore L2 + final L3，qualified/融合检查都在
    # `_finish_cell` 一处决定。fixture 不另走一条判据路。
    record = _finish_cell(task_id, 'A', Path(selected), root=root,
                          declaration=declaration, budget=budget, backend=backend,
                          dispatches=len(requests), errors=errors,
                          returncode=response.get('returncode'), connection=connection)
    record.update(scripted_fixture=True, fixture_requests=len(requests),
                  fixture_errors=list(errors))
    return record


def _widen_tool_timeout(built, timeout_s):
    """把 MCP 工具调用与 bash 的超时放宽到整格墙钟量级。

    融合 L2（768 MiB 输入、32 条 case）在 SSH 上要**数分钟**，而 `s5_arm_env.build`
    给的缺省是 60 s —— v6 的第 5 个仪器缺陷（"MCP 工具调用 60 s 超时"）在这里原样复现：
    2026-11-17 fixture 首跑 C 臂 `operator_verify` 直接 `Request timed out`。
    """
    limit = max(120, int(timeout_s)) * 1000
    for row in built.get('patch') or ():
        for entry in row.get('insert') or ():
            entry_id = entry.get('id')
            config = entry.setdefault('config', {})
            if entry_id == 'mcp-operator':
                config['toolCallTimeoutMs'] = limit
            elif entry_id == 'bash-sandbox':
                config['timeoutMs'] = limit
                config['maxTimeoutMs'] = limit
    return built


def _stage_sandbox_runtime(runtime, root):
    """把 operator_agent 与 numpy 放进 runtime，供沙箱内的 MCP 子进程 import。

    `s5_seatbelt.configure` 给 arm=C 放行了 `runtime/core` 与 `runtime/plugin`，
    给所有臂放行了 `runtime/deps`；这三处就是 MCP 子进程看得见的世界。
    """
    import shutil
    import sys
    from pathlib import Path

    # **不能**放进 runtime 根目录：`verify_runtime` 会把整个 runtime 树哈希一遍，
    # 多一个文件就是 "public runtime drift"。放到它外面，再由读白名单放行。
    sandbox_root = Path(root) / 'sandbox'
    core = sandbox_root / 'core'
    deps = sandbox_root / 'deps'
    core.mkdir(parents=True, exist_ok=True)
    deps.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    # 包按"仓库布局"找自己的脚本：`protocol_identity` 会在 <root>/remote_scripts/run_kernel.py
    # 上验身份，所以 remote_scripts 必须一起进沙箱（实测 FileNotFoundError）。
    for name in ('operator_agent', 'remote_scripts'):
        target = core / name
        if not target.exists():
            shutil.copytree(repo / name, target,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    site = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    for name in ('numpy', 'numpy.libs', 'typing_extensions.py', 'typing_extensions'):
        source = site / name
        target = deps / name
        if source.exists() and not target.exists():
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copyfile(source, target)
    return f'{core}{os.pathsep}{deps}'


def _sandbox_c(built, *, runtime, root, extra_roots=(), extra_files=()):
    """给 C 臂写一份"仓库不可读"的 Seatbelt profile。

    为什么不直接用 `s5_seatbelt.configure`：它开头会 `verify_runtime(runtime)`，而那份校验
    假定 runtime 根目录**只有** plugin/proxy/check 三类文件；我们要把 operator_agent 与 numpy
    也放进沙箱可读区，一放就"public runtime drift"。这里复用同一个 `profile()` 生成器，
    只把读白名单换成 {DSH、runtime/plugin、runtime/core、runtime/deps}。
    """
    import sys
    from pathlib import Path

    from benchmarks.experiments.s5 import s5_seatbelt
    from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE

    arm_root = Path(built['home']).parent
    runtime_root = Path(runtime['root'])
    tmp = arm_root / 'tmp'
    for folder in (Path(built['home']), Path(built['work']), tmp):
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    # 系统目录：C 臂要在沙箱里自己 ssh，ssh 与它的动态库/配置都在这些位置
    # （实测：只放行臂目录时 ssh 起不来；补上 /usr、/System、/bin、/private/etc 后
    #  `ssh -i key … echo ok` 通过，仓库仍然读不到）。
    reads = [Path(DSH_ROOT).parents[1], arm_root / 'bin', runtime_root / 'plugin',
             Path('/usr'), Path('/System'), Path('/bin'), Path('/private/etc'),
             *[Path(p) for p in extra_roots]]
    # `extra_files` 是**单个文件**的读许可（例如 ssh 私钥）：只放行这一个路径，
    # 不用把整个 `~/.ssh` 或用户家目录暴露给臂。
    files = [*s5_seatbelt.python_files(), Path(NODE), arm_root / 'patch.json',
             *[Path(item) for item in extra_files]]
    files += [Path(p) for p in ('/private/etc/hosts', '/private/etc/resolv.conf',
                               '/private/etc/localtime', '/private/etc/services',
                               '/private/var/select/sh')]
    if built.get('development_enabled'):
        files.append(Path(built['env']['OPERATOR_AGENT_S5_DEV_CONFIG']))
    # run_dir 是**任务的**目录，插件必须能往里写回执（开工单）：它不在 arm 的 home/work 下，
    # 不显式放行就是 PermissionError(EPERM) —— 实测 MCP 子进程同样继承这份沙箱。
    task_dir = Path(root) / 'run'
    task_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    writes = [Path(built['home']), Path(built['work']), tmp, task_dir]
    text = s5_seatbelt.profile(reads, files, writes, python_prefix=sys.base_prefix)
    target = arm_root / 'isolation.sb'
    with target.open('x') as handle:
        handle.write(text)
    built['isolation'] = dict(schema='s5-seatbelt-launch-v1', kind='macos-file-read-whitelist',
                              profile=str(target),
                              profile_sha256=s5_seatbelt.checksum(target),
                              runtime=runtime, read_roots=list(map(str, reads)),
                              read_files=list(map(str, files)),
                              write_roots=list(map(str, writes)))
    built['env']['DSH_PERMISSION_MODE'] = 'danger-full-access'
    built['env']['TMPDIR'] = str(tmp)
    built['env']['PYTHONDONTWRITEBYTECODE'] = '1'
    built['env']['OPERATOR_AGENT_S5_OUTER_REQUIRED'] = '1'
    built['outer_isolation_required'] = True
    built['isolation']['inner_policy_delegated_to_outer'] = True
    return built


#: 仪器级失败的标记：这些不是候选写错，而是**我们自己的跑法**没跑起来。
#: 2026-11-10 实测：钉卡失败会让 L1 直接报 `required GPU selection previously failed`；
#: 别人抢卡会让 L2 报 `CUDA out of memory`。把它们记成 "候选失败" 就是污染实验。
INSTRUMENT_MARKERS = ('required GPU selection previously failed',
                      'CUDA out of memory',
                      'OutOfMemoryError',
                      'invalid upstream timeout')


def _instrument_failure(body):
    """从终评/探索回执里认出仪器级失败；返回命中的标记，否则 None。"""
    receipts = [body.get('final') or {}, body.get('explore') or {}]
    for receipt in receipts:
        for verdict in receipt.get('verdicts') or ():
            text = str(verdict.get('message') or '')
            for marker in INSTRUMENT_MARKERS:
                if marker in text:
                    return marker
    for field in ('reason',):
        text = str(body.get(field) or '')
        for marker in INSTRUMENT_MARKERS:
            if marker in text:
                return marker
    return None


def _counts(root, arm=None):
    """从盘上的记录数出这一格的调用数、验证数与"到首次完整 L2"的墙钟。

    - **调用数**：`cell/outer-requests/request-NNNN.json` 一一对应一次派发（网关原样落盘）；
    - **验证数**：`run/verify-*.json`（每次工具/check.py 提交一份）；
    - **墙钟**：第一条 `L2 通过` 的回执的 `created_at` 减去最早一份请求的时间戳。
      两者都是**实测**时间，不是估算；缺任一端就返回 None，不补数。
    """
    import json
    from pathlib import Path

    root = Path(root)
    requests = sorted((root / 'cell' / 'outer-requests').glob('request-*.json'))
    # 两条路各有各的回执：C 臂走插件工具（`run/verify-*.json`），A 臂走 check.py
    # （`run/native-verification/result-*.json`，由 supervisor 的 trusted worker 写）。
    receipts = sorted((root / 'run').glob('verify-*.json'))
    native = sorted((root / 'run' / 'native-verification').glob('result-*.json'))
    # 终评新开的 final-run 也是这格的验证派发（explore L2 + final L3 两次），
    # 以前只数 run/ 下的回执，于是终评两次派发在账外。
    final_receipts = sorted((root / 'final-run').glob('verify-*.json'))
    # **一次派发只能数一次**：A 臂的会话提交同时留下 `native-verification/result-N.json`
    # 与 `run/verify-N.json`（后者是服务内部调 ht.verify 写的），两处都数会把 A 的
    # 预算翻倍，与 C 臂不可比。按臂取其一，终评的 final-run 两臂都数。
    if arm == 'A':
        session = native
    elif arm == 'C':
        session = receipts
    else:
        session = receipts + native
    verifications = len(session) + len(final_receipts)
    first_full = None
    for path in receipts:
        try:
            value = json.loads(path.read_text())
        except ValueError:
            continue
        if (value.get('levels') or {}).get('L2') is True:
            first_full = value.get('created_at')
            break
    if first_full is None and native:
        # 原生 worker 的回执不带 created_at，用它的落盘时间；这是**实测**时间戳，
        # 只是精度到文件系统粒度（秒级）。
        for path in native:
            try:
                value = json.loads(path.read_text())
            except ValueError:
                continue
            if (value.get('levels') or {}).get('L2') is True:
                first_full = path.stat().st_mtime
                break
    wall = None
    if first_full is not None and requests:
        started = requests[0].stat().st_mtime
        try:
            wall = round(float(first_full) - started, 1)
            if wall < 0:
                wall = None
        except (TypeError, ValueError):
            wall = None
    total = None
    if requests:
        stamps = []
        for path in final_receipts:
            # **完成时间**：`complete.json` 落盘那一刻 = 这次验证真正结束；回执的
            # `created_at` 早于收尾几步（记账、回写），会把墙钟算短十几秒
            # （2026-11-19 复核：A 355.9 vs 369.8）。
            done = path.with_suffix('') / 'complete.json'
            if done.is_file():
                stamps.append(done.stat().st_mtime)
                continue
            try:
                value = json.loads(path.read_text())
            except ValueError:
                continue
            stamp = value.get('created_at')
            stamps.append(stamp if isinstance(stamp, (int, float)) else path.stat().st_mtime)
        if stamps:
            total = round(float(max(stamps)) - requests[0].stat().st_mtime, 1)
            if total < 0:
                total = None
    return dict(dispatches=len(requests) or None,
                dispatch_source='measured' if requests else None,
                verifications=verifications,
                first_full_l2_s=wall,
                total_wall_s=total,
                wall_clock_source='recomputed_from_receipt_and_request_timestamps'
                                 if wall is not None else None)


def _positive(value):
    return type(value) in (int, float) and value > 0


def _finish_cell(task_id, arm, selected, *, root, declaration, budget, backend,
                 dispatches=None, errors=(), returncode=None, connection=None,
                 scripted_fixture=False, acceptance='delivery'):
    """两臂共用的收尾：supervisor 在同一套 host task 上做完整 L2 + 独立终评。

    A 臂与 C 臂**必须**走这里（2026-11-08 决定：终评同源）。差别只在"提交经哪条路到达
    supervisor"——A 走 `check.py`，C 走插件的 `operator_verify`。
    """
    from pathlib import Path

    from .ac_driver import cell, final_evaluation

    selected = Path(selected)
    conn = dict(connection or {})
    final_run = Path(root) / 'final-run'
    # 修复验收的主判据是 L2（不比性能）⇒ 终评停在 L2：省掉一次 L3 派发（约 150 s/格），
    # 且不改变判据（`acceptance == 'repair'` 时 qualified 只看 L2）。
    repair_mode_early = acceptance == 'repair'
    body = final_evaluation(final_run, selected.read_text(),
                            declaration=declaration, host_budget=budget, backend=backend,
                            stop_after='L2' if repair_mode_early else 'L3',
                            host=conn.get('host'), work_dir=conn.get('work_dir'),
                            persistent=bool(conn.get('persistent', False)),
                            worker_root=final_run / 'owned-worker')
    # **计数必须来自量具，不是人手数**：之前 `model_dispatches`/`verifications` 都是 None，
    # 报告里的"每格 3–6 次调用"是人工数请求文件得来的（2026-11-10 评审指出）。
    # `arm=` 必须传：A 臂一次会话提交会同时写 `native-verification/result-N.json` 与
    # `run/verify-N.json`，不带臂就会把同一次派发数两遍（2026-11-19 复核：A4 记 4，
    # 实际 3）。
    counted = _counts(root, arm=arm)
    # 达标 = 完整 L2 + 独立 L3 valid + >= min_speedup + 单核融合静态核验。
    from .fusion_check import FusionCheckError, check_fused_source
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.decide.correctness_profiles import REGISTERED_FUSED_KERNEL_TYPES
    final_receipt = body.get('final') or {}
    final_levels = final_receipt.get('levels') or {}
    l3_metrics = {}
    for verdict in final_receipt.get('verdicts') or []:
        if verdict.get('level') == 'L3':
            l3_metrics = verdict.get('metrics') or {}
            break
    declared_min = float((declaration.get('min_speedup') if isinstance(declaration, dict)
                          else 1.0) or 1.0)
    fused_task = (build_task(declaration).source.kernel_type
                  in REGISTERED_FUSED_KERNEL_TYPES)
    # 修复模式：判据只看“修好没有” —— 完整 L2 + 独立终评 L2 都过；不比性能、不查融合结构
    # （修的是既有结构）。见 docs/testing/v08-repair-ac-preregistration.md。
    repair_mode = acceptance == 'repair'
    final_l2 = bool((((body.get('final') or {}).get('levels') or {}).get('L2')) is True)
    try:
        fusion = (check_fused_source(selected.read_text(encoding='utf-8'))
                  if fused_task else dict(passed=True, schema='not-applicable', reason=''))
    except FusionCheckError as exc:
        fusion = dict(passed=False, schema='v08-fusion-static-v1', reason=str(exc))
    claimed_wall = (budget or {}).get('wall_clock_s')
    total_wall = counted.get('total_wall_s')
    over_wall = (type(claimed_wall) in (int, float) and _positive(total_wall)
                 and total_wall > claimed_wall)
    if repair_mode:
        qualified = bool(not over_wall and final_levels.get('L2') is True and final_l2)
    else:
        qualified = bool(
            not over_wall
            and final_levels.get('L2') is True
            and l3_metrics.get('measurement_validity') == 'valid'
            and _positive(l3_metrics.get('speedup_vs_baseline'))
            and l3_metrics.get('speedup_vs_baseline') >= declared_min
            and fusion.get('passed') is True)
    # 正常用完墙钟仍未达标是失败，不是 unknown；over_wall 单独记录在墙钟字段里。
    delivery_class = ('qualified' if qualified else
                      'failed' if over_wall else
                      'correct_only' if final_levels.get('L2') is True else 'failed')
    record = cell(task_id, arm, state='completed', final=body, source=selected,
                  model_dispatches=dispatches if dispatches is not None else counted['dispatches'],
                  verifications=counted['verifications'],
                  wall_clock_to_first_full_L2_s=counted['first_full_l2_s'],
                  wall_clock_total_s=counted.get('total_wall_s'),
                  qualified=qualified, delivery_class=delivery_class,
                  fusion_static=fusion,
                  final_speedup=l3_metrics.get('speedup_vs_baseline'),
                  final_validity=l3_metrics.get('measurement_validity'))
    record.update(outcome='delivered' if body['delivered'] else 'failed',
                  reason=body.get('reason'),
                  levels=(body['final'] or {}).get('levels') if body['final']
                  else (body['explore'] or {}).get('levels'),
                  scripted_fixture=bool(scripted_fixture), fixture_errors=list(errors),
                  dsh_returncode=returncode,
                  # 计数出处：`measured` 表示盘上有一一对应的记录；`recomputed` 表示事后从
                  # 转录/回执时间戳算出来的。两者不能混进同一个字段而不标注（评审 P0-①）。
                  metering=dict(dispatches=counted['dispatch_source'],
                                verifications='measured' if counted['verifications'] else None,
                                wall_clock=counted['wall_clock_source']))
    return record


def _c_script(declaration, budget, run_dir, code, backend='fake', ssh=None):
    """C 臂的 scripted 剧本：prepare → verify(explore) → verify(final) → 收尾。

    顺序被 `host_final_source` 强制：final 之前必须有一次**同源码**的完整 L2。
    """
    return [
        dict(tool='mcp__operator__operator_prepare',
             args=dict(request=declaration['request'], shapes=declaration['shapes'],
                       hardware=declaration['hardware'], dsl=declaration['dsl'],
                       precision=declaration['precision'], min_speedup=declaration.get('min_speedup', 1.0),
                       run_dir=str(run_dir),
                       host_budget=dict(wall_clock_s=budget['wall_clock_s'],
                                        max_verifications=budget['max_verifications'],
                                        max_model_calls=budget['max_model_calls'],
                                        max_output_tokens_per_call=budget['max_output_tokens_per_call'],
                                        total_output_token_reservations=budget['total_output_token_reservations']))),
        dict(tool='mcp__operator__operator_verify',
             args=dict(run_dir=str(run_dir), code=code, candidate_origin='outer',
                       stop_after='L2', backend=backend,
                       **({'host': (ssh or {}).get('host'),
                           'port': (ssh or {}).get('port'),
                           'user': (ssh or {}).get('user'),
                           'key_path': (ssh or {}).get('key_path')}
                          if backend == 'ssh' else {}))),
        # 终评**不在这里做**：方案 1 里 A/C 的终评同源，由 supervisor 在 `_finish_cell`
        # 用同一套 host task 产出。会话只负责把候选提交到判据上。
        dict(tool=None, args=None),
    ]


def _scripted_handlers(handler, requests, errors, model_id):
    return handler


def run_c_cell(declaration, *, task_id, candidate_code, root, budget, backend='fake',
               ssh=None, connection=None, model_id='ac-fixture', timeout=240, sandbox=False):
    """跑一格 C 臂（scripted 模型走插件 MCP 工具），返回 `ac_driver.cell` 形状的 dict。"""
    from benchmarks.experiments.s5 import s5_arm_env, s5_plugin_env
    from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE

    from .ac_driver import cell

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # **必须绝对**：MCP 子进程按它自己的 cwd 解析相对路径，实测会把 run 目录建到
    # `dsh/work/…` 下面，于是 supervisor 在自己的 run_dir 里一个回执都看不到。
    run_dir = (root / 'run').resolve()
    from operator_agent.app import hitl_tools as ht
    prepared = ht.prepare(declaration, run_dir, host_budget=dict(budget))
    plan = _c_script(declaration, budget, run_dir, candidate_code, backend=backend, ssh=ssh)
    requests, errors = [], []

    def handler_factory():
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    if self.path not in ('/v1/chat/completions', '/chat/completions'):
                        raise FixtureError(f'unexpected path {self.path}')
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    requests.append(request)
                    # 落一份首请求：工具面必须能被事后核对（沙箱里插件没挂上时，
                    # 模型只是"少了一批工具"，不留证据就只能靠猜）。
                    if len(requests) == 1:
                        first = root / 'fixture-requests'
                        first.mkdir(parents=True, exist_ok=True)
                        (first / 'request-0000.json').write_text(
                            json.dumps(request, ensure_ascii=False)[:2_000_000],
                            encoding='utf-8')
                    step = min(len([m for m in request['messages'] if m.get('role') == 'tool']), len(plan) - 1)
                    action = plan[step]
                    model = request.get('model') or model_id
                    if action['tool'] is None:
                        delta, finish = dict(content='Done: final evaluation submitted.'), 'stop'
                    else:
                        delta = dict(tool_calls=[dict(index=0, id=f'fixture-{len(requests)}',
                                                      type='function',
                                                      function=dict(name=action['tool'],
                                                                    arguments=json.dumps(action['args'])))])
                        finish = 'tool_calls'
                    chunks = [dict(id=f'fixture-{len(requests)}', model=model,
                                   choices=[dict(index=0, delta=delta, finish_reason=None)]),
                              dict(id=f'fixture-{len(requests)}', model=model,
                                   choices=[dict(index=0, delta={}, finish_reason=finish)])]
                    payload = _sse(chunks)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as error:                 # noqa: BLE001
                    errors.append(f'{type(error).__name__}: {error}')
                    self.send_response(500)
                    self.end_headers()
        return Handler

    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_factory())
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        runtime = s5_plugin_env.stage_runtime(root / 'runtime')

        # 方案 1（2026-11-08 决定）：A/C 的**终评同源**。插件以 **host 模式**起 MCP
        # （`OPERATOR_AGENT_WORKFLOW=host`），会话里只做提交；终评由 supervisor 在
        # `_finish_cell` 里用同一套 `prepare(host_budget) + purpose=final` 产出。
        #
        # 不走 `s5_plugin_env`（那条是 plugin-only-ac 的 S5 协议：MCP 子进程要握手一个
        # supervisor 开发端点，终评归属会变成两套）。`s5_arm_env.build('C')` 直接把
        # `operator_agent.app.mcp_server` 挂成 stdio server —— 与 R2 那几对同一种形状。
        built = s5_arm_env.build(
            'C', base_url=f'http://127.0.0.1:{server.server_port}/v1', model=model_id, tools=True,
            max_tokens=1024, context_window=65536, arm_dir=root / 'dsh')
        built['env'] = dict(built['env'], OPERATOR_AGENT_WORKFLOW='host')
        _widen_tool_timeout(built, budget['wall_clock_s'])
        if sandbox:
            # 沙箱里看不到仓库：插件与 operator_agent 都从 runtime 读。
            # `s5_seatbelt.configure` 已经允许读 runtime/{core,plugin,deps}（arm=C 时），
            # 所以把三样东西放进去，再把 PYTHONPATH 与 patch 指过去。
            python_path = _stage_sandbox_runtime(runtime, root)
            built['env'] = dict(built['env'], PYTHONPATH=python_path)
            for row in built['patch']:
                for entry in row.get('insert') or ():
                    config = entry.get('config') or {}
                    if isinstance(config.get('env'), dict):
                        config['env']['PYTHONPATH'] = python_path
                    if entry.get('id') == 'operator-tools':
                        # `name` 是插件模块的**路径**，DSH 会去 lstat 它；沙箱里仓库不可读，
                        # 必须换成 runtime 里的拷贝（只改 command 不够，实测 EPERM）。
                        entry['name'] = str(Path(runtime['root']) / 'plugin' / 'index.mjs')
                        config['command'] = str(Path(runtime['root']) / 'plugin' / 'index.mjs')
            _sandbox_c(built, runtime=runtime, root=root,
                       extra_roots=[Path(python_path.split(os.pathsep)[0]).parent,
                                    Path(python_path.split(os.pathsep)[1])])
        else:
            # 非沙箱（fixture 快速验证用）：归档的 runtime-deps/site 里有坏掉的 numpy，
            # 它排在 venv 之前会让 import 失败，所以只留仓库树。
            repo_root = str(Path(__file__).resolve().parents[3])
            built['env'] = dict(built['env'], PYTHONPATH=repo_root)
            for row in built['patch']:
                for entry in row.get('insert') or ():
                    env = (entry.get('config') or {}).get('env')
                    if isinstance(env, dict) and 'PYTHONPATH' in env:
                        env['PYTHONPATH'] = repo_root
        for folder in (built['work'], built['home']):
            Path(folder).mkdir(parents=True, exist_ok=True)
        response = s5_arm_env.start_headless(built, 'Follow the operator workflow.',
                                             timeout=timeout, capture_dir=root / 'host-output')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        pass

    # 会话里最后一次通过完整 L2 的提交就是"交付的那份源码"（选最后，不挑最好）
    selected, last = None, None
    for path in sorted(run_dir.glob('verify-*.json')):
        value = json.loads(path.read_text())
        last = value
        if (value.get('levels') or {}).get('L2') is True and value.get('source'):
            selected = value['source']
    if selected is None:
        return dict(task_id=task_id, arm='C', state='crashed', final_receipt=None, source=None,
                    outcome='unknown',
                    reason=f'no passing submission: {errors or response.get("stderr")}',
                    last_receipt=last, scripted_fixture=True,
                    model_dispatches=len(requests), fixture_requests=len(requests))
    record = _finish_cell(task_id, 'C', selected, root=root, declaration=declaration,
                          budget=budget, backend=backend, dispatches=len(requests),
                          errors=errors, returncode=response.get('returncode'),
                          connection=dict(connection or {}))
    return record


def main(argv=None):
    import argparse
    import os
    import tempfile

    parser = argparse.ArgumentParser(prog='ac_fixture', description=__doc__.splitlines()[0])
    parser.add_argument('--json-out', required=True)
    parser.add_argument('--plan', default=None,
                        help='v7 计划 JSON；省略时用 ac_plan.design("v7") + 冻结每格预算')
    parser.add_argument('--task-id', default=None)
    parser.add_argument('--arm', default='all', choices=['A', 'C', 'all'])
    parser.add_argument('--candidate-file', default=None,
                        help='脚本模型提交的候选源码；缺省用手写融合核')
    parser.add_argument('--root', default=None, help='这一对的私有目录')
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', default='/root/operator-p2-seed-20261116')
    parser.add_argument('--ssh-host', default=None)
    parser.add_argument('--ssh-port', default=None)
    parser.add_argument('--ssh-user', default=None)
    parser.add_argument('--ssh-key', default=None)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args(argv)

    from benchmarks.experiments.v08 import ac_plan
    plan = (json.loads(Path(args.plan).read_text(encoding='utf-8')) if args.plan
            else ac_plan.design('v7'))
    task = next((row for row in plan['tasks']
                 if args.task_id in (None, row['task_id'])), None)
    if task is None:
        raise SystemExit(f'计划里没有任务 {args.task_id!r}')
    task_id = task['task_id']
    declaration = dict(task['declaration'])
    raw_budget = {key: value for key, value in (plan.get('budget') or {}).items()
                  if value is not None}
    for key, value in ac_plan.FROZEN_FUSED_PER_CELL.items():
        raw_budget.setdefault(key, value)
    # `hitl_tools.prepare` 只认 HostTaskLimits 的字段；cells/gpu_minutes 是计划账，
    # 不是 host budget。在这里投影一次，不把计划字段漏给执行器。
    from operator_agent.execute.host_task_budget import HostTaskLimits
    budget = {key: raw_budget[key] for key in HostTaskLimits.__dataclass_fields__
              if raw_budget.get(key) is not None}

    execution = plan.get('execution') or {}
    gpu = execution.get('gpu') or {}
    if gpu.get('index') is not None:
        os.environ.setdefault('OPERATOR_AGENT_REQUIRED_GPU', str(gpu['index']))
        os.environ.setdefault('OPERATOR_AGENT_REQUIRED_GPU_UUID', str(gpu.get('uuid') or ''))
        os.environ.setdefault('OPERATOR_AGENT_REQUIRED_GPU_NAME',
                              str(gpu.get('name') or 'NVIDIA H20'))

    candidate = Path(args.candidate_file) if args.candidate_file else (
        Path(__file__).resolve().parent / 'kernels' / 'fused_ln_residual.py')
    code = candidate.read_text(encoding='utf-8')
    base = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix='ac-fused-fixture-'))
    base.mkdir(parents=True, exist_ok=True)
    connection = dict(host=args.host, work_dir=args.work_dir, persistent=True)
    ssh = dict(host=args.ssh_host or args.host, port=args.ssh_port,
               user=args.ssh_user, key_path=args.ssh_key)
    if args.arm in ('C', 'all') and not ssh.get('key_path'):
        raise SystemExit('C 臂 SSH 验证需要 --ssh-key（operator_verify 要求显式 key_path）')
    arms = ('A', 'C') if args.arm == 'all' else (args.arm,)
    cells = []
    for arm in arms:
        root = base / arm
        if arm == 'A':
            record = run_a_cell(declaration, task_id=task_id, candidate_code=code, root=root,
                                budget=budget, backend='ssh', connection=connection,
                                timeout=args.timeout)
        else:
            record = run_c_cell(declaration, task_id=task_id, candidate_code=code, root=root,
                                budget=budget, backend='ssh', ssh=ssh, connection=connection,
                                timeout=args.timeout)
        cells.append(record)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(cells if len(cells) > 1 else cells[0], ensure_ascii=False, indent=2,
                   default=str) + '\n', encoding='utf-8')
    print(json.dumps([{key: record.get(key) for key in
                       ('arm', 'state', 'outcome', 'delivery_class', 'qualified',
                        'final_validity', 'final_speedup', 'model_dispatches',
                        'verifications')} for record in cells],
                     ensure_ascii=False))
    return 0 if all(record.get('final_receipt') for record in cells) else 2


if __name__ == '__main__':
    raise SystemExit(main())