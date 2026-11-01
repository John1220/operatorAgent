"""「§22.2 本地受控执行：跑飞要拦得住，设不上要说得清」那张变异表的**可执行**形态

靶子是 `tests/unit/test_exec_guard.py`（32 条）+ `tests/unit/test_demo_exec_guard.py`（演示自称的读数），
另有两行落在外面的旧守卫上（跨进程带回 `lib_fallback` 证据那条）。

为什么单独一张表: §22.2 这一片有**两个**容易被读大的地方 ——
  ① 「杀了子进程」不等于「杀了进程树」；
  ② 「设上了 rlimit」不等于「如实报没设上的项」。
每行拆掉一层，看守卫是不是真的在守它。只守一层的话，另一层就是「文档里写着、代码里没有」。

跑: python3 -B -m benchmarks.calibration.mutation_calibrate --spec benchmarks/mut_specs/exec_guard.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "eg": "operator_agent/execute/exec_guard.py",
    "mg": "operator_agent/execute/mock_gpu.py",
}

TARGETS = ["tests/unit/test_exec_guard.py", "tests/unit/test_demo_exec_guard.py",
           "tests/unit/test_lib_fallback_producer.py"]

ROWS = [
    Row("M1 超时只杀直接子进程, 不杀进程组 (孙子进程变孤儿)", "eg",
        "        _kill_process_group(child, grace_ms=grace_ms)",
        "        os.kill(child.pid, signal.SIGKILL)",
        "**具体输入**: 候选起一个 1.0s 后写文件的孙子进程然后死循环, 墙钟 300ms ⇒ "
        "痕迹文件出现 ⇒ `test_the_whole_process_group_is_killed_including_a_grandchild` 红"
        "（以及演示里「孙子留下的痕迹存在? True」那行）"),
    Row("M2 正常返回后不再收尾 (进程树清理只剩超时那一路)", "eg",
        "        child.killpg(signal.SIGKILL)\n    wall_ms = (time.monotonic() - started) * 1000.0",
        "        pass\n    wall_ms = (time.monotonic() - started) * 1000.0",
        "**具体输入**: 候选起一个后台子进程后**正常返回** ⇒ 痕迹文件出现 ⇒ 同一条用例的"
        "第二半红（「正常返回也要清一次」这半只有行为证人，没有别的东西守着）"),
    Row("M3 rlimit 设不上却报成「已设」（三态压成布尔）", "eg",
        '''    except Exception as exc:
        return {"item": item, "state": "not_enforced", "requested": value,
                "reason": "%s: %s" % (type(exc).__name__, exc), "reading": ""}''',
        '''    except Exception as exc:
        return {"item": item, "state": "enforced", "requested": value,
                "reason": "%s: %s" % (type(exc).__name__, exc), "reading": ""}''',
        "**具体输入**: 本机 `Limits(memory_bytes=64MB)` ⇒ 内核 EINVAL, 但报告会说 enforced "
        "⇒ `test_the_memory_item_is_honest_on_every_platform` 红（不是「memory 没拦住」，"
        "而是「明明设不上却说设上了」—— 这正是 I65/I11 那一族）"),
    Row("M4 「调用方没请求」被读成「已设上」", "eg",
        '''    if value is None:
        return {"item": item, "state": "not_enforced", "requested": None,''',
        '''    if value is None:
        return {"item": item, "state": "enforced", "requested": None,''',
        "**具体输入**: `Limits(cpu_s=None, memory_bytes=None, file_size_bytes=None, "
        "processes=None)` ⇒ `test_an_item_that_was_not_requested_is_not_reported_as_enforced` 红"
        "（`None` = 不请求，不是「请求 0」，更不是「已设上」）"),
    Row("M5 产物超上限 ⇒ 截断后当通过", "eg",
        '''        result = {"status": "limit_exceeded", "kind": "artifact", "detail":
                  "产物 %d 字节 > 上限 %d 字节" % (len(blob), cap),
                  "output": None, "error": None, "elapsed_ms": result["elapsed_ms"],''',
        '''        result = {"status": "ok", "kind": None, "detail":
                  "产物 %d 字节 > 上限 %d 字节" % (len(blob), cap),
                  "output": None, "error": None, "elapsed_ms": result["elapsed_ms"],''',
        "**具体输入**: 候选返回 16MB 数组, 上限 256KB ⇒ 状态变 ok ⇒ "
        "`test_a_too_large_artifact_fails_instead_of_being_truncated` 与演示里"
        "「毫无疑问被读回来了吗? False」那行红"),
    Row("M6 `SIGXCPU` 不再被认成 CPU 那一档 (死法不分类)", "eg",
        '''    if signum == getattr(signal, "SIGXCPU", None):
        return STATUS_LIMIT_EXCEEDED, KIND_CPU, "撞上 CPU 时间上限 (SIGXCPU)"''',
        '''    if signum == getattr(signal, "SIGXCPU", None):
        return STATUS_CRASHED, None, "撞上 CPU 时间上限 (SIGXCPU)"''',
        "**具体输入**: 死循环 + `wall_ms=6000, cpu_s=1` ⇒ 状态从 limit_exceeded 变 crashed "
        "⇒ `test_the_cpu_limit_is_reported_as_its_own_kind` 与 `test_classify_death_maps_"
        "signals_without_guessing` 红（以及演示里 `kind=cpu` 那行）"),
    Row("M7 `SIGXFSZ` 不再被认成单文件大小那一档", "eg",
        '''    if signum == getattr(signal, "SIGXFSZ", None):
        return STATUS_LIMIT_EXCEEDED, KIND_FILE_SIZE, "撞上单文件大小上限 (SIGXFSZ)"''',
        '''    if signum == getattr(signal, "SIGXFSZ", None):
        return STATUS_CRASHED, None, "撞上单文件大小上限 (SIGXFSZ)"''',
        "**具体输入**: `classify_death(None, signal.SIGXFSZ, False)` ⇒ "
        "`test_classify_death_maps_signals_without_guessing` 红。⚠️ 本机 Darwin 在 "
        "`RLIMIT_FSIZE` 到顶时返回的是 `EFBIG`（走 `classify_exception`），**不是** SIGXFSZ "
        "—— 所以这一行在今天的行为层只有单测那条证人，端到端走不到它（Linux 才走得到）"),
    Row("M8 父侧不再补「读不到的项」⇒ 被杀的子进程默认全部设上了", "eg",
        '''    if report and report.get("enforcement"):
        return _ordered(Enforcement(**e) for e in report["enforcement"])''',
        '''    if report and report.get("enforcement"):
        return _ordered(Enforcement(**e) for e in report["enforcement"])
    return tuple(Enforcement(item=i, state=ENFORCED, requested=None) for i in ITEMS)''',
        "**具体输入**: 墙钟超时的候选（子进程没机会回报）⇒ 报告里 CPU/内存/文件大小/进程数"
        "全部变成 enforced ⇒ `test_a_timeout_does_not_claim_the_limits_it_could_not_read` 红"
        "（读不到却说设上了，正是 I65/I11 那一族）"),
    Row("M9 失败原因不再带读数 (detail 里不再有 maxrss)", "eg",
        '''            where += " (子进程 maxrss ≈ %.1f MB)" % rss      # 失败原因带读数''',
        '''            pass''',
        "**具体输入**: 单文件大小超限那条（候选写 20MB 被 1MB 上限拦住）⇒ detail 只剩 "
        "`OSError: [Errno 27] File too large @ <candidate>:4 in kernel`，没有用了多少内存的"
        "读数 ⇒ `test_a_huge_file_write_fails_with_a_reading` 与 "
        "`test_the_process_limit_stops_a_mild_fork_bomb` 里的 `assertIn(\"maxrss\", r.detail)` 红"
        "（「失败原因**带读数**」这句话的守卫就在这一格）"),
    Row("M10 `_pick_executor` 不看线程 (多线程父进程里照样 fork)", "eg",
        '    return EXECUTOR_FORK if threading.active_count() == 1 else EXECUTOR_SUBPROCESS',
        '    return EXECUTOR_FORK',
        "**具体输入**: 起一个线程后调 `_pick_executor(\"auto\")` ⇒ 返回 fork ⇒ "
        "`test_auto_picks_subprocess_when_another_thread_is_running` 红（fork 的死锁风险"
        "没人挡了）"),
    Row("M11 fork 起不来时不再退回 (异常直接穿成候选失败)", "eg",
        '''        except OSError:
            # fork 自己起不来（比如进程数上限已到）**不能**被读成「候选失败」:
            # 退回新解释器后端, 保证一模一样, 只是慢。用哪个后端仍进 `executor`。
            backend = EXECUTOR_SUBPROCESS''',
        '''        except OSError:
            raise''',
        "**具体输入**: `unittest.mock.patch(\"os.fork\", side_effect=OSError(EAGAIN, ...))` "
        "之后跑一个正常候选 ⇒ 异常穿出 `run_candidate`, 调用方只会看到「候选炸了」⇒ "
        "`test_a_fork_failure_falls_back_instead_of_failing_the_candidate` 红"
        "（起不来 ≠ 候选失败；退回路径必须留着）"),
    Row("M14 两个预算的不变量删掉 (产物装不进回传文件也放行)", "eg",
        '''        if (self.artifact_bytes is not None and self.file_size_bytes is not None
                and self.artifact_bytes > self.file_size_bytes):''',
        '''        if False:''',
        "**具体输入**: `Limits(file_size_bytes=1MB, artifact_bytes=8MB)` ⇒ 不再抛 "
        "`ValueError` ⇒ `test_the_two_budgets_cannot_contradict_each_other` 红。"
        "⚠️ 2026-10-04 实测过这条路径的后果: 子进程写到 FSIZE 上限被截断，父侧只读到一句 "
        "`UnpicklingError: pickle data was truncated`（`tests/unit/test_leetcuda_bench.py` "
        "三条红），看起来像「候选坏了」—— 可判定的组合必须在构造期失败"),
    Row("M12 MockGPU 不再走受控入口 (`_execute_python` 直接 exec)", "mg",
        "        guard = run_candidate(kernel_code, inputs,\n"
        "                              limits=limits_for(MODE_RUN, self.exec_limits))",
        "        _ns = {}\n"
        "        exec(kernel_code, _ns)\n"
        "        _out = _ns[\"kernel\"](**inputs)\n"
        "        guard = type(\"G\", (), {})()\n"
        "        guard.ok, guard.output, guard.elapsed_ms = True, _out, 0.0\n"
        "        guard.metrics, guard.not_enforced, guard.fully_enforced = {}, (), True\n"
        "        guard.executor, guard.status = \"in-process\", \"ok\"\n"
        "        guard.kind, guard.error, guard.detail = None, None, \"\"",
        "**具体输入**: 一份会改 harness 状态的候选（改 `np.zeros` / 环境变量 / cwd）"
        "⇒ 进程内 exec 会让那些改动**留在 harness 里**（`tests/unit/test_exec_guard.py` 的"
        "「候选改不动 harness」那组用例红）。合法候选那一趟它看不出来（两边返回值一样），"
        "所以这一条同时还要靠接线断言（`_execute_python` 里不许有 `exec`）",
        behavioral=True),
    Row("M13 盾的命中证据不再跨进程带回来 (父侧只剩空 metrics)", "eg",
        '''        metrics = lib_fallback_metrics()
        if not metrics and hit:
            metrics = lib_fallback_metrics(remote_error=bare)''',
        '''        metrics = {}''',
        "**具体输入**: `getattr(torch, \"mat\"+\"mul\")` 这种静态看不见的作弊候选 ⇒ "
        "`tests/unit/test_lib_fallback_producer.py::..."
        "test_verifier_l2_runtime_shield_hit_metrics` 与 "
        "`test_mock_gpu_execute_produces_metrics` 红（盾现在装在子进程里，证据必须带回来 —— "
        "否则 eligibility 规则 4 会在「唯一真正需要它的那条路径」上恒 False，正是 P1-4 那个洞）"),
]