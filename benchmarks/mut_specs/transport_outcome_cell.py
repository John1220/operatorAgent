"""`tests/unit/test_transport_outcome_cell.py` 那张变异表的可执行形态 (#102 第 1 片)

这片给执行侧加一格「听到回答 / 没听到回答」(Limbo `2609.29095v1` 行 220-232 的观测等价),
**只观测不改判决**。判据散在五个文件里, 所以变异表也跨五个文件打 —— 每一行都对应一种
「这一格会静默失效」的写法, 而不是抽象的「代码被改了」:

    T1 分类器不看 rc                       T6 verifier 本地那路直接取属性 (本片撞出的真崩溃)
    T2 把「答了失败」折进「没答话」          T7 `_round_transport` 取第一格而不是 L2 那格
    T3 `_ssh_exec` 不把 rc 挂到异常上       T8 血缘那格缺值时兜成 answered (拿缺证据当通过)
    T4 四个出口里漏一个带格                 T9 报表把「这一轮没报」并进 answered
    T5 verifier 远程那路缺格也填默认         T10 报表删掉「另有 N 轮是空的」那一支

跑: python3 -B -m benchmarks.calibration.mutation_calibrate \
      --spec benchmarks/mut_specs/transport_outcome_cell.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ms": "operator_agent/decide/measurement.py",
    "re": "operator_agent/execute/remote_executor.py",
    "vf": "operator_agent/decide/verifier.py",
    "cd": "operator_agent/app/conductor.py",
    "hp": "benchmarks/drivers/harness_pass_rate.py",
}

TARGETS = ["tests/unit/test_transport_outcome_cell.py"]

ROWS = [
    Row("T1 分类器不看远端 rc (`remote_rc == 255` 那一支废掉)", "ms",
        "    if remote_rc == SSH_TRANSPORT_EXIT_CODE:\n        return TRANSPORT_UNKNOWN",
        "    if False:\n        return TRANSPORT_UNKNOWN",
        "⇒ 2 条红 (`test_ssh_exit_255_is_unknown_even_when_the_exception_type_says_otherwise` 与 "
        "`test_ssh_exit_255_reaches_the_verdict_as_unknown` 那条整路)。255 是 ssh **自己**失败的保留码, "
        "丢掉它就等于把「命令可能根本没跑」读成「远端答了失败」—— 本片的全部依据就在这一行。"
        "⚠️ `test_absent_rc_does_not_become_unknown` 对它是**绿的** (它断的就是 answered), "
        "所以 T1 的证人只有那两条 —— 记下实际读数而不是我以为的 3 条"),

    Row("T2 把「答了, 答的是失败」折进「没答话」(两格并成一格)", "ms",
        "        if klass.__name__ in _NO_RESPONSE_EXCEPTIONS:\n            return TRANSPORT_UNKNOWN\n"
        "    return TRANSPORT_ANSWERED",
        "        if klass.__name__ in _NO_RESPONSE_EXCEPTIONS:\n            return TRANSPORT_UNKNOWN\n"
        "    return TRANSPORT_UNKNOWN",
        "⇒ 5 条红 (answered 那一侧全灭: 异常族 / rc=1,126,127 / rc 缺席 / 执行侧 benchmark 出口 / "
        "本地执行发布)。"
        "这一行是 T2 的**保守方向**判据本身: 误报 unknown 只是多问一句, 全报 unknown 则让"
        "这一栏失去任何分辨力, 而读者看不出区别 —— 它印出来仍然是个数"),

    Row("T3 `_ssh_exec` 不把 rc 挂到异常上 (255 与 1 在类型层同形)", "re",
        "            err.ssh_rc = proc.returncode",
        "            err.ssh_rc = 1",
        "⇒ `test_ssh_exit_255_reaches_the_verdict_as_unknown` 红 (既断 `ssh_rc == 255` 也断"
        "判决)。换异常类型会静默改掉文件里三处 `except RuntimeError` 的探测分支, 所以 rc 是"
        "**挂在异常上**带过去的; 这一行丢了, 那条真实路径就退回「只能靠异常类名猜」"),

    Row("T4 四个失败出口里漏一个带格 (verify_function 回旧形状)", "re",
        '                "action": "verify", "code": kernel_code,\n'
        '                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,\n'
        '                "max_error": max_error, "precision": precision,\n'
        '                "reference_source": reference_source or "",\n'
        "            })\n        except Exception as e:\n            return _transport_failure(e)",
        '                "action": "verify", "code": kernel_code,\n'
        '                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,\n'
        '                "max_error": max_error, "precision": precision,\n'
        '                "reference_source": reference_source or "",\n'
        "            })\n        except Exception as e:\n"
        '            return {"success": False, "error": str(e)}',
        "⇒ `test_verify_function_over_a_reset_connection_is_unknown` 红: KeyError。"
        "⚠️ 锚点必须从 `action=verify` 那一行起手: 这一族 `except Exception as e:` + "
        "`return _transport_failure(e)` 在文件里有三份 (benchmark / verify / benchmark_with_baselines), "
        "短的锚点会被规格自检按「不唯一」拒绝运行 —— 与上一片那条 R5 同一件事"),

    Row("T5 远端载荷缺这一格时 verifier 替它填 answered (造证据)", "vf",
        "                    metrics=({TRANSPORT_KEY: remote_verify[TRANSPORT_KEY]}\n"
        "                             if TRANSPORT_KEY in remote_verify else {}))",
        '                    metrics={TRANSPORT_KEY: remote_verify.get(TRANSPORT_KEY, "answered")})',
        "⇒ `test_a_remote_that_says_nothing_about_transport_is_not_answered_on_its_behalf` 红。"
        "远端脚本今天不报这一格, 于是「载荷里没这一格」是常态; 替它填 answered 之后, "
        "「远端答了失败」与「这一级根本没区分」在血缘里同形 —— 而那正是这片要拆开的两格"),

    Row("T6 本地执行那一路直接取属性 (MockGPU 的 ExecutionResult 没有这个字段)", "vf",
        "metrics={TRANSPORT_KEY: transport_outcome_of_result(result)})",
        "metrics={TRANSPORT_KEY: result.transport_outcome})",
        "⇒ `test_a_local_failure_does_not_raise_and_is_published_as_answered` 红 "
        "(AttributeError)。这一条不是假想的: 它是写这片用例时**撞出来的真崩溃** —— "
        "本机执行的失败分支此前零证人, 所以「候选算错了」那条判决会变成 harness 崩溃"),

    Row("T7 `_round_transport` 取第一格而不是 L2 那一格", "cd",
        '               if getattr(r, "level", None) == VerificationLevel.L2_FUNCTION), None)\n'
        '    metrics = getattr(l2, "metrics", None) if l2 else None\n'
        "    return (metrics or {}).get(TRANSPORT_KEY)",
        '               if getattr(r, "level", None) == VerificationLevel.L2_FUNCTION), None)\n'
        '    metrics = getattr(verify_results[0], "metrics", None) if verify_results else None\n'
        "    return (metrics or {}).get(TRANSPORT_KEY)",
        "⇒ `test_only_the_l2_row_counts` 红: 真实循环里第一格通常是 L1, 于是报表那一栏印的"
        "其实是「编译那一步有没有听到回答」。逐键判定不连坐 —— 这一行是它在血缘侧的版本"),

    Row("T8 血缘那一格缺值时兜成 answered (拿缺证据当通过)", "cd",
        "    return (metrics or {}).get(TRANSPORT_KEY)",
        '    return (metrics or {}).get(TRANSPORT_KEY, "answered")',
        "⇒ 5 条红 (`_round_transport` 那三条 + 真跑那组过 JSON 的两条)。"
        "与 #53 的 `speedup=None` 印成「没有加速」同族, 只是方向反过来: 这里会把"
        "「没走到远程执行」读成「每次都听到了回答」"),

    Row("T9 报表把「这一轮没报」并进 answered 那一格", "hp",
        "        t = r.get(TRANSPORT_KEY)\n        if t is not None:\n"
        "            transport_counts[t] = transport_counts.get(t, 0) + 1",
        "        t = r.get(TRANSPORT_KEY) or TRANSPORT_ANSWERED\n"
        "        transport_counts[t] = transport_counts.get(t, 0) + 1",
        "⇒ 3 条红 (`test_the_three_cells_are_counted_separately` 的两个数 + 打印那条的 "
        "`报了这一格的轮 6/8`)。「没检查」折进「检查过且没问题」是本仓最老的那一族 (I25/I27)"),

    Row("T10 报表删掉「另有 N 轮这一格是空的」那一支", "hp",
        '    if _silent:\n        lines.append(f"  另有 {_silent} 轮这一格是空的',
        "    if False:\n        lines.append(f\"  另有 {_silent} 轮这一格是空的",
        "⇒ 2 条红 (混合批次与全空批次各一条)。全空批次只剩 `answered 0` 时, 那一栏读起来像"
        "「查过了, 一次都没丢」—— 缺了分母的正面对照, 0 就不是一个测量值而是一个修辞"),
]
