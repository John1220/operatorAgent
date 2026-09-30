# -*- coding: utf-8 -*-
"""#102 第 2 片 (远端异常文本那道掩码) 的变异表。

跑法:
    python3 -B -m benchmarks.mutation_calibrate \\
        --spec benchmarks/mut_specs/transport_error_mask.py

九行全在行为分母：这一片的判据本身就是「给一份带秘密的输入, 看输出里还有没有」——
每行都有一个具体输入能观察到差异, 所以没有结构行 (按 #83, 没有可观察差异的行要写隔离理由,
而这里每一行都有)。

M2 那一行不是假想病: 它是本片**第一版的真红法** —— 第一版只摘 IPv4, 于是
`ssh: connect to host 198.51.100.7 port 22001` 变成 `connect to host [已隐去] port 22001`,
端口活了下来 (掩码摘了「地址」却没摘「地址的一部分」)。

M5/M8 是两条**反向对照**: 掩码扩得太宽 (把被修对象打码) 与 unknown 判得太宽 (凡带返回码都算
没听到回答) —— 两种过量都会绿掉「不该响的那一侧」，而本仓这一族的教训是「全绿可能只证这条
分支没人走」，所以两条都必须有活着的用例。
"""

from benchmarks.mutation_calibrate import Row

FILES = {
    "ms": "operator_agent/measurement.py",
    "rx": "operator_agent/remote_executor.py",
    "env": "operator_agent/environment.py",
}

TARGETS = [
    "tests/test_transport_error_mask.py",
    "tests/test_transport_outcome_cell.py",
]

ROWS = [
    Row("M1 `safe_error_text` 退回异常原文 (整条 argv 又过落盘边界)",
        "ms", '    cmd = getattr(exc, "cmd", None)\n    if cmd is not None:',
        '    cmd = None\n    if cmd is not None:',
        "存在具体输入可观察: 一条带 `-p 22001` 与 `-i /home/unit-user/.ssh/id_unit_test` 的 "
        "`TimeoutExpired` —— 走 `str()` 之后端口与密钥路径原样进 `error`, 而那句会落进 "
        "`result.json`、发给外部 LLM 的 Fixer prompt 与跨运行的失败签名", True),

    Row("M2 只摘 IPv4, 不摘 ssh 自己那句 `port <数字>`",
        "ms",
        "    out = _IPV4_WITH_PORT.sub(_MASKED, out)\n"
        '    return _PORT_WORD.sub(r"\\1 " + _MASKED, out)',
        "    out = _IPV4_WITH_PORT.sub(_MASKED, out)\n    return out",
        "存在具体输入可观察: `ssh: connect to host 198.51.100.7 port 22001` 改后变成 "
        "`connect to host [已隐去] port 22001` —— 摘了「地址」却没摘「地址的一部分」。**本片"
        "第一版就是这样, 是它自己撞红的** (掩码只管 argv 那一路时, ssh 写的 stderr 尾巴里的"
        "端口活了下来)", True),

    Row("M3 摘值时把旗标一起吃掉 (诊断能力回归)",
        "ms",
        '    out = _ARGV_SECRET_FLAG.sub(r" \\1 " + _MASKED, text)',
        '    out = _ARGV_SECRET_FLAG.sub(" " + _MASKED, text)',
        "存在具体输入可观察: `ssh -p 22001 -i <密钥>` 改后连 `-p`/`-i` 一起没了 ⇒ 读者分不清"
        "「这一路带过端口/密钥」与「什么都没带」—— 掩码顺手把诊断吃掉, 那是另一种「看不见」", True),

    Row("M4 `_cmd_shape` 把 argv 的最后一项也拼进文本",
        "ms",
        '        return f"{head} +{max(len(cmd) - 1, 0)} 个参数"',
        '        return f"{head} {\' \'.join(str(c) for c in cmd[1:])}"',
        "存在具体输入可观察: 远端命令行 (`/usr/bin/python3 run_kernel.py`) 带着 work_dir 的"
        "绝对路径与 runner 文件名活下来 —— 「看起来无害的东西」正是这一族的入口, 而形状判据的"
        "意思是**只留程序名与参数个数**", True),

    Row("M5 掩码扩到本地那一路 (候选 traceback 被打码)",
        "rx", "            result.error = str(e)", "            result.error = safe_error_text(e)",
        "存在具体输入可观察: 一条报错文本里带 IPv4 形状与四段版本号的**本地**候选核 —— 改后被"
        "远端那一路的掩码吃掉, 而候选报错是 Fixer 唯一能读到的被修对象信息。这一行钉的是本片的"
        "**边界**, 不是它的功能: 范围写宽了会把缺陷换成「看不见」", True),

    Row("M6 环境自检那条仍是 `str(e)[:80]`",
        "env",
        '            report.add(CheckItem("ssh", False, safe_error_text(e)[:120],\n'
        '                                 fix_hint="检查 SSH 免密登录与网络"))',
        '            report.add(CheckItem("ssh", False, str(e)[:80],\n'
        '                                 fix_hint="检查 SSH 免密登录与网络"))',
        "存在具体输入可观察: 自检的 `ssh` 那条走的是同一个 `_ssh_exec`, 超时异常带整条 argv ⇒ "
        "报告详情里端口与密钥路径活下来, 而报告是落进产物的 (第五个读者)", True),

    Row("M7 那一格不再读 `returncode` (scp 的 255 被读成「听到了回答」)",
        "rx",
        '    return transport_outcome(\n'
        '        exc, remote_rc=getattr(exc, "ssh_rc", getattr(exc, "returncode", None)))',
        '    return transport_outcome(exc, remote_rc=getattr(exc, "ssh_rc", None))',
        "存在具体输入可观察: `subprocess.run(scp_cmd, check=True)` 以 255 失败 (部署那一路打的是 "
        "`CalledProcessError`, 码在 `.returncode` 而不是 `_ssh_exec` 挂的 `.ssh_rc` 上) ⇒ 改后读成 "
        "`answered`, 也就是「上传到底有没有发生」这一格又被擦掉一次 —— 第 1 片立的那格在第二个"
        "出口上没穿上", True),

    Row("M8 反向: 把 `CalledProcessError` 整个列进「没听到回答」的名字表",
        "ms",
        '_NO_RESPONSE_EXCEPTIONS: tuple = (\n    "TimeoutExpired",',
        '_NO_RESPONSE_EXCEPTIONS: tuple = (\n    "CalledProcessError",\n    "TimeoutExpired",',
        "存在具体输入可观察: 一个 `rc=1` 的 scp 失败是「它跑完了、报回来了」—— 那是回答。"
        "按异常名一刀切会把它读成 unknown, 于是清点里的「不知道成没成」被系统性夸大 (与 #53 "
        "「缺证据被当成否定证据」同一族的**反方向**: 把证据当成缺证据)", True),

    Row("M9 三条 dict 出口里有一条自己写 `str(e)` 而不用共用的 `_transport_failure`",
        "rx",
        '                "action": "verify", "code": kernel_code,\n'
        '                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,\n'
        '                "max_error": max_error, "precision": precision,\n'
        '                "reference_source": reference_source or "",\n'
        '            })\n'
        "        except Exception as e:\n"
        "            return _transport_failure(e)   # 带「有没有听到回答」那一格 (见 _transport_failure)",
        '                "action": "verify", "code": kernel_code,\n'
        '                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,\n'
        '                "max_error": max_error, "precision": precision,\n'
        '                "reference_source": reference_source or "",\n'
        '            })\n'
        "        except Exception as e:\n"
        '            return {"success": False, "error": str(e)}',
        "存在具体输入可观察: 超时输入下那一格键直接消失 (`KeyError`), 而错误文本带着 argv —— "
        "本片的三个 dict 出口共用一个实现正是「两处各写一份就会有一份忘了掩码」的防线 (I22 那一族)", True),
]
