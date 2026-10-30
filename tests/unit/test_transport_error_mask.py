"""远端异常的文本过落盘边界之前的一道掩码 —— #102 第 2 片 (doc 60 §四 第 2 条)

外部一手 (Limbo `2609.29095v1` 行 226-232 那种「三种世界对客户端同形」的说法是第 1 片的依据;
这一片用的是它另一半: **回执的形状本身也是副作用** —— 一次失败的调用会把它的传输细节留在产物里)。

我方那处的形状（doc 60 §四 第 2 条逐字登记的欠账）:

    `error=str(e)` 那条不许再把异常原文直接当判决输入: 整条 ssh argv 会进 `error_log` →
    `result.json` 与 Fixer prompt, 而今天没出事只因为这些值默认缺席命令行, **不是一道掩码**。

⚠️ 这一句的关键词是「今天没出事」: 脱敏靠的是 `_build_ssh_base` 在 port/key/user **未显式给出**
时不把它们写进命令行 (交由 `~/.ssh/config` 解析)。那是**缺席**，不是掩码 —— 配置里一旦给出这三项
之一 (换一台没有 ssh config 的机器、或 CI 里直接传参), 那句 `str(TimeoutExpired)` 就带着
`-p <端口>`、`-i <密钥路径>`、`<用户>@<主机>` 进下面三个读者:

  1. `result.json` 里的 `error` / `error_log` —— 未跟踪, 但会被搬动、贴进 issue、塞进快照;
  2. **发给外部 LLM 的 Fixer prompt** (`fixer.py:508/569` 直接把错误文本拼进消息);
  3. **跨运行持久化的失败签名** (`fixer.py:209` 从错误文本里取一段当签名 ⇒ 写进 kernel_db 语料)。

`_truncate_error` 保头保尾也救不了：argv 的秘密正好在**头部** (`Command '['ssh', …, '-p', …`)。

**刻意的范围**（也不假装它覆盖了一切）:
  · 只掩**远端那一路**。`LocalExecutor` 的候选 traceback 原样进 prompt —— 那里的 `@triton.jit`、
    版本号与路径是被修的对象, 掩掉就等于把 Fixer 需要的信息吃掉 (有一条用例钉住这条不对称)。
  · 远端**载荷自带的** `result["error"]` (候选在远端跑出来的 traceback) 同样原样保留: 那不是
    传输细节。本片的靶子是「异常原文」, 不是「一切远端文本」。
  · `HTTPExecutor` 的两条各按去处处理: 它的 `/health` 探测 reason **掩了** (那份 reason 与 ssh
    那一路同一个去处 —— `Resolution.attempts`，将来要入库)；它的 compile 那条 raise 走的是
    `e.reason` 而不是整条 URL (`urllib` 的 `str()` 里不含 URL)，不在泄漏面上 (清点结论见 doc 60)。
  · **print 出口没掩** (如 `_pinned_gpu` 那句 `print(…: {e})`): 它只进被 `capture_execution`
    收到的 stdout 日志, 不进判决、不进签名。这一格留在这里当**已知未做**, 不假装覆盖。

变异标定（2026-09-30 实跑，基线 **19 passed**，表在 `benchmarks/mut_specs/transport_error_mask.py`）:

    M1 `safe_error_text` 退回 `str(exc)` (整条 argv 又进落盘边界)
    M2 掩码只摘 IPv4, 不摘 ssh 自己那句 `port <数字>` (**本片第一版的真红法**)
    M3 摘值时把旗标一起吃掉 ⇒ 「这一路带过端口/密钥」的线索没了 (诊断能力回归)
    M4 `_cmd_shape` 把 argv 最后一项 (远端命令行) 也拼进文本
    M5 掩码范围扩到本地那一路 (候选 traceback 被打码, Fixer 看不见被修对象)
    M6 环境自检那条仍是 `str(e)[:80]`
    M7 `transport_outcome` 不再读 `returncode` (scp 的 255 被读成「听到了回答」)
    M8 反向: 把 `CalledProcessError` 也列进「没听到回答」的名字表 (凡带返回码都不知成没成)
    M9 三条 dict 出口里有一条自己写 `str(e)` 而不用共用的 `_transport_failure`
    M10 探连通那一路的 reason 仍是 `f"{type(e).__name__}: {e}"` (argv 进 `Resolution.attempts`)
    M11 同一条尾巴的第二个出口 (HTTP 那一路的 probe reason)

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/unit/test_transport_error_mask.py -q
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.environment import EnvironmentChecker  # noqa: E402
from operator_agent.measurement import (TRANSPORT_ANSWERED, TRANSPORT_KEY,  # noqa: E402
                                        TRANSPORT_UNKNOWN, safe_error_text,
                                        mask_transport_secrets)
from operator_agent.remote_executor import LocalExecutor, SSHExecutor  # noqa: E402

# 全部是**合成**值 (RFC 5737 的文档地址段 + 不存在的用户/路径)。真实端口、真实用户名
# 与真实密钥文件名一律不许进这个文件 —— 那正是 `tests/hygiene/test_repo_hygiene.py` 的靶子。
FAKE_PORT = "22001"
FAKE_KEY = "/home/unit-user/.ssh/id_unit_test"
FAKE_HOST = "unit-gpu.example.invalid"
FAKE_USER = "unit-user"
FAKE_IP = "198.51.100.7"

SECRETS = (FAKE_PORT, FAKE_KEY, f"{FAKE_USER}@{FAKE_HOST}", FAKE_IP, "id_unit_test",
           "/home/unit-user")


def _ssh_argv(remote_cmd: str = "/usr/bin/python3 run_kernel.py") -> list:
    """`_build_ssh_base()` 在**显式给了** port/key/user 时会产出的那种 argv"""
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes", "-p", FAKE_PORT, "-i", FAKE_KEY,
            f"{FAKE_USER}@{FAKE_HOST}", remote_cmd]


def _assert_no_secrets(case: unittest.TestCase, text: str) -> None:
    for needle in SECRETS:
        case.assertNotIn(needle, text, f"秘密 {needle!r} 活了下来: {text!r}")


class TestMaskRules(unittest.TestCase):
    """判据本体: 哪些形状会被摘掉、哪些必须原样留着"""

    def test_the_whole_argv_never_survives(self):
        out = safe_error_text(subprocess.TimeoutExpired(_ssh_argv(), 120))
        _assert_no_secrets(self, out)
        self.assertIn("TimeoutExpired", out, "异常种类要留着 —— 否则读者只知道「失败了」")
        self.assertIn("ssh", out, "程序名是唯一保留的「哪一步」线索")
        self.assertIn("120", out, "超时秒数不是秘密, 而且是有用的诊断")
        # 「形状」不包括 argv 的**内容**: 最后那一项是远端命令行, 它会带着 work_dir 的
        # 绝对路径与 runner 文件名活下来 (M4 打的就是「顺手多带一点看起来无害的东西」)。
        self.assertNotIn("run_kernel.py", out, out)

    def test_called_process_error_uses_the_same_shape(self):
        out = safe_error_text(subprocess.CalledProcessError(255, _ssh_argv()))
        _assert_no_secrets(self, out)
        self.assertIn("rc=255", out)

    def test_ssh_own_stderr_loses_host_and_port(self):
        """**M2 的证人, 也是本片第一版的真红法**: 只摘 IPv4 时端口活了下来

        ssh 自己写的这句话不是 argv, 是 stderr 尾巴 —— 它进的是 `_ssh_exec` 那句
        `SSH exec failed [rc=…]: {detail}`, 所以掩码必须管**文本**而不只管异常对象。
        """
        raw = (f"ssh: connect to host {FAKE_IP} port {FAKE_PORT}: Connection timed out\n"
               f"{FAKE_USER}@{FAKE_HOST}: Permission denied (publickey)")
        out = mask_transport_secrets(raw)
        _assert_no_secrets(self, out)
        self.assertIn("Connection timed out", out, "剩下的半句还是有用的")
        self.assertIn("Permission denied", out)

    def test_a_diagnostic_with_no_secrets_survives_verbatim(self):
        """**反向对照**: 掩码不许顺手吃掉诊断信息, 否则它会以「更安全」的名义让产物变哑"""
        raw = ("SSH exec failed [rc=1]: TritonCompilationError: expected an expression "
               "at line 42 in kernel.py")
        self.assertEqual(raw, mask_transport_secrets(raw))

    def test_a_decorator_name_is_not_read_as_user_at_host(self):
        """`@triton.jit` 这类「前面没有字符的 @」不是 `user@host`"""
        raw = "SyntaxError: @triton.jit requires a function, got NoneType"
        self.assertEqual(raw, mask_transport_secrets(raw))

    def test_three_part_versions_are_not_read_as_ipv4(self):
        """IPv4 那条要四段, 所以 `3.9.6` / `1.22.4` 这类版本号不会被吃掉"""
        raw = "Incompatible torch build: triton 3.1.0 with torch 2.1.0+cu121, numpy 1.22.4"
        self.assertEqual(raw, mask_transport_secrets(raw))

    def test_empty_text_stays_empty(self):
        """空串不等于「掩掉了什么」: 掩码不许把空文本变成占位符"""
        self.assertEqual("", mask_transport_secrets(""))

    def test_the_flag_is_kept_while_its_value_is_masked(self):
        """**M3 的证人**: 摘的是「旗标 + 值」里的**值**; 旗标自己是「哪一步」的一部分"""
        out = mask_transport_secrets(f"run: ssh -p {FAKE_PORT} -i {FAKE_KEY} unit@x")
        _assert_no_secrets(self, out)
        self.assertIn("-p", out)
        self.assertIn("-i", out)


def _ssh_executor() -> SSHExecutor:
    """一个**显式给了** port/key/user 的 executor —— 这正是「缺席不是掩码」那个现场

    默认的 `_build_ssh_base()` 不写这三项 (交由 `~/.ssh/config` 解析), 于是全仓的脱敏一直
    「看起来是对的」; 这里显式给出, 才是掩码要守的那台机器。
    """
    ex = SSHExecutor(host=FAKE_HOST, port=int(FAKE_PORT), user=FAKE_USER,
                     key_path=FAKE_KEY)
    ex._deployed = True          # 跳过 deploy(): 本片只问「失败之后文本长什么样」
    return ex


def _timeout_exc():
    return subprocess.TimeoutExpired(_ssh_argv(), 120)


class TestTheRemoteExits(unittest.TestCase):
    """四个远程出口 + 环境自检那一条: 它们真的走这一份实现吗 (不是只有单测在自证)"""

    def test_execute_carries_masked_text_and_the_cell(self):
        ex = _ssh_executor()
        ex._ssh_exec_json = mock.Mock(side_effect=_timeout_exc())
        out = ex.execute("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertFalse(out.success)
        _assert_no_secrets(self, out.error)
        self.assertEqual(TRANSPORT_UNKNOWN, out.transport_outcome)

    def test_the_three_dict_exits_all_mask(self):
        """`_transport_failure` 是三条「返回 dict」的出口共用的那一个实现 (单一实现守卫)"""
        ex = _ssh_executor()
        ex._ssh_exec_json = mock.Mock(side_effect=_timeout_exc())
        cases = {
            "benchmark": lambda: ex.benchmark("def kernel(A): return A",
                                              {"A": np.ones((4, 4), np.float32)}),
            "verify_function": lambda: ex.verify_function(
                "def kernel(A): return A", {"A": np.ones((4, 4), np.float32)}, "matmul", 1e-3),
            "benchmark_with_baselines": lambda: ex.benchmark_with_baselines(
                "def kernel(A): return A", {"A": np.ones((4, 4), np.float32)}, "matmul"),
        }
        for name, call in cases.items():
            with self.subTest(exit=name):
                out = call()
                _assert_no_secrets(self, str(out.get("error", "")))
                self.assertEqual(TRANSPORT_UNKNOWN, out[TRANSPORT_KEY], name)

    def test_a_plain_remote_failure_text_is_not_mangled(self):
        """反向对照 (不发放): 远端答了「编译失败」时那句原文要能原样到 Fixer 手里"""
        ex = _ssh_executor()
        ex._ssh_exec_json = mock.Mock(side_effect=RuntimeError(
            "SSH exec failed [rc=1]: compile error on remote"))
        out = ex.benchmark("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual("SSH exec failed [rc=1]: compile error on remote", out["error"])
        self.assertEqual(TRANSPORT_ANSWERED, out[TRANSPORT_KEY])

    def test_scp_return_code_255_is_not_read_as_an_answer(self):
        """⚠️ 本片顺手量的第二件事 (第 1 片那一格的第二个出口)

        部署那一路是 `subprocess.run(scp_cmd, check=True)` ⇒ 它打的是 `CalledProcessError`,
        码在 `.returncode` 而不是 `_ssh_exec` 挂的 `.ssh_rc` 上。在那之前 scp 以 255 失败
        (= ssh 自己连不上, 上传到底有没有发生无从知道) 会被读成 `answered` —— 正是第 1 片
        要分开的两种世界被这一路合并掉了。
        """
        ex = _ssh_executor()
        ex._ssh_exec_json = mock.Mock(
            side_effect=subprocess.CalledProcessError(255, ["scp", "-i", FAKE_KEY, "a", "b"]))
        out = ex.benchmark("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual(TRANSPORT_UNKNOWN, out[TRANSPORT_KEY], out)
        _assert_no_secrets(self, str(out.get("error", "")))

    def test_a_non_255_return_code_is_still_an_answer(self):
        """**反向对照 (M8)**: 读 `.returncode` 不等于「凡是带返回码的都不知成没成」

        `rc=1` 的 scp 失败是「它跑完了、报回来了」——那是回答, 不是没听到。
        """
        ex = _ssh_executor()
        ex._ssh_exec_json = mock.Mock(
            side_effect=subprocess.CalledProcessError(1, ["scp", "a", "b"]))
        out = ex.benchmark("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual(TRANSPORT_ANSWERED, out[TRANSPORT_KEY], out)

    def test_the_environment_selfcheck_detail_is_masked(self):
        """第五个读者: 环境自检的那条也会把异常文本写进报告 → `result.json`"""
        def _boom(cmd):
            raise _timeout_exc()
        report = EnvironmentChecker(_boom, "~/.local/bin/uv", ".").check()
        ssh_item = [i for i in report.items if i.name == "ssh"]
        self.assertTrue(ssh_item, "构造前提: 自检确实报了 ssh 那一条")
        detail = ssh_item[0].detail or ""
        _assert_no_secrets(self, detail)
        self.assertIn("TimeoutExpired", detail)

    def test_the_probe_reason_keeps_the_class_but_not_the_argv(self):
        """⚠️ doc 49 §7.8 末那条 ⚠️ 说的就是这个出口: 解析结果**将来**要入库

        `probe_connection()` 的 `reason` 进 `Resolution.attempts` —— 那是被打印、也最可能
        被序列化进产物的那一格。超时输入下它以前是 `f"{type(e).__name__}: {e}"`, 而 `{e}`
        就是整条 ssh argv。
        """
        ex = _ssh_executor()
        ex._ssh_exec = mock.Mock(side_effect=_timeout_exc())
        ok, reason = ex.probe_connection()
        self.assertFalse(ok)
        _assert_no_secrets(self, reason)
        self.assertIn("TimeoutExpired", reason)
        self.assertEqual(1, reason.count("TimeoutExpired"),
                         "带 cmd 那一支的文本里已经有类名了, 再前缀一次就是两处 (与报表读不通)")

    def test_a_name_only_exception_still_names_its_class(self):
        """**反向对照 (with_class 那一半)**: 掩码不许顺手吃掉「哪一类失败」

        doc 49 §3.3 S3 的立片理由就是 reason 要分得开 DNS / 认证 / 超时, 而 `socket.gaierror`
        那类的原话里**没有**类名 —— 少了这一半, 那道掩码就会把本片要守的东西换成另一种看不见。
        """
        ex = _ssh_executor()
        ex._ssh_exec = mock.Mock(side_effect=OSError("nodename nor servname provided"))
        ok, reason = ex.probe_connection()
        self.assertFalse(ok)
        self.assertEqual("OSError: nodename nor servname provided", reason)

    def test_the_http_probe_reason_is_masked_too(self):
        """同一道掩码的第二个探连通出口 (HTTP 那一路) —— 没有它这一行就没有读者"""
        from urllib.error import URLError
        from operator_agent.remote_executor import HTTPExecutor
        ex = HTTPExecutor(base_url="http://203.0.113.9:18080")
        with mock.patch("urllib.request.urlopen",
                        side_effect=URLError(OSError("connect to host 203.0.113.9 port 18080"))):
            ok, reason = ex.probe_connection()
        self.assertFalse(ok)
        self.assertNotIn("203.0.113.9", reason, reason)
        self.assertNotIn("18080", reason, reason)
        self.assertIn("URLError", reason, "类名要留着: 那是「连不上」与「跑完了报错了」的分界")


class TestTheScopeIsExactlyRemote(unittest.TestCase):
    """「只掩远端那一路」这条边界也要有证人 (M5) —— 否则掩码会顺手把被修对象打码"""

    KERNEL_THAT_RAISES = (
        "def kernel(A, B):\n"
        "    raise ValueError('bad shape at 198.51.100.7 for @jit kernel v1.2.3.4')\n"
    )

    def test_local_candidate_traceback_is_not_masked(self):
        """本地那一路的原话要**带秘密形状地**活着穿过掩码边界

        故意把 IPv4 形状、四段版本号与 `@jit` 写进候选的报错文本里: 如果掩码被扩到这一路,
        这条会红。候选报错是 Fixer 唯一能读到的被修对象信息, 打码等于把缺陷换成「看不见」。
        """
        out = LocalExecutor().execute(self.KERNEL_THAT_RAISES,
                                      {"A": np.ones((4, 4), np.float32),
                                       "B": np.ones((4, 4), np.float32)})
        self.assertFalse(out.success)
        self.assertIn(FAKE_IP, out.error, "本地候选的 traceback 不该被远端那一路的掩码吃掉")
        self.assertIn("v1.2.3.4", out.error)
        self.assertIn("@jit", out.error)

    def test_remote_payload_errors_are_not_the_target(self):
        """远端**载荷**报回来的 error (候选在 GPU 上跑出来的文本) 不在本片的靶子里

        它不是传输细节, 而是被修对象的输出 —— 与本片的立片理由正对着: 掩码只管「异常原文」
        这一条通道, 把它扩成「一切远端文本」会连 L2/L3 需要的诊断一起掩掉。
        """
        ex = _ssh_executor()
        ex._ssh_exec_json = mock.Mock(return_value={
            "success": False, "error": "IndexError: shape mismatch at 42",
            TRANSPORT_KEY: TRANSPORT_ANSWERED})
        out = ex.execute("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual("IndexError: shape mismatch at 42", out.error)


if __name__ == "__main__":
    unittest.main()
