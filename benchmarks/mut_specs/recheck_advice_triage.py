"""`tests/test_recheck_arxiv_anchors.py::TestAdviceIsTriaged` 那张变异表的可执行形态

这一片修的是**检查器自己给错建议**：五篇里有一篇瞬时取回失败时，`recheck_arxiv_anchors.py`
的读数会掉到 22/36，而收尾那句无条件写着「要改的是文档里那张表」——照做就把一次网络抖动
「核」成了一份错的表。与 I21/I32 同族：**读数要连「它是在什么条件下取的」一起说**。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/recheck_advice_triage.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {"rc": "benchmarks/recheck_arxiv_anchors.py"}

TARGETS = ["tests/test_recheck_arxiv_anchors.py"]

ROWS = [
    Row("R1 分诊整条不走 (永远说「改表」)", "rc",
        "    if failed or n_no_bytes:",
        "    if False:",
        "造一行 `no_bytes` + failed=['9.9v1'] ⇒ 收尾仍写「要改的是文档里那张表」，"
        "而那句正把取数失败指导成改表"),

    Row("R2 只认 fetch_failed，不认表里的 no_bytes 行", "rc",
        "    if failed or n_no_bytes:",
        "    if failed:",
        "只给 `rows` 不给 `fetch_failed` (表的行状态说没取到、而 id 列表空) ⇒ 误判成表错"),

    Row("R3 全绿时也冒出一句建议", "rc",
        '    bad = [r for r in rows if r.get("status") not in ("ok", "cited")]\n'
        "    if not bad:\n"
        '        return ""',
        '    bad = [r for r in rows if r.get("status") not in ("ok", "cited")]\n'
        "    if False:\n"
        '        return ""',
        "两行都 ok/cited 时不该有任何输出 —— 只在报警那一侧取样测不出方向，"
        "这条是反向对照"),

    Row("R4 main 不问建议 (硬编码那句)", "rc",
        "        advice = closing_advice(rows, fetch_failed)",
        '        advice = "⇒ 上面这些数要改的是**文档里那张表**, 不是这里 (表是唯一真源)"',
        "假 urlopen 全抛错跑一次 `main` ⇒ 退出码仍是 1，但收尾不再带 id、也不说「没取回字节」。"
        "AST 那条同时红 (main 里没有 closing_advice 这个 Name)"),

    # ⚠️ 这一行的锚点 2026-09-30 被本片自己撞模糊了: 新加的「HTTPError 且正文为空」分支里也有
    #    一句一模一样的 `fetch_failed.append(arxiv_id)` ⇒ 表直接报「锚点出现 2 次」。
    #    把上一行 (`except Exception as exc:`) 一起写进锚点来定住位置 —— 那是唯一的一条。
    Row("R5 except 里不记 id (接线断)", "rc",
        "        except Exception as exc:  # noqa: BLE001\n            fetch_failed.append(arxiv_id)\n",
        "        except Exception as exc:  # noqa: BLE001\n            pass  # (变异掉: 不记是哪个 id)\n",
        "假 urlopen 全抛错 ⇒ 收尾那句印成「本轮 0 个 id 没取回字节」，具体是哪个查不到"),
]
