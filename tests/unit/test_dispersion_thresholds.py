# -*- coding: utf-8 -*-
"""G4/G5 两个产出方的**静态**守卫。

为什么是静态的：这两个脚本的全部意义在于**单文件 scp 到远端机器上跑**（见 runbook §4
第 0 步），所以它们 import 不到仓里的任何 schema，而且本机根本没有可用的 triton ——
`import benchmarks.calibration.calibrate_dispersion_thresholds` 在开发机上直接 ModuleNotFoundError。
一类「采集器写一个键、门读另一个键」的错误就是这样躲过运行期检查的：门一直 unknown，
而 `--from-probe` 照样把文件读进去、一个错都不报。

所以这里读源码（AST），钉三件事：
  1. 脚本里复制的那几个档位常量必须与 `operator_agent.decide.measurement` 逐字相同；
  2. `PROBE_JSON` 里必须有 `uuid` 与 `noise`，且 `noise` 覆盖采集器要的那三个键；
  3. 形状清单的 `PROBE_JSON` 必须有 `uuid` 与 `shapes`，条目带 `label`/`bytes`。
"""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

from benchmarks.probes.collect_unknown_card_probe import NOISE_THRESHOLD_KEYS
from benchmarks.probes.collect_shape_inventory import main as inventory_main
from benchmarks.probes.collect_shape_inventory import parse_shape
from operator_agent.decide import measurement as m

ROOT = Path(__file__).resolve().parents[2]
DISPERSION = ROOT / "benchmarks/calibration/calibrate_dispersion_thresholds.py"
INVENTORY = ROOT / "benchmarks/probes/collect_shape_inventory.py"


def _module(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_constants(tree):
    """模块层的 `NAME = <literal>` 赋值。只收字面量 —— 那才是可搬走的常量。"""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    return out


def _dict_keys(node):
    """`{...}` 字面量的字符串键（跳过 **展开）。"""
    if not isinstance(node, ast.Dict):
        return set()
    return {k.value for k in node.keys if isinstance(k, ast.Constant)
            and isinstance(k.value, str)}


def _assigned_dict_keys(tree, name):
    """找 `name = {...}` 的赋值，返回其字符串键集。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return _dict_keys(node.value)
    return set()


def _all_dict_key_sets(tree):
    """模块里每一个字典字面量的键集。"""
    return [_dict_keys(n) for n in ast.walk(tree) if isinstance(n, ast.Dict)]


def _noise_dict_key_sets(tree):
    """**真正变成 `noise` 段**的那些字典的键集。

    只认两条路径：`noise = {...}` 与 `noise.update({...})`。先前用「全模块任何一个字典
    覆盖到三键就行」，于是把 payload 产出处那一个键改名时守卫仍然是绿的（M2b 实测逃掉）——
    别的地方还留着一份完整的键集，而门读到的是 payload 那一份。
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "noise":
            out.append(_dict_keys(node.value))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "update" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "noise" and node.args:
            out.append(_dict_keys(node.args[0]))
    return out


def _json_dumps_payload_keys(tree):
    """所有 `json.dumps(...)` 载荷的键集，按出现顺序返回列表。

    两种写法都要认：直接塞字面量 `json.dumps({...})`，以及先 `payload = {...}` 再
    `json.dumps(payload)`（形状清单用的就是后者）。只认前者会让守卫对后一种写法
    **静默失效** —— 它会在「什么都没检查」的情况下变绿。
    """
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            keys = _dict_keys(node.value)
            if keys:
                assigned[node.targets[0].id] = keys
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dumps" and node.args):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Dict):
            keys = _dict_keys(arg)
        elif isinstance(arg, ast.Name):
            keys = assigned.get(arg.id, set())
        else:
            keys = set()
        if keys:
            out.append(keys)
    return out


class TestCopiedConstantsAreStillTheSame(unittest.TestCase):
    """单文件 scp 的代价是常量被抄了一份。抄错不会崩，只会让档位判定与生产悄悄分叉。"""

    def setUp(self):
        self.consts = _module_constants(_module(DISPERSION))

    def test_launch_dominance_thresholds_match(self):
        self.assertEqual(self.consts["LAUNCH_DOMINANCE_VALID"], m.LAUNCH_DOMINANCE_VALID)
        self.assertEqual(self.consts["LAUNCH_DOMINANCE_UNINFORMATIVE"],
                         m.LAUNCH_DOMINANCE_UNINFORMATIVE)

    def test_dispersion_thresholds_match(self):
        self.assertEqual(self.consts["DISPERSION_DEGRADED_INHERITED"], m.DISPERSION_DEGRADED)
        self.assertEqual(self.consts["DISPERSION_UNINFORMATIVE"], m.DISPERSION_UNINFORMATIVE)

    def test_contamination_threshold_matches(self):
        self.assertEqual(self.consts["INHERITED_UTIL_PCT"], m.CONTAMINATION_UTIL_PCT)

    def test_inherited_band_is_the_shipped_table(self):
        """`INHERITED_BAND` 不是随手写的对照值：它就是 NOISE_BAND 本身。

        对不上就意味着「撞了继承值要报」这条守卫在拿错的靶子比。
        """
        shipped = {v.value: f for v, f in
                   ((m.Validity.VALID, m.NOISE_BAND[m.Validity.VALID]),
                    (m.Validity.DEGRADED, m.NOISE_BAND[m.Validity.DEGRADED]),
                    (m.Validity.UNINFORMATIVE, m.NOISE_BAND[m.Validity.UNINFORMATIVE]))}
        self.assertEqual(self.consts["INHERITED_BAND"], shipped)

    def test_tiers_are_the_band_keys(self):
        self.assertEqual(set(self.consts["TIERS"]),
                         {v.value for v in m.NOISE_BAND})


class TestNoisePayloadIsReadableByTheCollector(unittest.TestCase):

    def setUp(self):
        self.tree = _module(DISPERSION)

    def test_payload_carries_uuid_and_noise(self):
        payloads = _json_dumps_payload_keys(self.tree)
        self.assertTrue(any({"uuid", "noise"} <= keys for keys in payloads),
                        "没有任何一个 PROBE_JSON 载荷同时带 uuid 与 noise："
                        "采集器会拒绝盖来源戳，G4 继续 unknown")

    def test_noise_block_covers_the_three_keys_the_collector_requires(self):
        # 字面量，不用采集器那个常量迭代：常量被削掉一项时，这组用例会跟着一起缩而全绿。
        required = ("band", "contamination_util_pct", "dispersion_degraded")
        self.assertEqual(set(NOISE_THRESHOLD_KEYS), set(required),
                         "采集器要求的三键变了：产出方与用例都要跟着改")
        key_sets = _noise_dict_key_sets(self.tree)
        covered = any(set(required) <= keys for keys in key_sets)
        self.assertTrue(covered, "变成 noise 段的字典没有一个同时给出 %r：采集器会拒"
                        "（半截标定）。见到的是 %r" % (required, key_sets))

    def test_band_is_filled_per_tier_not_as_one_scalar(self):
        """G4 比的是分档字典。给一个标量会被门判 fail，给三个档才等于仓里那张表。"""
        self.assertIn('band["valid"]', DISPERSION.read_text(encoding="utf-8"),
                      "band 没有按档写入的痕跡")


class TestRefusalsDoNotLookLikeResults(unittest.TestCase):
    """`derive()` 拒绝时必须返回 None，**不能**返回退出码 2。

    2026-11-04 在 8 卡共享 H20 上实测：`--floor-us 20.224`（不冲 L2 的地板）配本脚本
    （冲 256MB）触发「两个读数不是同一把尺」这条拒绝，`derive()` 返回了整数 2，调用方
    只把 `None` 当失败 ⇒ 那个 2 被 append 进 runs，100 行后炸成
    `TypeError: 'int' object is not subscriptable`。拒绝理由（混了两把尺）被埋在
    TypeError 底下，看起来像脚本坏了。拒绝路径必须长得像拒绝。

    这里用 AST 而不是真调用：本模块 import 期就要 `triton.language`，无 GPU/无 triton
    的测试机上导不进来；而「拒绝不许返回整数」本来就是一条只读源码就够的判据。
    """

    def setUp(self):
        self.tree = _module(DISPERSION)

    def _derive_returns(self):
        out = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "derive":
                out.extend(n for n in ast.walk(node) if isinstance(n, ast.Return))
        self.assertTrue(out, "在 %s 里找不到 derive() 的 return —— 用例自己失效了" % DISPERSION)
        return out

    def test_refusals_return_none_not_an_exit_code(self):
        ints = [n for n in self._derive_returns()
                if isinstance(n.value, ast.Constant) and isinstance(n.value.value, int)
                and not isinstance(n.value.value, bool)]
        self.assertEqual(ints, [],
                         "derive() 里出现 `return <整数>`：调用方只把 None 当失败，整数会被"
                         "当成结果继续往下算（2026-11-04 真机踩到，报成 TypeError）")


class TestShapeInventoryPayload(unittest.TestCase):

    def setUp(self):
        self.tree = _module(INVENTORY)

    def test_payload_carries_uuid_and_shapes(self):
        payloads = _json_dumps_payload_keys(self.tree)
        self.assertTrue(any({"uuid", "shapes"} <= keys for keys in payloads),
                        "形状清单的 PROBE_JSON 必须同时带 uuid 与 shapes")

    def test_shape_entries_carry_label_and_bytes(self):
        text = INVENTORY.read_text(encoding="utf-8")
        for token in ('"label"', '"bytes"'):
            self.assertIn(token, text, "形状条目少了 %s：门读不到就会判 unknown" % token)

    def test_refuses_an_empty_declaration(self):
        """空清单会让 G5 落到 `not g5` 那一支判 unknown，看起来像「还没核对」。

        这条必须是**行为**断言：先前写成「源码里有没有那句话」的文本检查，变异 M8
        （`if not args.shape:` → `if False:`）直接逃掉 —— 字符串还在，只是再也没人走到。
        """
        with self.assertRaises(SystemExit) as caught:
            inventory_main([])
        self.assertIn("至少给一个", str(caught.exception))

    def test_parses_the_same_shape_syntax_as_the_cli(self):
        self.assertEqual(parse_shape("1024,768"), (1024, 768))
        self.assertEqual(parse_shape(" 64, 64, 64 "), (64, 64, 64))

    def test_rejects_malformed_shapes_before_allocating(self):
        for bad in ("", "1024,", "abc", "0,768", "-4,8"):
            with self.assertRaises(SystemExit, msg=bad):
                parse_shape(bad)

    def test_the_written_file_is_directly_feedable_to_from_probe(self):
        """`--out` 写的必须是带 `PROBE_JSON ` 前缀的那一份。

        先前写裸 JSON，采集器报「shapes.txt 里没有 PROBE_JSON 行」——文件在手，
        `--from-probe` 却读不了。这条用例直接拿采集器的读取器去读它写出来的文件。
        """
        from benchmarks.probes.collect_shape_inventory import emit
        from benchmarks.probes.collect_unknown_card_probe import load_probes
        payload = {"uuid": "GPU-x", "shapes": [{"label": "1024x768 fp16", "bytes": 1}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shapes.txt")
            emit(json.dumps(payload), path)
            self.assertEqual(load_probes([path]), [payload])


if __name__ == "__main__":
    unittest.main()