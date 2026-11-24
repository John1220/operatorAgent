"""回执对账工具自己的守卫：它必须**真的**会红。

一个只会说 ok 的对账脚本比没有更糟 —— 它会把「重放不成立」签成成立。
"""
import json
import shutil
from pathlib import Path

from benchmarks.repro import compare_replays as tool

PACKAGE = 'fused_ln_tail_20261118'
COMMITTED = tool.ROOT/PACKAGE


def test_a_package_matches_itself(tmp_path):
    fresh = tmp_path/PACKAGE
    shutil.copytree(COMMITTED, fresh)
    assert tool.compare(COMMITTED, fresh, PACKAGE) == []


def test_a_changed_judgement_is_a_difference(tmp_path):
    fresh = tmp_path/PACKAGE
    shutil.copytree(COMMITTED, fresh)
    path = fresh/'replay-candidate.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    value['levels'] = {'L1': True, 'L2': True}          # 判据变了
    path.write_text(json.dumps(value), encoding='utf-8')
    problems = tool.compare(COMMITTED, fresh, PACKAGE)
    assert any('levels' in problem for problem in problems)


def test_run_identity_fields_describe_this_run_and_are_not_compared(tmp_path):
    """三个路径字段每次跑都不同（`run/` 下的候选目录名带哈希），差别本身不是漂移。"""
    fresh = tmp_path/PACKAGE
    shutil.copytree(COMMITTED, fresh)
    path = fresh/'replay-candidate.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    value['evidence'] = str(fresh/'run'/'candidate-other'/'verify-1'/'result.json')
    value['run_dir'] = str(fresh/'run'/'candidate-other')
    path.write_text(json.dumps(value), encoding='utf-8')
    assert tool.compare(COMMITTED, fresh, PACKAGE) == []


def test_a_missing_receipt_is_a_failure_not_a_skip(tmp_path):
    fresh = tmp_path/PACKAGE
    fresh.mkdir()
    problems = tool.compare(COMMITTED, fresh, PACKAGE)
    assert problems and '没有重跑出' in problems[0]
