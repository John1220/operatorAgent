"""Search selection must remain auditable after ordinary artifact reload."""
import copy
import json
from pathlib import Path

import pytest

from operator_agent.artifact_store import ArtifactStore
from operator_agent.report import build_report
from operator_agent.run_contract import audit_value, json_value
from operator_agent.schemas import KernelArtifact, MemoryPlan, PipelineStage, UnifiedSketch
from tests.unit.test_search_objective import execute


def test_selected_candidate_and_history_survive_save_load(tmp_path):
    art, _, _, _, generated = execute(tmp_path, ratios=((2., 2.), (3., .95)))
    raw = json.loads((Path(art.saved_dir)/'result.json').read_text())
    loaded = ArtifactStore().load(art.saved_dir)
    for key in ('search_objective', 'candidate_selection', 'stop'):
        assert raw[key] == art.metadata[key] == loaded.metadata[key]
    assert loaded.code == generated[0] != generated[1]
    assert loaded.sketch.to_dict() == art.sketch.to_dict()
    assert loaded.iteration_history == art.iteration_history
    assert loaded.benchmark_data == json_value(art.benchmark_data)
    for stored, current in zip(loaded.verification_results, art.verification_results):
        assert (stored.level, stored.passed) == (current.level, current.passed)
        expected = json_value(current.metrics)
        assert all(expected[k] == v for k, v in stored.metrics.items())
        # Decision publication annotates the in-memory L3 and benchmark after
        # verification.json was saved; it does not replace measurement evidence.
        assert expected.keys() - stored.metrics.keys() <= {
            'claim_level', 'objective', 'implementation_decision'}
    text = build_report(artifact=loaded)
    assert '声明 workload 的搜索选优' in text
    assert '第 1 轮；workload score=2.0×' in text
    assert '回退到较早最佳候选: 是' in text
    assert 'reward (对本轮前 best)' in text
    assert 'SLOW' in text and '没有 reward/进度比较' in text
    assert '仅为本会话已观测候选' in text


def test_sketch_uses_registered_schema_roundtrip_not_dataclass_keyword_names(tmp_path):
    sketch = UnifiedSketch(algorithm='layernorm', version=2,
        memory_plan=MemoryPlan(['X'], ['smem'], ['acc']),
        pipeline=[PipelineStage('load', 'global', 'local', 'coalesced')],
        optimization_hints=['old', 'new'], hints_this_version=['new'], target_dsl='triton')
    store = ArtifactStore(str(tmp_path))
    saved = store.save(KernelArtifact(code='source', sketch=sketch))
    assert store.load(str(saved)).sketch.to_dict() == sketch.to_dict()


def test_only_registered_search_keys_are_serialized_with_explicit_nonfinite_diagnostics(tmp_path):
    metadata = dict(search_objective=dict(identity={'schema':'audit'}, best=None,
                                         diagnostic=[float('nan'), float('inf')]),
                    candidate_selection=dict(selected_round=None),
                    last_probe_stop=dict(reason='passed'), unregistered='private-context')
    store = ArtifactStore(str(tmp_path));saved = store.save(KernelArtifact(code='source', metadata=metadata))
    raw = json.loads((saved/'result.json').read_text())
    for key in ('search_objective', 'candidate_selection', 'last_probe_stop'):
        assert raw[key] == audit_value(metadata[key]) == store.load(str(saved)).metadata[key]
    assert 'metadata' not in raw and 'unregistered' not in raw and 'private-context' not in str(raw)
    assert '$nonfinite' in str(raw)


def test_no_search_keys_or_report_section_are_invented_for_legacy_artifacts(tmp_path):
    store = ArtifactStore(str(tmp_path));saved = store.save(KernelArtifact(code='source'))
    raw = json.loads((saved/'result.json').read_text());loaded = store.load(str(saved))
    for key in ('search_objective', 'candidate_selection', 'last_probe_stop'):
        assert key not in raw and key not in loaded.metadata
    assert '声明 workload 的搜索选优' not in build_report(artifact=loaded)
    assert 'reward (对本轮前 best)' not in build_report(artifact=loaded)


def test_no_qualified_candidate_report_does_not_claim_a_winner(tmp_path):
    def floor(p, i, n):p['candidate']['median_time_ms'] = .001
    art, _, _, _, _ = execute(tmp_path, change=floor)
    loaded = ArtifactStore().load(art.saved_dir)
    assert loaded.metadata['search_objective']['best'] is None
    text = build_report(artifact=loaded)
    assert '没有本会话合格赢家' in text and '保留候选: 第' not in text


@pytest.mark.parametrize('drift', ('code', 'round'))
def test_report_does_not_grant_winner_when_selection_identity_disagrees(tmp_path, drift):
    art, _, _, _, _ = execute(tmp_path)
    art = copy.deepcopy(art)
    if drift == 'code':art.code += '# changed\n'
    else:art.metadata['candidate_selection']['selected_round'] = 100
    text = build_report(artifact=art)
    assert '选优身份与返回源码不一致' in text and '保留候选: 第' not in text


def test_degraded_probe_stop_remains_distinct_from_final_hold_after_reload(tmp_path):
    from tests.unit.test_workload_evidence import task
    def degraded(p, i, n):
        if i == 1 and n == 32:p['candidate']['dispersion_iqr_over_median'] = .2
    art, _, _, _, _ = execute(tmp_path, ratios=((1.5, 1.5), (2., 2.)),
                              change=degraded, t=task(min_speedup=1.8))
    loaded = ArtifactStore().load(art.saved_dir)
    assert loaded.metadata['last_probe_stop']['reason'] == 'passed'
    assert loaded.metadata['stop']['reason'] == 'qualified_candidate_hold'
    assert not loaded.all_passed
    text = build_report(artifact=loaded)
    assert '末次探针停因: `passed`；不替代最终停因' in text
