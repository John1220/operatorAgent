"""Generation receipts survive real adapters; all models/devices are local fakes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.generate.agents import CoderAgent
from operator_agent.generate.fixer import FixerAgent
from operator_agent.generate.generation_context import records_cursor, records_since, validate_candidate_record
from operator_agent.generate.repair_loop import repair
from operator_agent.core.run_contract import code_hash
from tests.unit.test_execute_inner_behavior import _CODE, _Llm, _task
from tests.unit.test_fixer_generation_records import ORIGINAL, FIXED, Replies, task, failures, reply
from tests.unit.test_repair_loop import FakeVerifier, bad, ok


def conductor(tmp_path, llm=None):
    instance = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path='off'),
        llm=llm or _Llm(), use_coder_pipeline=False, max_optimize_rounds=1,
        run_root=str(tmp_path/'run'))
    instance.store.output_dir = tmp_path/'artifacts'
    return instance


def inputs():
    return {'A': np.eye(16, dtype=np.float32), 'B': np.eye(16, dtype=np.float32)}


def test_production_single_path_returns_and_saves_trace_and_resets_next_run(tmp_path):
    instance = conductor(tmp_path)
    assert instance.coder.pipeline is None
    first = instance.execute(_task(), inputs())
    records = first.metadata['candidate_records']
    frozen = copy.deepcopy(records)
    assert len(records) == 1
    assert records[0]['producer'] == 'CoderAgent'
    assert records[0]['source'] is None and records[0]['source_retention'] == 'hash_only'
    assert records[0]['source_hash'] == code_hash(first.code)
    assert instance.coder.last_candidate_record['source'] == first.code == _CODE.strip()
    assert first.metadata['selected_source_hash'] == code_hash(first.code)
    assert validate_candidate_record(records[0])
    sidecar = json.loads((tmp_path/'run/generation-records.json').read_text())
    assert sidecar['candidate_records'] == records
    assert sidecar['selected_source_hash'] == code_hash(first.code)
    saved = json.loads((Path(first.saved_dir)/'result.json').read_text())
    assert saved['candidate_records'] == records
    loaded = instance.store.load(first.saved_dir)
    assert loaded.metadata['candidate_records'] == records
    assert loaded.metadata['selected_source_hash'] == code_hash(first.code)
    second = instance.execute(_task(), inputs())
    assert len(second.metadata['candidate_records']) == 1
    assert first.metadata['candidate_records'] == frozen
    assert first.metadata['candidate_records'] is not second.metadata['candidate_records']


def test_model_failure_is_persisted_before_finalize_without_retry(tmp_path):
    error = TimeoutError('private transport details')
    class FailedModel(_Llm):
        calls = 0
        def chat(self, *args, **kwargs):
            self.calls += 1
            raise error
    model = FailedModel()
    instance = conductor(tmp_path, model)
    with pytest.raises(TimeoutError) as caught:
        instance.execute(_task(), inputs())
    assert caught.value is error and model.calls == 1
    evidence = json.loads((tmp_path/'run/generation-records.json').read_text())
    assert evidence['selected_source_hash'] is None
    record, = evidence['candidate_records']
    assert record['status'] == 'model_failed' and record['source'] is None
    assert 'private transport details' not in json.dumps(record)


def test_evidence_write_failure_does_not_mask_model_error(tmp_path, monkeypatch):
    instance = conductor(tmp_path)
    fixer = FixerAgent(llm=Replies([TimeoutError('original model error')]))
    from operator_agent.core import run_contract
    monkeypatch.setattr(run_contract, 'atomic_json', lambda *a, **k: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(TimeoutError, match='original model error'):
        instance._call_generator(fixer, fixer.run_with_action, ORIGINAL, task(), failures())
    assert instance.candidate_records[0]['status'] == 'model_failed'


def test_repair_loop_preserves_full_receipt_and_final_source_binding():
    model = Replies([reply(FIXED)])
    fixer = FixerAgent(llm=model)
    verifier = FakeVerifier([bad(), ok()])
    result = repair(ORIGINAL, task(), {'X': np.ones((2, 3), dtype=np.float32)},
                    verifier, fixer, max_rounds=2)
    record, = result['candidate_records']
    assert record == result['rounds'][0]['candidate_records'][0]
    assert record['context']['input_metadata_status'] == 'known'
    assert record['parent_source_hash'] == code_hash(ORIGINAL)
    assert record['source_hash'] == result['selected_source_hash'] == code_hash(FIXED)
    assert verifier.calls == [ORIGINAL, FIXED] and len(model.sent) == 1
    assert record['correctness_proven'] is False


def test_failed_repair_keeps_receipt_and_selects_original_source():
    fixer = FixerAgent(llm=Replies([TimeoutError('offline')]))
    verifier = FakeVerifier([bad()])
    result = repair(ORIGINAL, task(), {}, verifier, fixer, max_rounds=2)
    assert result['rounds'][0]['action'] == 'fixer_failed'
    assert result['candidate_records'][0]['status'] == 'model_failed'
    assert result['selected_source_hash'] == code_hash(ORIGINAL)
    assert result['code'] == ORIGINAL and verifier.calls == [ORIGINAL]


def test_ask_followup_retains_both_calls_and_propagates_input_metadata(tmp_path):
    model = Replies(['```agent-action\n{"kind":"explain_failure","rationale":"clarify"}\n```', reply(FIXED)])
    instance = conductor(tmp_path)
    instance.fixer = FixerAgent(llm=model)
    xs = {'X': np.zeros((2, 3), dtype=np.float32)}
    first = instance._call_generator(instance.fixer, instance.fixer.run_with_action,
        ORIGINAL, task(), failures(), _test_inputs=xs)
    second = instance._answer_ask_and_retry(first, repair_base=ORIGINAL, task=task(),
        failed_results=failures(), route=SimpleNamespace(fix_not_rewrite=True, error_shape=''),
        evidence='', record={}, test_inputs=xs)
    assert second.code == FIXED and len(model.sent) == 2
    assert [r['status'] for r in instance.candidate_records] == ['no_candidate', 'prechecked']
    assert all(r['context']['input_metadata_status'] == 'known' for r in instance.candidate_records)


def test_cursor_supports_reset_append_outcome_and_detached_results():
    fixer = FixerAgent(llm=Replies([reply(FIXED), reply(ORIGINAL)]))
    first = fixer.run_with_action(ORIGINAL, task(), failures())
    cursor = records_cursor(fixer)
    assert records_since(fixer, cursor) == []
    second = fixer.run_with_action(FIXED, task(), failures())
    assert records_since(fixer, cursor) == list(second.candidate_records)
    assert records_since(fixer, cursor, outcome=first) == list(first.candidate_records)
    producer = SimpleNamespace(candidate_records=list(first.candidate_records))
    cursor = records_cursor(producer)
    producer.candidate_records.extend(second.candidate_records)
    detached = records_since(producer, cursor)
    assert detached == list(second.candidate_records)
    detached[0]['status'] = 'observer change'
    assert producer.candidate_records[-1]['status'] == 'prechecked'
    assert records_since(object(), (None, 0)) == []


def test_pipeline_failure_receipts_reach_coder_wrapper():
    fixer = FixerAgent(llm=Replies([reply(FIXED)]))
    receipt = fixer.run_with_action(ORIGINAL, task(), failures()).candidate_records[0]
    class FailedPipeline:
        candidate_records = []
        last_candidate_record = None
        def run(self, *args, **kwargs):
            self.candidate_records = [receipt]
            raise TimeoutError('fixture')
    coder = CoderAgent(KernelDatabase(jsonl_path='off'), llm=object(), pipeline=FailedPipeline())
    with pytest.raises(TimeoutError):
        coder.run(None, task())
    assert coder.candidate_records == [receipt]


@pytest.mark.parametrize('phase', ['done', 'failed'])
def test_status_exposes_generation_evidence_after_success_or_failure(tmp_path, phase):
    from operator_agent.app import hitl_tools
    evidence = str(tmp_path/'generation-records.json')
    status = dict(phase=phase, pid=1)
    if phase == 'failed':
        status['generation_evidence'] = evidence
    else:
        (tmp_path/'conductor.json').write_text(json.dumps(dict(generation_evidence=evidence)))
    (tmp_path/'run-status.json').write_text(json.dumps(status))
    result = hitl_tools.autonomous_status(tmp_path)
    assert result['generation_evidence'] == evidence and result['phase'] == phase
