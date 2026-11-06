"""The public MCP repair route consumes shared guidance and online diagnosis."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent.app import hitl_tools as ht, mcp_server
from operator_agent.generate.generation_spec import GENERATION_SPEC, generation_spec_identity
from operator_agent.execute.llm_client import LLMClient
from operator_agent.core.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.core.tolerance import effective_atol


@pytest.mark.parametrize('operator', ('layernorm', 'softmax', 'relu'))
def test_prepare_uses_the_trusted_single_input_profile_not_legacy_affine_names(tmp_path, operator):
    declaration = dict(request=operator+' fp32', precision='fp32', shapes=[[7,33]])
    prepared = ht.prepare(declaration, tmp_path/operator)
    assert set(prepared['inputs']) == {'X'}
    assert prepared['generation_spec'] == generation_spec_identity()
    task = ht.build_task(declaration)
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.decide.verifier import Verifier
    inputs = freeze_from_task(task).cases[0].materialize()
    assert Verifier.correctness_preflight(task, inputs) is None
    assert inputs['X'].shape == (7,33)


def test_mcp_repair_receives_diagnosis_and_returns_compact_evidence(tmp_path, monkeypatch):
    run = tmp_path/'prepared'
    declaration = dict(request='layernorm fp32 triton', precision='fp32', dsl='triton', shapes=[[7,33]])
    ht.prepare(declaration, run)
    task = ht.build_task(declaration)
    atol, rtol, rho = effective_atol('fp32', task.constraints.max_error)
    bad = VerificationResult(Level.L2_FUNCTION, False, 'numerical mismatch', metrics={
        'correctness': {'verdict':'failed', 'cases':[]}, 'correctness_primary_verdict':'failed',
        'input_fidelity_expected':{'X':{'shape':[7,33]}}, 'input_fidelity_verified':True,
        'max_error':.04, 'max_rel_error':.03, 'bad_elements':231,
        'compared_elements':231, 'matched_ratio':0., 'coverage':1., 'reference_nonfinite':0,
        'atol':atol, 'rtol':rtol, 'required_matched_ratio':rho})
    outcomes = [[VerificationResult(Level.L1_COMPILE, True), bad],
                [VerificationResult(Level.L1_COMPILE, True), VerificationResult(
                    Level.L2_FUNCTION, True, metrics={'correctness':{'verdict':'passed'}})]]
    verification = Mock(side_effect=outcomes)
    monkeypatch.setattr(ht, 'Verifier', lambda _: SimpleNamespace(verify=verification))
    monkeypatch.setattr(ht, '_make_backend', lambda *a, **k: (
        SimpleNamespace(generation_instructions=''), {'requested':'mock'}))
    source = (Path(__file__).resolve().parents[1]/'fixtures/single_candidate_unmasked_variance.txt').read_text()
    source = source.replace('.to(tl.float16)', '.to(tl.float32)')
    corrected = source.replace('x_centered = x - mean',
                               'x_centered = tl.where(mask, x - mean, 0.0)')
    assert corrected != source
    seen = []
    def reply(client, messages, **kwargs):
        seen.append(messages[0]['content'])
        return '```python\n'+corrected+'\n```'
    monkeypatch.setattr(LLMClient, 'chat', reply)
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    monkeypatch.setenv('LLM_API_KEY', 'synthetic-key')
    response = mcp_server._call_tool('operator_repair', {'run_dir':str(run), 'code':source,
                                                       'stop_after':'L2', 'max_rounds':2})
    payload = json.loads(response['content'][0]['text'])
    assert payload['passed'] and payload['full_l2_passed'] is True
    assert verification.call_count == 2
    assert payload['rounds'][0]['diagnosis']['state'] == 'supported_hypothesis'
    assert GENERATION_SPEC in seen[0] and 'padded_centered_variance' in seen[0]
    compact = payload['rounds'][0]['diagnosis']['findings'][0]
    assert 'source_sites' not in compact and compact['repair_obligations']
    saved = json.loads(Path(payload['evidence']).read_text())
    full = saved['rounds'][0]['diagnosis']['findings'][0]
    assert full['source_sites'] and full['witness']
    receipt, = saved['candidate_records']
    from operator_agent.generate.generation_context import validate_candidate_record
    from operator_agent.core.run_contract import code_hash
    assert validate_candidate_record(receipt)
    assert receipt['source'] == corrected.strip()
    assert receipt['parent_source_hash'] == code_hash(source)
    assert receipt['context']['input_metadata_status'] == 'known'
    compact_receipt, = payload['candidate_records']
    assert 'source' not in compact_receipt and 'context' not in compact_receipt
    assert compact_receipt['record_hash'] == receipt['record_hash']
    assert payload['rounds'][0]['candidate_record_hashes'] == [receipt['record_hash']]
    assert payload['selected_source_hash'] == code_hash(corrected.strip())
    assert len(json.dumps(payload).encode()) < 12000


def test_real_cpu_layernorm_public_verify_preserves_full_suite(tmp_path):
    run = tmp_path/'cpu'
    ht.prepare(dict(request='layernorm fp32', precision='fp32', shapes=[[7,33]]), run)
    source = ('import numpy as np\ndef kernel(X):\n'
              '    z = X - X.mean(axis=-1, keepdims=True)\n'
              '    return z / np.sqrt((z*z).mean(axis=-1, keepdims=True)+1e-5)\n')
    result = ht.verify(run, source, stop_after='L2')
    assert result['levels'] == {'L1':True, 'L2':True}
    assert result['verdicts'][-1]['metrics']['correctness']['verdict'] == 'passed'
