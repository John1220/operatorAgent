"""Mock model, static precheck and fake-device integration only. NO GPU claims."""
import contextlib
import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import Conductor, KernelDatabase, llm_client as lc
from operator_agent.artifact_store import ArtifactStore
from operator_agent.budget import BudgetPlan
from operator_agent.kernel_db import FailureDatabase
from operator_agent.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused
from operator_agent.llm_client import LLMClient
from operator_agent.run_contract import code_hash, load_checkpoint, protocol_identity
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level, VerificationResult
from operator_agent.single_candidate_coder import (SingleCandidateCoder, SingleCandidatePolicy,
                                                  audit_generation, input_metadata, precheck)
from tests.unit.test_workload_evidence import task, inputs, backend


CODE = '''import torch
import triton
import triton.language as tl

@triton.jit
def fixture_jit(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y+i, tl.load(X+i))

def kernel(A, B):
    return torch.empty_like(A)
'''
# The source is ONLY a static/mock fixture, not a correct matmul implementation.


def setup(tmp_path, min_speedup=10.):
    t = task(min_speedup=min_speedup)
    t.target.dsl = 'triton'
    xs = [inputs(16), inputs(32)]
    sketch = UnifiedSketch(algorithm='matmul', target_dsl='triton')
    budget = LLMCallBudget(LLMCallLimits(max_calls=1, max_output_tokens=4096,
        total_output_token_reservations=4096), tmp_path / 'calls')
    llm = LLMClient(base_url='https://fixture.invalid/v1', model='fixture', api_key='PRIVATE_KEY',
                    call_budget=budget)
    coder = SingleCandidateCoder(llm, t, sketch, xs)
    return coder, t, sketch, xs


def response(code=CODE, envelope=None):
    model = dict(code=code, rationale='static mock fixture; not real GPU correctness') if envelope is None else envelope
    return json.dumps(dict(model='fixture', choices=[dict(finish_reason='stop', message=dict(
        content=json.dumps(model)))], usage=dict(prompt_tokens=120, completion_tokens=150)))


def generate(coder, t, sketch, xs):
    return coder.run(sketch, t, test_inputs=xs)


def test_one_request_exact_model_source_frozen_context_and_offline_audit(tmp_path, monkeypatch):
    coder, t, sketch, xs = setup(tmp_path)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    assert generate(coder, t, sketch, xs) == CODE
    assert coder.last_finalists == () and send.call_count == 1
    with pytest.raises(LLMCallRefused, match='already attempted'):
        generate(coder, t, sketch, xs)
    body = json.loads(send.call_args.args[0]['body_text'])
    context = json.loads(body['messages'][1]['content'])
    assert context['input_metadata'][0]['A']['shape'] == [16, 16]
    assert 'PRIVATE_KEY' not in json.dumps(coder.declaration())
    assert context['contract']['min_speedup'] == 10.
    root = coder.llm.call_budget.root
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    proof = coder.check_candidate(CODE)
    assert proof['completed'] and proof['model_authored_source']
    assert proof['gpu_measurements'] == 0 and not proof['restored_eligibility']
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert (root / 'candidate.json').stat().st_mode & 0o777 == 0o444
    assert coder.last_generation['candidate_hash'] == code_hash(CODE)
    from operator_agent.generation_context import validate_candidate_record
    record = json.loads((root / 'candidate-record.json').read_text())
    assert validate_candidate_record(record) == coder.last_candidate_record
    assert record['source_hash'] == code_hash(CODE)
    assert record['strategy'] == 'bounded-single'
    assert record['context']['knowledge']['skill'] is None
    assert record['correctness_proven'] is False
    assert (root / 'candidate-record.json').stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize('which', ('task', 'sketch', 'inputs', 'reference', 'disk', 'policy', 'in_memory', 'code_path'))
def test_frozen_context_drift_refused_before_request(tmp_path, monkeypatch, which):
    coder, t, sketch, xs = setup(tmp_path)
    if which == 'task': t.constraints.min_speedup = 1.
    if which == 'sketch': sketch.tiling.block_m += 16
    if which == 'inputs': xs[0] = inputs(64)
    if which == 'reference': t.source.reference_source = 'def reference(A,B): return A'
    if which == 'code_path': t.source.code_path = '/unapproved/reference.py'
    if which == 'policy': coder.policy = SingleCandidatePolicy(max_output_tokens=4000)
    if which == 'in_memory': coder._context['contract']['min_speedup'] = 1.
    if which == 'disk':
        path = coder.llm.call_budget.root / 'generation-context.json'
        path.chmod(0o600); path.write_text('{}')
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('drift dispatched'))
    with pytest.raises(ValueError, match='drift'): generate(coder, t, sketch, xs)
    assert coder.llm.call_budget.summary()['reserved_calls'] == 0


@pytest.mark.parametrize('envelope', [dict(code=CODE, rationale='x', reference='changed'),
    dict(code=CODE, rationale=''), dict(code=CODE, rationale='x' * 1001),
    dict(code='```python\n'+CODE+'```', rationale='x'), dict(code=True, rationale='x'),
    dict(code='', rationale='x'), dict(code='x' * 32769, rationale='x')])
def test_bad_envelope_consumes_only_one_call_no_template_or_repair(tmp_path, monkeypatch, envelope):
    coder, t, sketch, xs = setup(tmp_path)
    send = Mock(return_value=response(envelope=envelope))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    with pytest.raises(LLMCallRefused, match='no repair/retry/template'): generate(coder, t, sketch, xs)
    with pytest.raises(LLMCallRefused): generate(coder, t, sketch, xs)
    assert send.call_count == 1 and coder.last_generation['status'] == 'failed'
    assert coder.llm.call_budget.summary()['aborted'] == 'candidate_generation_failed'
    assert (coder.llm.call_budget.root / 'candidate.json').exists()


@pytest.mark.parametrize('code', [CODE.replace('def kernel(A, B):', 'def kernel(X):'),
    CODE.replace('@triton.jit', ''), CODE.replace('return torch.empty_like(A)', 'return torch.matmul(A, B)'),
    CODE + '\nimport os\n', CODE + '\nfrom subprocess import run\n',
    CODE + '\nx=eval("1")\n', CODE + '\nx=torch.__dict__\n',
    CODE.replace('import torch', 'import torch as t'), CODE + '\ndef kernel(A,B): return A\n',
    CODE + '\ninvalid python : :\n', CODE.replace('@triton.jit', '@triton.autotune(configs=[])\n@triton.jit'),
    CODE + '\ntriton.compile("arbitrary")\n'])
def test_invalid_model_code_is_retained_but_never_executed_or_repaired(tmp_path, monkeypatch, code):
    coder, t, sketch, xs = setup(tmp_path)
    send = Mock(return_value=response(code))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    with pytest.raises(LLMCallRefused): generate(coder, t, sketch, xs)
    assert coder.last_generation['code'] == code
    assert coder.last_generation['candidate_hash'] == code_hash(code)
    assert coder.last_generation['status'] == 'failed' and send.call_count == 1


def test_disclosed_context_has_no_tensor_values_or_scalar_values():
    x = np.full((4, 4), 12345.678, dtype=np.float32)
    metadata = input_metadata({'X': x, 'epsilon': 987654.321})
    text = json.dumps(metadata)
    assert '12345' not in text and '987654' not in text
    assert metadata[0]['epsilon'] == {'scalar_type': 'float', 'value_disclosed': False}
    with pytest.raises(ValueError): input_metadata({'secret': object()})


def test_trusted_layernorm_template_is_inside_narrow_static_mode():
    # Static test only; use its source as an example, not a generated credential.
    from benchmarks.workload_llm_search_h20 import source
    assert precheck(source(8), [{'X': dict(shape=[8192, 15360], dtype='float16')}]) is None


@pytest.mark.parametrize('which', ('response', 'candidate', 'request', 'context', 'model', 'count', 'reserved', 'declared_repair'))
def test_readonly_audit_rejects_tampered_lineage(tmp_path, monkeypatch, which):
    coder, t, sketch, xs = setup(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: response())
    generate(coder, t, sketch, xs)
    decl, budget, lineage = coder.declaration(), coder.llm.call_budget.summary(), copy.deepcopy(coder.last_generation)
    if which == 'response': budget['attempts'][0]['response_text'] += ' '
    if which == 'candidate': lineage['candidate_hash'] = 'f' * 64
    if which == 'request': budget['attempts'][0]['request']['messages'] = []
    if which == 'context': decl['context']['contract']['min_speedup'] = 1.
    if which == 'model': budget['identity']['model'] = 'other'
    if which == 'count': budget['reserved_calls'] = 2
    if which == 'reserved': budget['output_tokens_reserved'] = 0
    if which == 'declared_repair': decl['repairs'] = 1
    with pytest.raises((ValueError, KeyError)): audit_generation(decl, budget, lineage, CODE)


def conductor(tmp_path, coder, b, **overrides):
    options = dict(backend=b, llm=coder.llm,
        kernel_db=KernelDatabase(str(tmp_path / 'corpus.jsonl'), persist_retrievals=False),
        failure_db=FailureDatabase(str(tmp_path / 'failure.jsonl')),
        use_coder_pipeline=False, max_optimize_rounds=1,
        heartbeat_interval=1000, budget=BudgetPlan(oracle_total=3), single_candidate_coder=coder)
    options.update(overrides)
    c = Conductor(**options)
    c.store.output_dir = tmp_path / 'artifacts'
    return c


@pytest.mark.parametrize('phase', ('valid_below_target', 'degraded', 'l1_failed', 'l2_failed'))
def test_production_single_conductor_persists_own_source_without_followup(tmp_path, monkeypatch, phase):
    coder, t, sketch, xs = setup(tmp_path)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    def change(p, n, count):
        if phase == 'degraded' and n == 32: p['gpu_util_pct'] = 80
    b, cases = backend(change)
    c = conductor(tmp_path, coder, b)
    forbidden = lambda *a, **k: pytest.fail('unexpected Designer/Search/Fixer/finalist followup')
    c.designer.run = forbidden; c.search.run = forbidden; c.fixer.run_with_action = forbidden
    c._select_finalist = forbidden
    c.planner.plan = forbidden
    # Fake compile/function only. Production L3 qualification + full workload,
    # source/contract binding, selection, save/load and seals remain live.
    c.verifier._verify_compile = lambda *a, **k: VerificationResult(Level.L1_COMPILE,
        phase != 'l1_failed', 'fake compile only', metrics={'static_guards': True})
    c.verifier._verify_function = lambda *a, **k: VerificationResult(Level.L2_FUNCTION,
        phase != 'l2_failed', 'fake correctness only', metrics={'max_error': 0.})
    with contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(t, xs, target_gflops=0.)
    assert send.call_count == 1 and art.code == CODE and not art.all_passed
    assert len(art.iteration_history) == 1 and len(cases) == (2 if phase in ('valid_below_target', 'degraded') else 0)
    assert not (tmp_path / 'corpus.jsonl').exists()
    reason = 'rounds_exhausted' if phase in ('valid_below_target', 'degraded') else 'l1_l2_failed'
    assert art.metadata['stop']['reason'] == reason
    loaded = ArtifactStore(str(tmp_path)).load(art.saved_dir)
    assert loaded.code == CODE and loaded.sketch.to_dict() == art.sketch.to_dict()
    assert loaded.metadata['llm_cost'] == art.metadata['llm_cost']
    assert loaded.metadata['llm_call_budget'] == json.loads(json.dumps(art.metadata['llm_call_budget']))
    generation = loaded.metadata['code_generation']
    assert audit_generation(generation['declaration'], loaded.metadata['llm_call_budget'], generation['lineage'], loaded.code)['completed']
    folder = Path(art.saved_dir)
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    _, binding = load_checkpoint(folder / 'execution_snapshot.json', t, xs, c._run_controls)
    assert binding['manifest']['controls']['code_generation'] == generation['declaration']
    assert before == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    if phase == 'valid_below_target':
        assert loaded.benchmark_data['measurement_validity'] == 'valid'
        assert loaded.metadata['candidate_selection']['selected_round'] == 1
    elif phase == 'degraded':
        assert loaded.benchmark_data['measurement_validity'] == 'degraded'
        assert loaded.metadata['candidate_selection']['selected_round'] is None


def test_conductor_never_auto_provisions_in_single_mode(tmp_path, monkeypatch):
    coder, t, sketch, xs = setup(tmp_path)
    from tests.backend_fakes import DeclaredFakeBackend
    forbidden = lambda *a, **k: pytest.fail('provision/install reached')
    b = DeclaredFakeBackend(ensure_environment=forbidden, analyze_dependencies=forbidden,
        compile=lambda *a: False, get_hardware_info=lambda: dict(name='NVIDIA H20'))
    c = conductor(tmp_path, coder, b)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: response())
    with contextlib.redirect_stdout(io.StringIO()): art = c.execute(t, xs)
    assert not art.all_passed


def test_generation_failure_leaves_original_attempt_and_never_dispatches_verifier(tmp_path, monkeypatch):
    coder, t, sketch, xs = setup(tmp_path)
    b, _ = backend()
    c = conductor(tmp_path, coder, b)
    send = Mock(return_value=response('import os\n'))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    c.verifier.verify = lambda *a, **k: pytest.fail('invalid generated candidate dispatched')
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(LLMCallRefused):
        c.execute(t, xs)
    assert send.call_count == 1 and coder.last_generation['status'] == 'failed'
    assert coder.last_generation['code'] == 'import os\n'
    assert coder.llm.call_budget.summary()['reserved_calls'] == 1
    assert not list(c.store.output_dir.rglob('result.json'))


def test_passing_single_candidate_still_requires_independent_confirmation(tmp_path, monkeypatch):
    # Freeze a separate trusted fixture target before calling the model. This
    # is not a relaxation of any live/previous acceptance campaign.
    coder, t, sketch, xs = setup(tmp_path, min_speedup=1.05)
    b, cases = backend()
    c = conductor(tmp_path, coder, b, budget=BudgetPlan(oracle_total=20))
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    c.verifier._verify_compile = lambda *a, **k: VerificationResult(Level.L1_COMPILE, True,
        'fake compile only', metrics={'static_guards': True})
    c.verifier._verify_function = lambda *a, **k: VerificationResult(Level.L2_FUNCTION, True,
        'fake correctness only', metrics={'max_error': 0.})
    with contextlib.redirect_stdout(io.StringIO()): art = c.execute(t, xs, target_gflops=0.)
    assert art.all_passed and cases == [16, 32, 16, 32] and send.call_count == 1
    assert art.remeasure is not None and art.metadata['budget']['spent']['confirm'] == 2
    assert art.metadata['corpus_admission']['kept'] is True


def test_same_client_cannot_refreeze_a_second_generation_context(tmp_path):
    coder, t, sketch, xs = setup(tmp_path)
    before = (coder.llm.call_budget.root / 'generation-context.json').read_bytes()
    t.constraints.min_speedup = 1.
    with pytest.raises(ValueError, match='already frozen'):
        SingleCandidateCoder(coder.llm, t, sketch, xs)
    assert (coder.llm.call_budget.root / 'generation-context.json').read_bytes() == before


@pytest.mark.parametrize('override', ({'use_coder_pipeline': True}, {'max_optimize_rounds': 2}, {'llm': None}))
def test_ambiguous_or_unbounded_conductor_config_rejected(tmp_path, override):
    coder, _, _, _ = setup(tmp_path)
    b, _ = backend()
    with pytest.raises(ValueError, match='single candidate requires'): conductor(tmp_path, coder, b, **override)


def test_resume_and_execution_drift_refused_before_hardware_or_model(tmp_path, monkeypatch):
    coder, t, sketch, xs = setup(tmp_path)
    b, _ = backend()
    c = conductor(tmp_path, coder, b)
    monkeypatch.setattr(b, 'get_hardware_info', lambda: pytest.fail('hardware reached'))
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('model reached'))
    with pytest.raises(ValueError, match='cannot resume'): c.execute(t, xs, resume_from='old')
    c.max_optimize_rounds = 2
    with pytest.raises(ValueError, match='policy drift'): c.execute(t, xs)
    c.max_optimize_rounds = 1; t.constraints.min_speedup = 1.
    with pytest.raises(ValueError, match='drift'): c.execute(t, xs)


def test_generation_sources_are_in_production_protocol_identity():
    files = protocol_identity()
    assert all('operator_agent/'+name in files for name in
        ('llm_client.py', 'llm_budget.py', 'llm_transport.py', 'single_candidate_coder.py'))
