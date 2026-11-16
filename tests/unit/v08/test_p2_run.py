from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmarks.experiments.v08 import p2_plan, p2_run
from operator_agent.app.hitl_tools import build_task
from operator_agent.core.run_contract import code_hash
from operator_agent.decide.noise_preflight import PROTOCOL


CODE = 'import triton\nimport triton.language as tl\n@triton.jit\ndef f(X):\n    pass\ndef kernel(X):\n    return X\n'
SKETCH = dict(algorithm='layernorm', target_dsl='triton',
              tiling=dict(block_m=16, block_n=512, block_k=32))


@pytest.mark.parametrize('stop', [None, 'preflight-start.json', 'preflight-final.json'])
def test_p2_pilot_routes_model_and_independent_evaluation(tmp_path, monkeypatch, stop):
    monkeypatch.setattr(p2_plan, 'SEED_SOURCE_HASH', code_hash(CODE))
    trace = []

    def gate(path):
        trace.append(path.name)
        return dict(schema=PROTOCOL, action='hold' if path.name == stop else 'start', ok=True,
                    channel=p2_plan.PROBE_CHANNEL, identity=dict(backend='ssh', shape=list(p2_plan.SHAPE)),
                    planned_effect_pct=2.5, effective_band_pct=0.1)

    def evaluate(arm, code):
        trace.append(f'{arm}:final')
        correctness = dict(verdict='passed', cases=[dict(case_id=case_id, verdict='passed')
                                                    for case_id in p2_plan.required_case_ids()])
        metrics = dict(baseline_type='pytorch_eager', measurement_validity='valid',
                       median_time_ms={'seed': 1.1, 'off': 1.0, 'on': 0.95}[arm])
        return dict(ok=True, evidence=arm, evidence_complete=True, validity='valid',
                    stop_after='L3', verification_purpose='final', levels=dict(L1=True, L2=True),
                    contract_hash=p2_plan.CONTRACT_HASH, source_hash=code_hash(code),
                    verdicts=[dict(level='L2', passed=True, metrics=dict(correctness=correctness)),
                              dict(level='L3', metrics=metrics)])

    class Model:
        def __init__(self, arm):
            self.arm = arm

        def chat_json(self, _messages, **_kwargs):
            trace.append(f'{self.arm}:search')
            return dict(suggestion='Use the shared reference implementation')

        def chat(self, messages, **_kwargs):
            trace.append(f'{self.arm}:coder')
            assert CODE.rstrip() in messages[0]['content']
            return CODE + f'# synthetic candidate {self.arm}\n'

    @contextmanager
    def clients(arm_dir):
        yield Model(arm_dir.name), lambda: dict(test_only=True)

    root = tmp_path / 'pilot'
    result = p2_run.run_pilot(root=root, seed_code=CODE, sketch=SKETCH,
        task=build_task(p2_plan.DECLARATION), preflight=gate, evaluate=evaluate, clients=clients)
    assert result['formal_gain_claim'] is False
    expected = ['preflight-start.json']
    if stop != expected[0]:
        expected += ['seed:final', 'off:search', 'off:coder', 'on:search', 'on:coder',
                     'preflight-final.json']
    if stop is None:
        expected += ['off:final', 'on:final']
        assert result['verdict'] == 'optimize_helps'
    else:
        assert result['verdict'] == 'held'
    assert trace == expected
    assert p2_run.read_json(root / 'pilot.json') == result


@pytest.mark.parametrize('phase', ['final_complete', 'final_failed'])
def test_live_evaluation_waits_for_l2_and_honors_host_final_state(tmp_path, monkeypatch, phase):
    from operator_agent.app import hitl_tools

    calls = []

    def prepare(_declaration, run, *, host_budget):
        run.mkdir(parents=True)
        assert host_budget['max_verifications'] == 2

    def verify(run, code, *, verification_purpose, stop_after, **_kwargs):
        calls.append((verification_purpose, stop_after, code))
        if stop_after == 'L2':
            return dict(levels=dict(L2=True))
        path = run / 'result.json'
        path.write_text('{"validity":"valid"}')
        return dict(evidence=str(path), verification_purpose='final', host_task=dict(phase=phase))

    monkeypatch.setattr(hitl_tools, 'prepare', prepare)
    monkeypatch.setattr(hitl_tools, 'verify', verify)
    evaluate = p2_run.live_evaluation(tmp_path, SimpleNamespace(verify_timeout=10, host='test',
                                                              work_dir='/tmp/test'))
    result = evaluate('on', CODE)
    assert calls == [('explore', 'L2', CODE), ('final', 'L3', CODE)]
    if phase == 'final_complete':
        assert result == dict(validity='valid', verification_purpose='final')
    else:
        assert result == {}
