"""Offline fixtures exercise candidates and retain the normal evidence gates."""
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent.app import hitl_tools as ht, cli, mcp_server
from operator_agent.app.conductor import Conductor
from operator_agent.core.artifact_store import ArtifactStore
from operator_agent.core.backend_caps import Cap
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.environment import compare_fingerprints, fingerprint_of
from operator_agent.decide.measurement import Validity, assess
from operator_agent.decide.verifier import Verifier
from operator_agent.execute.fake_backend import FakeBackend, FakeConfig, FakeLLM
from operator_agent.execute.llm_budget import ProductionBudgetExhausted, ProductionLLMCallBudget
from operator_agent.execute.task_budget import TaskBudgetConfig


CODE = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
FIXED = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0).copy()\n'
DECL = dict(request='relu', shapes=[[4, 8]], dsl='python', min_speedup=0.)


def test_full_pipeline_uses_no_device_probe_or_model_transport(tmp_path, monkeypatch):
    from operator_agent.execute import llm_client, llm_transport, mock_gpu
    forbidden = Mock(side_effect=AssertionError('offline run attempted GPU/network access'))
    monkeypatch.setattr(mock_gpu, '_has_cuda', forbidden)
    monkeypatch.setattr(llm_client, '_resolve_api_key', forbidden)
    monkeypatch.setattr(llm_transport, 'dispatch', forbidden)
    monkeypatch.setattr(llm_client, 'bounded_dispatch', forbidden)
    monkeypatch.setattr(Verifier, '_measure_baseline', forbidden)
    monkeypatch.setenv('LLM_API_KEY_FILE', '/no/offline/credential/file')
    budget = ProductionLLMCallBudget(TaskBudgetConfig(max_llm_calls=4).llm_limits(), tmp_path/'journal')
    llm = FakeLLM('relu', call_budget=budget)
    prompts = {}
    respond = llm._respond
    def record_prompt(messages, **kwargs):
        prompts[kwargs['caller_tag']] = messages[0]['content']
        return respond(messages, **kwargs)
    monkeypatch.setattr(llm, '_respond', record_prompt)
    task = ht.build_task(DECL)
    inputs = freeze_from_task(task).cases[0].materialize()
    backend = FakeBackend(event_path=tmp_path/'events.jsonl')
    conductor = Conductor(backend=backend, llm=llm, run_root=str(tmp_path),
                          use_coder_pipeline=False, max_optimize_rounds=1)
    artifact = conductor.execute(task, inputs)
    forbidden.assert_not_called()
    assert {v.level.value: v.passed for v in artifact.verification_results} == {
        'L1': True, 'L2': True, 'L3': False}
    assert all(v.metrics['test_only'] for v in artifact.verification_results)
    assert artifact.benchmark_data['speedup_vs_baseline'] == 2.
    assert artifact.benchmark_data['measurement_l3_pass_refused'] is True
    assert not any(artifact.benchmark_data['eligibility'].values())
    assert conductor.kernel_db.corpus_path == str(tmp_path/'fake-kernel_db.jsonl')
    assert conductor.failure_db._jsonl == tmp_path/'fake-failure_db.jsonl'
    saved = Path(artifact.saved_dir)
    data = json.loads((saved/'result.json').read_text())
    assert data['execution_context']['test_only'] is True
    assert data['execution_context']['gpu_acceptance'] is False
    assert 'FAKE / test_only' in (saved/'report.md').read_text()
    assert ArtifactStore(output_dir=str(tmp_path/'output')).load(saved).metadata[
        'execution_context']['test_only'] is True
    assert llm.cost_tracker.summary()['total_cost_usd'] == 0
    journal = budget.summary()
    assert journal['reserved_calls'] == 2
    assert all(call['dispatched_monotonic'] is None and call['api_usage'] is None
               and call['usage_source'] == 'test_fixture' for call in journal['attempts'])
    assert list((tmp_path/'log').glob('*.log'))
    records = data['candidate_records']
    assert records and all(row['source_author'] == 'test_fixture' for row in records)
    assert {'Designer', 'Coder'} <= prompts.keys()
    assert all('Import only numpy/math.' in prompt for prompt in prompts.values())


def test_repair_prompt_includes_fake_execution_restrictions(tmp_path, monkeypatch):
    from operator_agent.execute import llm_client
    prompts = []
    def client(**options):
        llm = FakeLLM('relu', call_budget=options['call_budget'], timeout=options['timeout'])
        respond = llm._respond
        def record_prompt(messages, **kwargs):
            prompts.append(messages[0]['content'])
            return respond(messages, **kwargs)
        llm._respond = record_prompt
        return llm
    monkeypatch.setenv('LLM_BASE_URL', 'http://unused.invalid/v1')
    monkeypatch.setenv('LLM_API_KEY', 'unused-fixture-key')
    monkeypatch.setattr(llm_client, 'LLMClient', client)
    ht.prepare(DECL, tmp_path)
    wrong = 'import numpy as np\ndef kernel(X):\n    return X.copy()\n'
    result = ht.repair(tmp_path, wrong, backend='fake', stop_after='L2', max_rounds=2,
                       max_llm_calls=3, wall_clock_s=30)
    assert result['initial_full_l2_passed'] is False
    assert result['full_l2_passed'] is True
    assert len(prompts) == 1 and 'Import only numpy/math.' in prompts[0]


def test_backend_executes_wrong_candidate_instead_of_substituting_reference():
    inputs = {'X': np.array([-2., 3.], dtype=np.float32)}
    wrong = 'import numpy as np\ndef kernel(X):\n    return np.minimum(X, 0)\n'
    backend = FakeBackend()
    assert backend.compile(wrong)
    output = backend.execute(wrong, inputs)
    np.testing.assert_array_equal(output.output, [-2., 0.])
    assert output.metrics['input_fidelity']


@pytest.mark.parametrize('scenario', ['compile-repair', 'correctness-repair'])
def test_injected_fault_persists_for_same_source_and_repairs_with_new_source(scenario):
    backend = FakeBackend(FakeConfig(scenario=scenario))
    inputs = {'X': np.array([-2., 3.], dtype=np.float32)}
    if scenario == 'compile-repair':
        for _ in range(2):
            with pytest.raises(SyntaxError, match='injected'):
                backend.compile(CODE)
        assert backend.compile(FIXED)
    else:
        for _ in range(2):
            np.testing.assert_array_equal(backend.execute(CODE, inputs).output, [1., 4.])
        np.testing.assert_array_equal(backend.execute(FIXED, inputs).output, [0., 3.])


@pytest.mark.parametrize('source', ['import torch\ndef kernel(X):\n return X\n',
                                    'import triton\ndef kernel(X):\n return X\n',
                                    'def kernel(X):\n return __import__("torch")\n'])
def test_accelerator_source_is_refused_without_import(source):
    with pytest.raises(ValueError):
        FakeBackend().compile(source)


@pytest.mark.parametrize('source', [
    pytest.param('from numpy import load\ndef kernel(X):\n return load("candidate.npy")\n',
                 id='io-from-import'),
    pytest.param('from numpy import load as read_array\ndef kernel(X):\n'
                 ' return read_array("candidate.npy")\n', id='io-import-alias'),
    pytest.param('from numpy import ctypeslib as native\ndef kernel(X):\n'
                 ' return native.load_library("candidate", ".")\n', id='native-import-alias'),
    pytest.param('from numpy.ctypeslib import load_library as native_loader\n'
                 'def kernel(X):\n return native_loader("candidate", ".")\n',
                 id='native-submodule-import'),
    pytest.param('import numpy.lib.npyio as npyio\ndef kernel(X):\n'
                 ' return npyio.DataSource().exists("http://127.0.0.1:9/fake")\n',
                 id='datasource-submodule-alias'),
    pytest.param('from numpy.lib.npyio import DataSource as Source\ndef kernel(X):\n'
                 ' return Source().exists("http://127.0.0.1:9/fake")\n',
                 id='datasource-from-submodule-alias'),
    pytest.param('from numpy import DataSource as Source\ndef kernel(X):\n'
                 ' return Source().exists("http://127.0.0.1:9/fake")\n',
                 id='datasource-from-import-alias'),
    pytest.param('import numpy as np\ndef kernel(X):\n'
                 ' return np.lib.npyio.DataSource().exists("http://127.0.0.1:9/fake")\n',
                 id='datasource-attribute-url'),
    pytest.param('import numpy as np\ndef kernel(X):\n'
                 ' np.savez("candidate.npz", X=X)\n return X\n', id='io-attribute'),
])
@pytest.mark.parametrize('operation', ['compile', 'execute'])
def test_numpy_io_is_refused_before_candidate_dispatch(source, operation, monkeypatch):
    from operator_agent.execute.fake_backend import backend as fake_backend
    forbidden = Mock(side_effect=AssertionError('IO source reached candidate execution or network'))
    monkeypatch.setattr(fake_backend, 'run_compile', forbidden)
    monkeypatch.setattr(fake_backend, 'run_candidate', forbidden)
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    backend = FakeBackend()
    with pytest.raises(ValueError):
        if operation == 'compile':
            backend.compile(source)
        else:
            backend.execute(source, {'X': np.array([1.], dtype=np.float32)})
    forbidden.assert_not_called()


def test_hardware_profiles_are_synthetic_distinct_and_missing_stays_missing():
    small = FakeBackend(FakeConfig(profile='small'))
    large = FakeBackend(FakeConfig(profile='large'))
    unknown = FakeBackend(FakeConfig(profile='unknown')).get_hardware_info()
    assert not unknown['hardware_report']
    a, b = small.get_hardware_info(), large.get_hardware_info()
    assert a['hardware_report']['shared_memory_per_sm_kb'] < b['hardware_report']['shared_memory_per_sm_kb']
    assert compare_fingerprints(fingerprint_of(a, small.label), fingerprint_of(b, large.label)).drifted
    assert not a['cuda_available'] and 'uuid' not in a['hardware_report']
    assert Cap.REAL_DEVICE not in small.capabilities
    with pytest.raises(RuntimeError, match='hardware probe failure'):
        FakeBackend(FakeConfig(scenario='hardware-missing')).get_hardware_info()


def test_fixed_performance_is_not_progress_or_gpu_acceptance_even_with_real_device_claim():
    backend = FakeBackend(FakeConfig(scenario='no-improvement'))
    first = backend.benchmark(CODE, {})
    assert first == backend.benchmark(FIXED, {})
    assert first['baselines']['fake_fixture']['median_time_ms'] == first['median_time_ms']
    assessment = assess(median_time_ms=1., launch_floor_us=0.,
                        backend_caps={Cap.TEST_ONLY, Cap.REAL_DEVICE}, correctness_verified=True)
    assert assessment.validity is Validity.UNINFORMATIVE
    assert assessment.l3_pass_is_refused and assessment.l3_pass_refusal_causes
    assert not assessment.can_gate and not assessment.can_count_stagnation


def test_fake_model_uses_finite_budget_without_api_usage(tmp_path):
    budget = ProductionLLMCallBudget(TaskBudgetConfig(max_llm_calls=1).llm_limits(), tmp_path/'journal')
    llm = FakeLLM('relu', call_budget=budget)
    assert llm.chat_json([], caller_tag='Designer')['algorithm'] == 'elementwise'
    with pytest.raises(ProductionBudgetExhausted) as stopped:
        llm.chat([], caller_tag='Coder')
    assert stopped.value.reason == 'llm_call_budget'
    assert budget.summary()['reserved_calls'] == 1


@pytest.mark.parametrize('options,error', [
    (dict(backend='mock', fake={'scenario': 'success'}), 'invalid_fake_config'),
    (dict(backend='fake', fake={'scenario': 'typo'}), 'invalid_fake_config'),
    (dict(backend='fake', host='unused'), 'invalid_fake_config'),
])
def test_invalid_fake_options_fail_before_creating_run(tmp_path, options, error):
    with pytest.raises(ht.ToolError) as exc:
        ht.run_autonomous(DECL, tmp_path/'run', **options)
    assert exc.value.code == error
    assert not (tmp_path/'run').exists()


def test_fake_cannot_fulfil_real_gpu_declaration(tmp_path):
    for declaration, error in ((dict(DECL, dsl='triton'), 'fake_python_only'),
                               (dict(DECL, required_caps=['real_device']), 'fake_capability_missing')):
        with pytest.raises(ht.ToolError) as exc:
            ht.run_autonomous(declaration, tmp_path/'run', backend='fake')
        assert exc.value.code == error


def test_cli_and_mcp_forward_fake_configuration(monkeypatch, tmp_path, capsys):
    launch = Mock(return_value={'ok': True})
    monkeypatch.setattr(cli, 'run_autonomous', launch)
    assert cli.main(['run', '--request', 'relu', '--shapes', '4,8', '--run-dir', str(tmp_path),
                     '--backend', 'fake', '--fake-scenario', 'compile-repair',
                     '--fake-profile', 'large', '--llm-mode', 'fake']) == 0
    assert launch.call_args.kwargs['fake'] == dict(scenario='compile-repair', profile='large', llm='fake')
    monkeypatch.setattr(mcp_server, 'run_autonomous', launch)
    mcp_server._call_tool('operator_run', dict(DECL, run_dir=str(tmp_path), backend='fake',
                                              fake={'llm': 'live'}))
    assert launch.call_args.kwargs['fake'] == {'llm': 'live'}
    assert 'fake' not in launch.call_args.args[0]


def test_live_model_is_explicit_and_still_requires_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv('LLM_BASE_URL', raising=False)
    with pytest.raises(ht.ToolError) as exc:
        ht.run_autonomous(DECL, tmp_path/'run', backend='fake', fake={'llm': 'live'})
    assert exc.value.code == 'llm_not_configured'
    assert not (tmp_path/'run').exists()


def test_live_model_mode_selects_the_existing_client(tmp_path, monkeypatch):
    from operator_agent.app import autonomous
    from operator_agent.execute import llm_client
    from operator_agent.execute.task_clock import monotonic
    config = dict(backend='fake', fake={'llm': 'live'}, max_rounds=1,
                  task_budget=TaskBudgetConfig(wall_clock_s=20).as_dict())
    (tmp_path/'autonomous.json').write_text(json.dumps(config))
    (tmp_path/'declaration.json').write_text(json.dumps(DECL))
    # A local fixture stands in for the existing live client's responses; this
    # tests adapter selection without spending API calls or changing transport.
    live = Mock(side_effect=lambda **kw: FakeLLM('relu', call_budget=kw['call_budget'],
                                                timeout=kw['timeout']))
    monkeypatch.setattr(llm_client, 'LLMClient', live)
    for name in ('OPERATOR_AGENT_OUTPUT_DIR', 'OPERATOR_AGENT_LOG_DIR',
                 'OPERATOR_AGENT_KERNEL_DB', 'OPERATOR_AGENT_FAILURE_DB'):
        monkeypatch.setenv(name, str(tmp_path/name))  # restore worker env changes on teardown
    started = monotonic()
    assert autonomous._worker_main(tmp_path, started+20, started) == 0
    live.assert_called_once()
    summary = json.loads((tmp_path/'conductor.json').read_text())
    assert summary['execution_context']['model_mode'] == 'live'
    assert summary['all_passed'] is False
