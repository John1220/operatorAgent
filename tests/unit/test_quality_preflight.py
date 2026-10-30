"""The readiness gate never installs dependencies or grants candidate credit."""
import ast
import builtins
import contextlib
import io
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_agent import quality_preflight as p
from operator_agent.backend_caps import Cap
from operator_agent.remote_executor import LocalExecutor, RemoteExecutor, SSHExecutor
from operator_agent.run_contract import code_hash, digest


def _assignments(source):
    return {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ('RUNNER_HASHES_IDENTITY', 'GPU_INDEX', 'CONTROL_HASH')}


def result_for(source):
    constants = _assignments(source)
    return dict(schema=p.PROBE_SCHEMA, status='passed', phase='control', runner_verified=True,
        runner_hashes_identity=constants['RUNNER_HASHES_IDENTITY'],
        control_sha256=constants['CONTROL_HASH'],
        identity=dict(device='cuda', index=constants['GPU_INDEX'], uuid='GPU-synthetic-001',
            name='Synthetic CUDA', compute_capability='9.0', torch_version='2.8.0',
            triton_version='3.4.0', numpy_version='2.0.2', cuda_version='12.8'),
        control=dict(passed=True, device='cuda', dtype='torch.float32', finite=True,
                     compared_elements=257, max_abs_error=0.))


def ssh(tmp_path, mutate=None):
    # Exercise the production adapter type without constructing a connection.
    backend = object.__new__(SSHExecutor)
    backend.platform = 'nvidia'
    backend._persistent = False
    backend.timeout = 300
    backend.work_dir = '~/synthetic-project'
    backend._gpu_pin = 2
    backend._deployed = False
    backend._verified_python = None
    backend._pinned_gpu = Mock(return_value=2)
    path = tmp_path/'runner.py'
    path.write_text('declared runner bytes\n')
    backend._remote_files = Mock(return_value=[(path, '~/synthetic-project/runner.py')])
    def reply(command, stdin_data):
        result = result_for(stdin_data)
        if mutate:
            mutate(result)
        return json.dumps(result)
    backend._ssh_exec = Mock(side_effect=reply)
    backend.deploy = Mock(side_effect=AssertionError('must not deploy'))
    backend.ensure_environment = Mock(side_effect=AssertionError('must not provision'))
    backend.get_hardware_info = Mock(side_effect=AssertionError('may implicitly provision'))
    backend.compile = Mock(side_effect=AssertionError('may implicitly provision'))
    backend.verify_function = Mock(side_effect=AssertionError('may implicitly provision'))
    return backend


def test_preflight_identity_is_pure_detached_and_versions_exact_sources():
    value = p.preflight_identity()
    assert value['control_sha256'] == code_hash(p.CONTROL_SOURCE)
    assert value['probe_sha256'] == code_hash(p.PROBE_SOURCE)
    value['version'] = 'changed'
    assert p.preflight_identity()['version'] == p.PREFLIGHT_VERSION
    compile(p.CONTROL_SOURCE, '<control>', 'exec')
    compile(p.PROBE_SOURCE, '<probe>', 'exec')


def test_pass_pins_same_backend_and_checks_existing_runtime_without_installing(tmp_path):
    backend = ssh(tmp_path)
    result = p.preflight_backend(backend)
    assert result['status'] == 'passed'
    assert result['identity']['index'] == 2 and result['identity']['uuid'] == 'GPU-synthetic-001'
    assert result['runner_verified'] is True
    assert result['runner_hashes_identity']
    assert backend._deployed is True and backend._verified_python == '.venv/bin/python'
    assert backend.timeout == 300
    assert result['model_calls'] == 0 and result['candidate_correctness_proven'] is False
    assert result['performance_measured'] is False and result['dependencies_installed'] is False
    assert result['runner_deployed'] is False and result['remote_cancellation_proven'] is False
    assert result['hash'] == digest({k:v for k,v in result.items() if k != 'hash'})
    command = backend._ssh_exec.call_args.args[0]
    assert 'uv ' not in command and 'pip ' not in command and 'curl ' not in command
    assert 'test -x .venv/bin/python' in command
    assert 'CUDA_VISIBLE_DEVICES=2' in command
    assert command.startswith('cd "$HOME"/')
    for method in ('deploy', 'ensure_environment', 'get_hardware_info', 'compile', 'verify_function'):
        getattr(backend, method).assert_not_called()


def test_remote_facade_uses_the_same_pinned_inner_backend(tmp_path):
    inner = ssh(tmp_path)
    facade = object.__new__(RemoteExecutor)
    facade._backend = inner
    result = p.preflight_backend(facade)
    assert result['status'] == 'passed'
    assert inner._verified_python == '.venv/bin/python'
    assert inner._pinned_gpu.call_count == 1


@pytest.mark.parametrize('caps', [frozenset(), {Cap.REAL_DEVICE},
    {Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING,Cap.EMULATED},
    {Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING,Cap.SIMULATED_PERF}])
def test_missing_or_conflicting_capabilities_stop_before_any_probe(caps):
    backend = SimpleNamespace(capabilities=caps)
    result = p.preflight_backend(backend)
    assert result['status'] == 'failed'
    assert result['checks'][0]['name'] == 'backend_capabilities'


def test_unknown_provider_is_not_probed_via_methods_that_can_provision():
    backend = SimpleNamespace(capabilities={Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING},
        get_hardware_info=Mock(side_effect=AssertionError('do not call')))
    assert p.preflight_backend(backend)['status'] == 'failed'
    backend.get_hardware_info.assert_not_called()


def test_declared_provider_method_is_used_without_recognizing_its_backend_class(tmp_path):
    receipt = p.preflight_backend(ssh(tmp_path))
    provider = SimpleNamespace(
        capabilities={Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING,
                      Cap.READONLY_QUALITY_PREFLIGHT},
        quality_preflight=Mock(return_value=receipt),
        get_hardware_info=Mock(side_effect=AssertionError('must not provision')))
    assert p.preflight_backend(provider) == receipt
    provider.quality_preflight.assert_called_once_with()
    provider.get_hardware_info.assert_not_called()


@pytest.mark.parametrize('receipt', [None, {}, {'status':'passed'}])
def test_declared_provider_must_return_a_complete_standard_receipt(receipt):
    provider = SimpleNamespace(
        capabilities={Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING,
                      Cap.READONLY_QUALITY_PREFLIGHT},
        quality_preflight=Mock(return_value=receipt))
    result = p.preflight_backend(provider)
    assert result['status'] == 'unknown'
    assert result['checks'][-1]['name'] == 'readonly_quality_preflight'


@pytest.mark.parametrize('field,value', [('identity', None), ('runner_verified', False)])
def test_rehashed_incomplete_readiness_does_not_open_model_dispatch(tmp_path, field, value):
    receipt = p.preflight_backend(ssh(tmp_path))
    receipt[field] = value
    receipt['hash'] = digest({k:v for k,v in receipt.items() if k != 'hash'})
    provider = SimpleNamespace(
        capabilities={Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING,
                      Cap.READONLY_QUALITY_PREFLIGHT},
        quality_preflight=Mock(return_value=receipt))
    assert p.preflight_backend(provider)['status'] == 'unknown'


def test_provider_exception_does_not_expose_connection_or_key_details():
    provider = SimpleNamespace(
        capabilities={Cap.REAL_DEVICE,Cap.REMOTE_VERIFY,Cap.GPU_PINNING,
                      Cap.READONLY_QUALITY_PREFLIGHT},
        quality_preflight=Mock(side_effect=RuntimeError('sensitive-user /private/secret.key')))
    result = p.preflight_backend(provider)
    assert result['status'] == 'unknown'
    assert 'sensitive-user' not in json.dumps(result) and 'secret.key' not in json.dumps(result)


def test_default_close_states_only_that_no_backend_session_is_owned():
    result = LocalExecutor().close()
    assert result == dict(quiescent=True, session_created=False, scope='owned_backend_sessions')


def test_unresolved_facade_close_does_not_probe_or_reconnect(monkeypatch):
    facade = RemoteExecutor(mode='ssh', host='synthetic-host')
    initialize = Mock(side_effect=AssertionError('cleanup must not initialize a backend'))
    monkeypatch.setattr(facade, '_init_backend', initialize)
    result = facade.close()
    initialize.assert_not_called()
    assert result == dict(quiescent=True, session_created=False,
                          backend_resolved=False, scope='owned_backend_sessions')


def test_resolved_facade_close_forwards_to_existing_session_owner(monkeypatch):
    inner = LocalExecutor()
    close = Mock(return_value=dict(quiescent=False, scope='owned_backend_sessions'))
    monkeypatch.setattr(inner, 'close', close)
    facade = RemoteExecutor(mode='ssh', host='synthetic-host')
    facade._backend = inner
    result = facade.close(grace=.5)
    close.assert_called_once_with(grace=.5)
    assert result == dict(quiescent=False, scope='owned_backend_sessions')


def test_persistent_mode_is_explicitly_refused_before_running_probe(tmp_path):
    backend = ssh(tmp_path)
    backend._persistent = True
    backend._deployed = True
    backend._verified_python = '.venv/bin/python'
    result = p.preflight_backend(backend)
    assert result['status'] == 'failed'
    assert result['checks'][-1] == dict(name='execution_mode', status='failed',
                                      reason='persistent_not_supported')
    assert backend._deployed is False and backend._verified_python is None
    backend._pinned_gpu.assert_not_called()
    backend._ssh_exec.assert_not_called()


@pytest.mark.parametrize('phase,error_type', [('runner','FileNotFoundError'),
                                           ('runtime','ModuleNotFoundError'),
                                           ('control','RuntimeError')])
def test_missing_runner_import_or_control_failure_refuses_readiness(tmp_path, phase, error_type):
    def mutate(value):
        value.update(status='failed', phase=phase, error_type=error_type)
        if phase == 'runner':
            value['runner_verified'] = False
    backend = ssh(tmp_path, mutate)
    result = p.preflight_backend(backend)
    assert result['status'] == 'failed'
    assert result['checks'][-1] == dict(name=phase,status='failed',error_type=error_type)
    assert backend._deployed is False and backend._verified_python is None


@pytest.mark.parametrize('field,value', [('uuid',''), ('uuid','unknown'), ('index',1),
    ('torch_version','not installed'), ('triton_version','unknown'), ('cuda_version',''),
    ('numpy_version','unknown'), ('numpy_version',''),
    ('device','cpu'), ('compute_capability','unknown')])
def test_unknown_or_different_gpu_and_runtime_identity_refuses_readiness(tmp_path, field, value):
    backend = ssh(tmp_path, lambda result: result['identity'].update({field:value}))
    result = p.preflight_backend(backend)
    assert result['status'] == 'unknown'
    assert result['checks'][-1]['name'] == 'runtime_identity'
    assert backend._deployed is False


@pytest.mark.parametrize('field,value', [('passed',False), ('finite',False),
    ('compared_elements',256), ('compared_elements',True), ('device','cpu'),
    ('dtype','torch.float16'), ('max_abs_error',.1), ('max_abs_error',False)])
def test_partial_or_inconsistent_control_results_never_pass(tmp_path, field, value):
    backend = ssh(tmp_path, lambda result: result['control'].update({field:value}))
    assert p.preflight_backend(backend)['status'] == 'unknown'
    assert backend._verified_python is None


@pytest.mark.parametrize('field,value', [('runner_verified',False),
                                      ('runner_hashes_identity','different'),
                                      ('control_sha256','different')])
def test_source_or_runner_hash_drift_never_grants_verified_execution(tmp_path, field, value):
    backend = ssh(tmp_path, lambda result: result.update({field:value}))
    assert p.preflight_backend(backend)['status'] == 'unknown'
    assert backend._deployed is False and backend._verified_python is None


def test_transport_timeout_preserves_unknown_and_does_not_expose_argv(tmp_path):
    backend = ssh(tmp_path)
    backend._ssh_exec.side_effect = subprocess.TimeoutExpired(
        ['ssh','-i','/private/connection/secret.key','sensitive-user@host'],300)
    result = p.preflight_backend(backend)
    assert result['status'] == 'unknown'
    assert backend.timeout == 300
    serialized = json.dumps(result)
    assert 'sensitive-user' not in serialized and 'secret.key' not in serialized
    assert 'TimeoutExpired' in serialized


@pytest.mark.parametrize('output', ['not JSON', '{}\n{}', '{"x":NaN}', '{"x":1,"x":2}', 'x'*20000])
def test_missing_ambiguous_nonfinite_or_oversized_probe_evidence_stays_unknown(tmp_path, output):
    backend = ssh(tmp_path)
    backend._ssh_exec.side_effect = None
    backend._ssh_exec.return_value = output
    assert p.preflight_backend(backend)['status'] == 'unknown'


def test_gpu_selection_unknown_stops_without_running_control(tmp_path):
    backend = ssh(tmp_path)
    backend._pinned_gpu.return_value = None
    assert p.preflight_backend(backend)['status'] == 'unknown'
    backend._ssh_exec.assert_not_called()


def test_recheck_invalidates_stale_verified_runtime_state_on_failure(tmp_path):
    backend = ssh(tmp_path)
    assert p.preflight_backend(backend)['status'] == 'passed'
    backend._pinned_gpu.return_value = None
    assert p.preflight_backend(backend)['status'] == 'unknown'
    assert backend._deployed is False and backend._verified_python is None


def test_timeout_is_bounded_and_restored_and_shell_workdir_is_literal(tmp_path):
    backend = ssh(tmp_path)
    backend.work_dir = '/project with spaces/$(must-not-run)'
    observed = []
    def reply(command, stdin_data):
        observed.append(backend.timeout)
        assert command.startswith("cd '/project with spaces/$(must-not-run)' && ")
        return json.dumps(result_for(stdin_data))
    backend._ssh_exec.side_effect = reply
    assert p.preflight_backend(backend)['status'] == 'passed'
    assert observed == [p.MAX_PROBE_SECONDS]
    assert backend.timeout == 300


def test_probe_source_keeps_installs_and_persistent_cache_out_of_the_control():
    assert 'TemporaryDirectory' in p.PROBE_SOURCE
    assert "os.environ['TRITON_CACHE_DIR']" in p.PROBE_SOURCE
    assert "os.environ['CUDA_CACHE_PATH']" in p.PROBE_SOURCE
    assert 'pip install' not in p.PROBE_SOURCE and 'uv add' not in p.PROBE_SOURCE


@pytest.mark.parametrize('missing', [True, False])
def test_actual_probe_program_checks_remote_runner_bytes_before_imports(tmp_path, missing):
    backend = ssh(tmp_path)
    deployed = tmp_path/'remote-runner.py'
    if not missing:
        deployed.write_text('stale deployed source\n')
    local = backend._remote_files.return_value[0][0]
    backend._remote_files.return_value = [(local, str(deployed))]
    def execute_probe(command, stdin_data):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(stdin_data, '<readiness-probe>', 'exec'), {})
        return captured.getvalue()
    backend._ssh_exec.side_effect = execute_probe
    result = p.preflight_backend(backend)
    assert result['status'] == 'failed'
    assert result['checks'][-1]['name'] == 'runner'
    assert result['checks'][-1]['error_type'] == ('FileNotFoundError' if missing else 'RuntimeError')
    assert result['runner_verified'] is False


@pytest.mark.parametrize('missing_module', ['torch', 'triton', 'numpy'])
def test_actual_probe_program_reports_runtime_import_failure_without_install(tmp_path, monkeypatch, missing_module):
    backend = ssh(tmp_path)
    local = backend._remote_files.return_value[0][0]
    backend._remote_files.return_value = [(local, str(local))]
    original_import = builtins.__import__
    def reject_runtime(name, *args, **kwargs):
        if name == missing_module:
            raise ModuleNotFoundError('synthetic missing runtime')
        if name in ('torch', 'triton', 'numpy'):
            return SimpleNamespace(__version__='synthetic')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', reject_runtime)
    # The probe normally runs in its own remote process. Give this in-process
    # program test its own environment mapping so temporary cache settings vanish.
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    def execute_probe(command, stdin_data):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(stdin_data, '<readiness-probe>', 'exec'), {})
        return captured.getvalue()
    backend._ssh_exec.side_effect = execute_probe
    result = p.preflight_backend(backend)
    assert result['status'] == 'failed'
    assert result['checks'][-1] == dict(name='runtime', status='failed',
                                      component=missing_module, error_type='ModuleNotFoundError')
    assert result['runner_verified'] is True
    assert backend._verified_python is None
