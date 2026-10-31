"""Readiness evidence before paid generation; never provision the SSH runtime.

The normal SSH hardware/compile APIs may deploy and install dependencies. This
adapter instead uses the existing SSH transport and already-present project
interpreter. It runs a fixed control in a private temporary directory and cache;
it does not install packages, deploy a runner, or verify an authored candidate.
Transport timeout is bounded, but does not prove remote GPU cancellation.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import time

from ..core.backend_caps import Cap
from ..execute.llm_budget import strict_object
from ..core.run_contract import code_hash, digest


PREFLIGHT_VERSION = 'operator-quality-preflight-v1'
PROBE_SCHEMA = 'operator-quality-runtime-probe-v1'
MAX_PROBE_SECONDS = 60
MAX_PROBE_BYTES = 16384

# This control is deliberately independent of model replies and task references.
# Its 257-element tail exercises real JIT launch, loads, stores and synchronization.
CONTROL_SOURCE = '''import torch
import triton
import triton.language as tl

@triton.jit
def _quality_add(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(C + offsets, a + b, mask=mask)

def kernel(A, B):
    C = torch.empty_like(A)
    _quality_add[(triton.cdiv(A.numel(), 128),)](A, B, C, A.numel(), 128)
    return C
'''


# Kept separate from the control so importing this module needs no torch/triton.
PROBE_SOURCE = '''import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

identity = {}
phase = 'runner'
component = None
result = {'schema': SCHEMA, 'status': 'unknown', 'identity': identity,
          'control_sha256': CONTROL_HASH, 'runner_verified': False}
try:
    for declared in RUNNER_FILES:
        path = Path(os.path.expanduser(declared['path']))
        component = path.name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != declared['sha256']:
            raise RuntimeError('runner_source_drift')
    result.update(runner_verified=True, runner_hashes_identity=RUNNER_HASHES_IDENTITY)
    phase = 'runtime'
    with tempfile.TemporaryDirectory(prefix='operator-quality-preflight-') as scratch:
        os.environ['TRITON_CACHE_DIR'] = str(Path(scratch) / 'triton-cache')
        os.environ['CUDA_CACHE_PATH'] = str(Path(scratch) / 'cuda-cache')
        os.environ['XDG_CACHE_HOME'] = str(Path(scratch) / 'xdg-cache')
        component = 'torch'
        import torch
        component = 'triton'
        import triton
        component = 'numpy'
        import numpy
        component = 'cuda'
        if not torch.cuda.is_available():
            raise RuntimeError('cuda_unavailable')
        properties = torch.cuda.get_device_properties(0)
        identity.update(device='cuda', index=GPU_INDEX,
            uuid=str(getattr(properties, 'uuid', '')),
            name=properties.name,
            compute_capability=str(properties.major) + '.' + str(properties.minor),
            torch_version=str(torch.__version__), triton_version=str(triton.__version__),
            numpy_version=str(numpy.__version__),
            cuda_version=str(torch.version.cuda or ''))
        phase = 'control'
        component = 'triton_control'
        module_path = Path(scratch) / 'control.py'
        module_path.write_text(CONTROL)
        spec = importlib.util.spec_from_file_location('_operator_quality_control', str(module_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        a = torch.arange(257, device='cuda', dtype=torch.float32) / 17
        b = torch.full_like(a, 0.5)
        output = module.kernel(a, b)
        torch.cuda.synchronize()
        expected = a + b
        valid = (output.device.type == 'cuda' and output.dtype == torch.float32
                 and output.shape == a.shape and bool(torch.isfinite(output).all().item())
                 and bool(torch.equal(output, expected)))
        result.update(status='passed' if valid else 'failed', phase=phase,
            control=dict(passed=valid, device=output.device.type, dtype=str(output.dtype),
                         compared_elements=output.numel(), finite=bool(torch.isfinite(output).all().item()),
                         max_abs_error=float((output - expected).abs().max().item())))
except Exception as error:
    result.update(status='failed', phase=phase, component=component,
                  error_type=type(error).__name__)
print(json.dumps(result, allow_nan=False))
'''


def preflight_identity() -> dict:
    """A detached identity for frozen plans, with no environment probing."""
    return dict(version=PREFLIGHT_VERSION, source_sha256=code_hash(Path(__file__).read_text()),
                control_sha256=code_hash(CONTROL_SOURCE), probe_sha256=code_hash(PROBE_SOURCE),
                max_probe_seconds=MAX_PROBE_SECONDS)


def _finish(checks, identity, status, *, runner_verified=False,
            runner_hashes_identity=None, **extra):
    body = dict(schema=PREFLIGHT_VERSION, status=status, identity=identity, checks=checks,
                policy=preflight_identity(), observed_at=time.time(),
                model_calls=0, candidate_correctness_proven=False,
                performance_measured=False, dependencies_installed=False,
                runner_deployed=False, runner_verified=runner_verified,
                runner_hashes_identity=runner_hashes_identity,
                remote_cancellation_proven=False, **extra)
    return dict(body, hash=digest(body))


def _remote_path(value):
    # Preserve the one documented remote-home prefix without evaluating shell
    # substitutions. All user-configurable remaining path characters are quoted.
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('invalid remote work directory')
    return ('"$HOME"/' + shlex.quote(value[2:]) if value.startswith('~/')
            else shlex.quote(value))


def _text(value):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 160
            and value.lower().strip() not in ('unknown', 'none', 'n/a', 'not installed'))


def _identity(raw, gpu):
    fields = ('device', 'index', 'uuid', 'name', 'compute_capability',
              'torch_version', 'triton_version', 'numpy_version', 'cuda_version')
    if not isinstance(raw, dict):
        return None
    if (raw.get('device') != 'cuda' or type(raw.get('index')) is not int
            or raw['index'] != gpu or any(not _text(raw.get(k)) for k in fields
                                        if k not in ('device', 'index'))
            or not re.fullmatch(r'[A-Za-z0-9_.:-]+', raw['uuid'])
            or not re.fullmatch(r'[0-9]+\.[0-9]+', raw['compute_capability'])):
        return None
    return {key: raw[key] for key in fields}


def preflight_backend(backend) -> dict:
    """Call a declared readonly readiness provider before creating a model.

    Capabilities are the dispatch contract. The provider owns its probe and
    returns passed/failed/unknown evidence; only ``passed`` permits generation.
    Readiness never bypasses candidate L2.
    """
    checks = []
    identity = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            capabilities = frozenset(backend.capabilities)
    except Exception as error:
        checks.append(dict(name='backend_capabilities', status='unknown', error_type=type(error).__name__))
        return _finish(checks, identity, 'unknown')
    required = {Cap.REAL_DEVICE, Cap.REMOTE_VERIFY, Cap.GPU_PINNING,
                Cap.READONLY_QUALITY_PREFLIGHT}
    missing = required - capabilities
    if missing or capabilities & {Cap.EMULATED, Cap.SIMULATED_PERF}:
        checks.append(dict(name='backend_capabilities', status='failed',
                           missing=sorted(cap.value for cap in missing)))
        return _finish(checks, identity, 'failed')
    checks.append(dict(name='backend_capabilities', status='passed'))

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = backend.quality_preflight()
        if (not isinstance(result, dict) or result.get('schema') != PREFLIGHT_VERSION
                or result.get('status') not in ('passed', 'failed', 'unknown')
                or not isinstance(result.get('identity'), dict)
                or (result.get('status') == 'passed' and result.get('runner_verified') is not True)
                or result.get('hash') != digest({k: v for k, v in result.items() if k != 'hash'})
                or type(result.get('model_calls')) is not int or result['model_calls'] != 0
                or result.get('dependencies_installed') is not False
                or result.get('runner_deployed') is not False
                or result.get('candidate_correctness_proven') is not False
                or result.get('performance_measured') is not False):
            raise ValueError('invalid readonly quality preflight receipt')
        return result
    except Exception as error:
        checks.append(dict(name='readonly_quality_preflight', status='unknown', error_type=type(error).__name__))
        return _finish(checks, identity, 'unknown')


def _preflight_ssh(inner) -> dict:
    """NVIDIA SSH provider implementation, called by its declared capability.

    The same provider retains the physical GPU pin for subsequent one-shot
    execution. Existing runner bytes are checked before marking them deployed;
    this function never deploys or installs dependencies itself.
    """
    checks = [dict(name='backend_capabilities', status='passed'),
              dict(name='ssh_backend', status='passed')]
    identity = {}
    inner._verified_python = None
    inner._deployed = False
    if inner._persistent:
        checks.append(dict(name='execution_mode', status='failed', reason='persistent_not_supported'))
        return _finish(checks, identity, 'failed')
    timeout = inner.timeout
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        checks.append(dict(name='probe_transport', status='unknown', reason='invalid_timeout'))
        return _finish(checks, identity, 'unknown')
    inner.timeout = min(timeout, MAX_PROBE_SECONDS)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gpu = inner._pinned_gpu()
            if type(gpu) is not int or gpu < 0 or inner.selected_gpu != gpu:
                checks.append(dict(name='gpu_selection', status='unknown', reason='gpu_not_pinned'))
                return _finish(checks, identity, 'unknown')
            checks.append(dict(name='gpu_selection', status='passed', index=gpu))
            runner_files = []
            runner_hashes = {}
            for local, remote in inner._remote_files():
                if not isinstance(remote, str) or not remote or len(runner_files) >= 64:
                    raise ValueError('invalid runner deployment declaration')
                local = Path(local)
                name = local.name
                if name in runner_hashes:
                    raise ValueError('duplicate runner deployment declaration')
                runner_hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()
                runner_files.append(dict(path=remote, sha256=runner_hashes[name]))
            if not runner_files:
                raise ValueError('empty runner deployment declaration')
            runner_identity = digest(runner_hashes)
            source = ('SCHEMA = ' + repr(PROBE_SCHEMA) + '\nGPU_INDEX = ' + str(gpu)
                      + '\nCONTROL_HASH = ' + repr(code_hash(CONTROL_SOURCE))
                      + '\nRUNNER_FILES = ' + repr(runner_files)
                      + '\nRUNNER_HASHES_IDENTITY = ' + repr(runner_identity)
                      + '\nCONTROL = ' + repr(CONTROL_SOURCE) + '\n' + PROBE_SOURCE)
            command = ('cd ' + _remote_path(inner.work_dir)
                       + ' && test -x .venv/bin/python && CUDA_VISIBLE_DEVICES=' + str(gpu)
                       + ' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -')
            output = inner._ssh_exec(command, stdin_data=source)
        if not isinstance(output, str) or len(output.encode()) > MAX_PROBE_BYTES:
            raise ValueError('invalid probe response size')
        # Torch/Triton may print warnings. Take exactly one tagged JSON result;
        # extra JSON objects, duplicate keys and nonfinite numbers are rejected.
        values = [strict_object(line.strip()) for line in output.splitlines()
                  if line.strip().startswith('{')]
        if len(values) != 1 or values[0].get('schema') != PROBE_SCHEMA:
            raise ValueError('missing or ambiguous probe result')
        result = values[0]
        if result.get('control_sha256') != code_hash(CONTROL_SOURCE):
            raise ValueError('control source identity drift')
        if result.get('status') == 'failed':
            phase = result.get('phase')
            if phase not in ('runner', 'runtime', 'control'):
                raise ValueError('unknown probe failure phase')
            check = dict(name=phase, status='failed')
            error_type = result.get('error_type')
            if isinstance(error_type, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', error_type):
                check['error_type'] = error_type
            allowed_components = (set(runner_hashes) if phase == 'runner' else
                                  {'torch', 'triton', 'numpy', 'cuda'} if phase == 'runtime' else
                                  {'triton_control'})
            if result.get('component') in allowed_components:
                check['component'] = result['component']
            checks.append(check)
            runner_verified = (result.get('runner_verified') is True
                               and result.get('runner_hashes_identity') == runner_identity)
            return _finish(checks, identity, 'failed',
                           runner_verified=runner_verified,
                           runner_hashes_identity=runner_identity if runner_verified else None)
        if (result.get('runner_verified') is not True
                or result.get('runner_hashes_identity') != runner_identity):
            checks.append(dict(name='runner', status='unknown', reason='runner_identity_not_verified'))
            return _finish(checks, identity, 'unknown')
        checks.append(dict(name='runner', status='passed', hashes_identity=runner_identity))
        observed = _identity(result.get('identity'), gpu)
        if observed is None:
            checks.append(dict(name='runtime_identity', status='unknown', reason='incomplete_gpu_or_runtime_identity'))
            return _finish(checks, identity, 'unknown', runner_verified=True,
                           runner_hashes_identity=runner_identity)
        identity = observed
        checks.append(dict(name='runtime_identity', status='passed'))
        control = result.get('control')
        if (result.get('status') != 'passed' or not isinstance(control, dict)
                or control.get('passed') is not True or control.get('finite') is not True
                or control.get('device') != 'cuda' or control.get('dtype') != 'torch.float32'
                or type(control.get('compared_elements')) is not int or control['compared_elements'] != 257
                or type(control.get('max_abs_error')) not in (int, float) or control['max_abs_error'] != 0):
            checks.append(dict(name='control', status='unknown', reason='incomplete_control_evidence'))
            return _finish(checks, identity, 'unknown', runner_verified=True,
                           runner_hashes_identity=runner_identity)
        checks.append(dict(name='control', status='passed', source_sha256=code_hash(CONTROL_SOURCE),
                           compared_elements=257, max_abs_error=0.0))
        # These flags reflect checked existing state. No deployment occurred.
        # The SSH execution adapter must preserve its `cd work_dir` prefix when
        # using this fixed interpreter fragment instead of auto-syncing uv run.
        inner._deployed = True
        inner._verified_python = '.venv/bin/python'
        return _finish(checks, identity, 'passed', runner_verified=True,
                       runner_hashes_identity=runner_identity)
    except Exception as error:
        checks.append(dict(name='probe_transport', status='unknown', error_type=type(error).__name__))
        return _finish(checks, identity, 'unknown')
    finally:
        inner.timeout = timeout
