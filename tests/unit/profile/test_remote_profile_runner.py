import ast
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'remote_scripts').is_dir())


def load_profile_functions():
    source = (ROOT / 'remote_scripts' / 'run_kernel.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    wanted = {'_write_profile_result', 'run_profile'}
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == wanted
    namespace = {'os': os, 'json': json}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / 'remote_scripts' / 'run_kernel.py'), 'exec'),
         namespace)
    return namespace['run_profile']


class FakeTensor:
    def __init__(self, values, dtype='float32'):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def reshape(self, _shape):
        return self

    def __getitem__(self, index):
        value = self.values[index]
        return SimpleNamespace(item=lambda: value)

    def item(self):
        return self.values[0]


class FakeDecoy:
    def __init__(self, events):
        self.events = events

    def __add__(self, _value):
        self.events.append('decoy')
        return self


def fake_runtime(events, *, available=True):
    class Tensor(FakeTensor):
        pass

    torch = SimpleNamespace(
        Tensor=Tensor,
        float32='float32',
        device=lambda name: name,
        ones=lambda *_args, **_kwargs: FakeDecoy(events),
        no_grad=contextmanager(lambda: (yield)),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            synchronize=lambda: events.append('sync'),
            nvtx=SimpleNamespace(
                range_push=lambda name: events.append(('push', name)),
                range_pop=lambda: events.append('pop')),
            get_device_name=lambda _index: 'NVIDIA H20',
            get_device_properties=lambda _index: SimpleNamespace(uuid='GPU-fixture')),
        __version__='fixture-torch')
    return torch, SimpleNamespace(__version__='fixture-triton')


def test_profile_runner_writes_bound_result_and_captures_only_nvtx_range(tmp_path, monkeypatch):
    events = []
    torch, triton = fake_runtime(events)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'triton', triton)
    calls = []

    def kernel(X):
        calls.append(tuple(X.shape))
        return torch.Tensor([1.0, 2.0])

    run_profile = load_profile_functions()
    namespace = dict(run_profile.__globals__)
    namespace.update(
        torch=torch, triton=triton,
        shield_torch=contextmanager(lambda: (yield)),
        _deserialize_inputs=lambda inputs: inputs,
        _device_inputs=lambda inputs, _device, contiguous: inputs,
        _load_module=lambda _code: SimpleNamespace(kernel=kernel),
        materialize_seed_inputs=lambda spec: spec['inputs'],
        seed_input_digest=lambda inputs: hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()).hexdigest())
    run_profile.__globals__.update(namespace)
    result_path = tmp_path / 'nested' / 'result.json'

    result = run_profile('candidate', {'X': FakeTensor([3.0])},
                         result_path=str(result_path), warmup=2)

    assert result['success'] is True, result
    assert result['range'] == 'candidate'
    assert result['decoy'] is True
    assert result['checksum'] == dict(first=1.0, last=2.0, dtype='float32', shape=[2])
    assert calls == [(1,), (1,), (1,)]
    assert events.count('decoy') == 2
    assert events.index(('push', 'candidate')) < events.index('pop')
    assert json.loads(result_path.read_text()) == result


def test_profile_runner_refuses_seed_digest_drift_before_candidate(tmp_path, monkeypatch):
    events = []
    torch, triton = fake_runtime(events)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'triton', triton)
    called = []
    run_profile = load_profile_functions()
    run_profile.__globals__.update(
        torch=torch, triton=triton,
        _deserialize_inputs=lambda inputs: inputs,
        _device_inputs=lambda inputs, _device, contiguous: inputs,
        _load_module=lambda _code: SimpleNamespace(kernel=lambda **kwargs: called.append(kwargs)),
        materialize_seed_inputs=lambda spec: spec['inputs'],
        seed_input_digest=lambda _inputs: 'different')
    result_path = tmp_path / 'result.json'

    result = run_profile('candidate', {}, inputs_spec={'digest': 'expected', 'inputs': {}},
                         result_path=str(result_path))

    assert result['success'] is False
    assert result['error'] == 'profile input seed digest mismatch'
    assert called == []
    assert json.loads(result_path.read_text()) == result


def test_profile_runner_persists_failure_instead_of_losing_receipt(tmp_path, monkeypatch):
    torch, triton = fake_runtime([])
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'triton', triton)
    run_profile = load_profile_functions()
    run_profile.__globals__.update(
        torch=torch, triton=triton,
        _deserialize_inputs=lambda _inputs: (_ for _ in ()).throw(RuntimeError('bad inputs')))
    result_path = tmp_path / 'result.json'

    result = run_profile('candidate', {}, result_path=str(result_path))

    assert result['success'] is False
    assert 'bad inputs' in result['error']
    assert json.loads(result_path.read_text()) == result
