import json
import threading

import pytest

from benchmarks.s5_perf_mcp import PerformanceRelay


def make_relay(tmp_path):
    task = tmp_path / 'workspace' / 'task'
    task.mkdir(parents=True)
    (task / 'contract.json').write_text(json.dumps({'hash': 'a' * 64}))
    output = tmp_path / 'relay'
    output.mkdir()
    relay = object.__new__(PerformanceRelay)
    relay.task_dir = task.resolve()
    relay.workspace = task.parent.resolve()
    relay.output = output.resolve()
    relay.declaration = {'request': 'layernorm fp32 triton'}
    relay.pin = {'host': 'algorithm-h20', 'gpu_index': 2,
                 'gpu_uuid': 'GPU-12345678-1234-1234-1234-123456789abc',
                 'gpu_name': 'NVIDIA H20'}
    relay.max_candidates = 3
    relay.max_verify_calls = 3
    relay.max_bench_calls = 3
    relay.max_tool_gpu_calls = 6
    relay.allowed_tools = set({'operator_capabilities', 'operator_prepare',
                                'operator_verify', 'operator_bench'})
    relay.candidates = {}
    relay.reservation_lock = threading.Lock()
    relay.verify_calls = 0
    relay.bench_calls = 0
    relay.prepare_calls = 0
    return relay


def call(name, run_dir, **arguments):
    return {'jsonrpc': '2.0', 'id': 'test', 'method': 'tools/call',
            'params': {'name': name, 'arguments': {'run_dir': str(run_dir), **arguments}}}


def test_performance_relay_freezes_host_level_and_candidate_count(tmp_path):
    relay = make_relay(tmp_path)
    code = 'def kernel(X):\n    return X\n'
    result = relay._validate(call('operator_verify', relay.task_dir, code=code,
                                  stop_after='L2', backend='ssh', host='algorithm-h20'))
    assert result['params']['arguments']['timeout'] == 600
    assert relay.verify_calls == 1
    assert len(relay.candidates) == 1

    with pytest.raises(ValueError, match='limited to L2'):
        relay._validate(call('operator_verify', relay.task_dir, code=code,
                             stop_after='L3', backend='ssh', host='algorithm-h20'))

    other = 'def kernel(X):\n    return X + 1\n'
    relay._validate(call('operator_bench', relay.task_dir, backend='ssh',
        host='algorithm-h20', candidates=[{'name': 'second', 'code': other}],
        max_candidates=3, deadline_s=10))
    assert relay.bench_calls == 1
    assert len(relay.candidates) == 2


def test_performance_relay_never_allows_a_fourth_distinct_candidate(tmp_path):
    relay = make_relay(tmp_path)
    code = ['def kernel(X):\n    return X + %d\n' % value for value in range(4)]
    relay._validate(call('operator_verify', relay.task_dir, code=code[0], stop_after='L2'))
    relay._validate(call('operator_verify', relay.task_dir, code=code[1], stop_after='L2'))
    relay._validate(call('operator_bench', relay.task_dir,
        candidates=[{'name': 'third', 'code': code[2]}]))
    with pytest.raises(ValueError, match='three candidate limit'):
        relay._validate(call('operator_bench', relay.task_dir,
            candidates=[{'name': 'fourth', 'code': code[3]}]))


def test_performance_relay_caps_gpu_tool_calls_even_for_repeat_candidates(tmp_path):
    relay = make_relay(tmp_path)
    code = 'def kernel(X):\n    return X\n'
    for _ in range(3):
        relay._validate(call('operator_verify', relay.task_dir, code=code, stop_after='L2'))
    for _ in range(3):
        relay._validate(call('operator_bench', relay.task_dir,
            candidates=[{'name': 'same', 'code': code}]))
    with pytest.raises(ValueError, match='benchmark call limit|GPU tool call limit'):
        relay._validate(call('operator_bench', relay.task_dir,
            candidates=[{'name': 'same', 'code': code}]))


def test_performance_relay_accepts_task_directory_and_relative_candidate_files(tmp_path):
    relay = make_relay(tmp_path)
    seed = 'def kernel(X):\n    return X\n'
    candidate = 'def kernel(X):\n    return X + 1\n'
    (relay.workspace / 'candidate.py').write_text(seed)
    (relay.workspace / 'candidate_v1.py').write_text(candidate)

    bench = relay._validate(call('operator_bench', relay.task_dir,
        candidates=[{'name': 'seed', 'code_file': 'candidate.py'},
                    {'name': 'candidate-v1', 'code_file': 'candidate_v1.py'}]))
    assert bench['params']['arguments']['run_dir'] == str(relay.task_dir)
    assert bench['params']['arguments']['backend'] == 'ssh'
    assert [item['code'] for item in bench['params']['arguments']['candidates']] == [seed, candidate]

    verify = relay._validate(call('operator_verify', relay.task_dir,
                                  code_file='candidate_v1.py', stop_after='L2'))
    assert verify['params']['arguments']['run_dir'] == str(relay.task_dir)
    assert verify['params']['arguments']['backend'] == 'ssh'
    assert verify['params']['arguments']['code'] == candidate


def test_performance_relay_resolves_task_relative_to_session_workspace(tmp_path):
    relay = make_relay(tmp_path)
    (relay.workspace / 'candidate.py').write_text('def kernel(X):\n    return X\n')

    result = relay._validate(call('operator_verify', 'task', code_file='candidate.py'))
    assert result['params']['arguments']['run_dir'] == str(relay.task_dir)
    assert result['params']['arguments']['code'].startswith('def kernel')


def test_performance_relay_can_hide_prepare_for_an_already_frozen_task(tmp_path):
    relay = make_relay(tmp_path)
    relay.allowed_tools.discard('operator_prepare')
    with pytest.raises(ValueError, match='outside this performance experiment'):
        relay._validate(call('operator_prepare', relay.task_dir, **relay.declaration))
