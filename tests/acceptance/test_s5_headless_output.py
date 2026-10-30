"""Failed headless parsing must retain diagnostics without accepting noise."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

import pytest

from benchmarks.experiments.s5 import s5_arm_env as arms


def launch(tmp_path, monkeypatch, result):
    home = tmp_path/'participant/home'; home.mkdir(parents=True)
    work = tmp_path/'participant/work'; work.mkdir()
    entry = tmp_path/'entry.js'; entry.write_text('unused')
    built = dict(home=str(home), work=str(work), dsh_entry=str(entry), patch=[], env={})
    monkeypatch.setattr(arms, 'NODE', 'fake-node')
    def run(*args, **kwargs):
        assert not kwargs.get('text')
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(arms.subprocess, 'run', run)
    output = tmp_path/'supervisor/output'
    return built, output


def test_lf_framing_preserves_unicode_separators_and_original_bytes(tmp_path, monkeypatch):
    event = dict(type='final', text='a\x85b\u2028c\u2029d')
    stdout = (json.dumps(event, ensure_ascii=False)+'\n').encode()
    proc = subprocess.CompletedProcess([], 0, stdout, b'diagnostic\r\n')
    built, output = launch(tmp_path, monkeypatch, proc)
    response = arms.start_headless(built, 'unused', capture_dir=output)
    assert response['events'] == [event]
    assert (output/'stdout.bin').read_bytes() == stdout
    assert (output/'stderr.bin').read_bytes() == b'diagnostic\r\n'
    assert response['output_capture']['stdout']['sha256'] == hashlib.sha256(stdout).hexdigest()
    assert not response['output_capture']['stdout']['truncated']


@pytest.mark.parametrize('stdout,reason', [
    (b'{"type":"session"}\nconsole noise\n{"type":"final","text":"OK"}\n', 'not JSONL at line 2'),
    (b'{"type":"session"}\n\n', 'not JSONL at line 2'),
    (b'[]\n', 'not an object at line 1'),
    (b'\xff\n', 'not UTF-8'),
])
def test_invalid_output_retained_and_never_filtered_into_success(tmp_path, monkeypatch, stdout, reason):
    built, output = launch(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, stdout, b'err'))
    with pytest.raises(RuntimeError, match=reason):
        arms.start_headless(built, 'unused', capture_dir=output)
    assert (output/'stdout.bin').read_bytes() == stdout
    assert (output/'stderr.bin').read_bytes() == b'err'
    assert json.loads((output/'metadata.json').read_text())['returncode'] == 0


def test_timeout_retains_partial_streams(tmp_path, monkeypatch):
    exc = subprocess.TimeoutExpired('fixture', 1, output=b'partial-json', stderr=b'timeout diagnostic')
    built, output = launch(tmp_path, monkeypatch, exc)
    with pytest.raises(subprocess.TimeoutExpired):
        arms.start_headless(built, 'unused', capture_dir=output)
    assert (output/'stdout.bin').read_bytes() == b'partial-json'
    assert json.loads((output/'metadata.json').read_text())['timed_out'] is True


def test_oversized_capture_is_bounded_and_refuses_execution_result(tmp_path, monkeypatch):
    stdout = b'x'*(arms.OUTPUT_BOUND+1)
    built, output = launch(tmp_path, monkeypatch, subprocess.CompletedProcess([], 0, stdout, b''))
    with pytest.raises(RuntimeError, match='exceeded the harness bound'):
        arms.start_headless(built, 'unused', capture_dir=output)
    assert (output/'stdout.bin').stat().st_size == arms.OUTPUT_BOUND
    meta = json.loads((output/'metadata.json').read_text())
    assert meta['stdout']['bytes'] == arms.OUTPUT_BOUND+1 and meta['stdout']['truncated']
    assert meta['stdout']['sha256'] == hashlib.sha256(stdout).hexdigest()


SDK = arms.DSH_ROOT.parent/'@modelcontextprotocol/client/dist/index.mjs'
READY = bool(shutil.which('node')) and SDK.is_file()


@pytest.mark.skipif(not READY, reason='installed Node MCP SDK unavailable')
def test_installed_sdk_resource_diagnostics_reproduce_and_go_to_stderr():
    script = ('import {Client} from '+json.dumps(SDK.as_uri())+';'
              'const c=new Client({name:"fixture",version:"1"});'
              'await c.listResources();await c.listResourceTemplates();'
              'process.stdout.write(JSON.stringify({type:"final",text:"OK"})+"\\n");')
    baseline = subprocess.run([arms.NODE, '--input-type=module', '-e', script],
                              capture_output=True, check=True)
    assert b'Client.listResources()' in baseline.stdout
    with pytest.raises(RuntimeError, match='not JSONL'):
        arms._parse_events(baseline.stdout.decode())
    fixed = subprocess.run([arms.NODE, '--import', arms.DEBUG_PRELOAD,
                            '--input-type=module', '-e', script], capture_output=True, check=True)
    assert arms._parse_events(fixed.stdout.decode()) == [dict(type='final', text='OK')]
    assert b'Client.listResources()' in fixed.stderr
    assert b'Client.listResourceTemplates()' in fixed.stderr


@pytest.mark.skipif(not READY, reason='installed DSH runtime unavailable')
def test_actual_headless_resources_keep_jsonl_and_capture_diagnostics(tmp_path):
    from benchmarks.experiments.s5 import s5_seatbelt, s5_runtime
    if sys.platform != 'darwin' or not s5_runtime.available():
        pytest.skip('macOS runtime required')
    from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
    from benchmarks.experiments.s5.s5_vertical_tool_provider import _sse
    from operator_agent import s5_shared_budget as budget
    from benchmarks.experiments.s5.s5_formal_run import _final_text
    ledger=tmp_path/'budget.sqlite'
    budget.create(ledger, task_id='resource-fixture', max_calls=3, max_output_tokens=1024,
                  total_output_tokens=3072, wall_clock_s=120)
    requests=[]
    text='OK\x85\u2028\u2029'
    def provider(request):
        requests.append(request)
        number=len(requests)
        if number==3:
            return _sse(dict(content=text), 'stop')
        assert number<3
        name=('list_mcp_resources' if number==1 else 'list_mcp_resource_templates')
        call=dict(index=0, id='resource_'+str(number), type='function',
            function=dict(name=name, arguments=json.dumps(dict(server='operator'))))
        return _sse(dict(tool_calls=[call]), 'tool_calls')
    runtime=s5_seatbelt.stage_runtime(tmp_path/'runtime')
    with GatewayServer(ledger,task_id='resource-fixture',model='fake-model',participant='outer',
        adapter=provider,max_calls_limit=3) as gateway:
        built=arms.build('C',base_url=gateway.base_url,arm_dir=tmp_path/'participant',
                         tools=True,max_tokens=1024,isolation_runtime=runtime)
        response=arms.start_headless(built,'Only execute the scripted fixture.',timeout=90,
                                     capture_dir=tmp_path/'capture')
    assert response['returncode']==0 and _final_text(response['events'])==text
    assert len(requests)==3
    stderr=(tmp_path/'capture/stderr.bin').read_bytes()
    assert b'Client.listResources()' in stderr and b'Client.listResourceTemplates()' in stderr
    assert arms._parse_events((tmp_path/'capture/stdout.bin').read_text())==response['events']
