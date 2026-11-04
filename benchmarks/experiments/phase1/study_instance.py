"""Read-only SSH instance observation, separate from preregistered expectations.

No provisioning, upload, package installation, model request or kernel execution
is performed. Capturing a receipt is an explicit network operation; validation
alone is pure. Hashes detect drift, not hostile same-UID receipt forgery.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

from operator_agent.core.run_contract import digest

SCHEMA='phase1-study-instance-v1'
_HEX=re.compile(r'[0-9a-f]{64}\Z')
_RUNTIME=('python','torch','triton','cuda')

# Fixed code, not user/candidate source. -I prevents the work directory or
# ambient PYTHONPATH from supplying spoofed torch/triton modules.
REMOTE_SCRIPT='''import csv,hashlib,json,pathlib,platform,subprocess,sys
import torch,triton
config=json.loads(sys.argv[1])
root=pathlib.Path(config["work_dir"])
versions=dict(python=platform.python_version(),torch=getattr(torch,"__version__",None),triton=getattr(triton,"__version__",None),cuda=getattr(torch.version,"cuda",None))
query=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15,check=True)
devices=[]
for row in csv.reader(query.stdout.splitlines()):
 if len(row)!=4:raise ValueError("unexpected NVIDIA identity row")
 index,uuid,name,driver=[v.strip() for v in row]
 devices.append(dict(index=int(index),uuid=uuid,name=name,driver_version=driver))
matching=[d for d in devices if d["index"]==config["gpu_index"]]
if len(matching)!=1:raise ValueError("selected GPU index unavailable or ambiguous")
files={}
for name in config["runner_names"]:
 path=root/name
 if path.is_symlink() or not path.is_file():raise ValueError("runner is not a regular file")
 files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(dict(runtime=versions,gpu=matching[0],runner_files=files),sort_keys=True,allow_nan=False))
'''


def _text(value,name):
    if not isinstance(value,str) or not value.strip() or any(c in value for c in '\r\n\x00'):
        raise ValueError('invalid '+name)
    return value


def _version(value,name):
    value=_text(value,name)
    if not re.fullmatch(r'[0-9]+\.[0-9]+[0-9A-Za-z.+_-]*',value):
        raise ValueError('missing or invalid '+name)
    return value


def _gpu(value):
    if (not isinstance(value,dict) or set(value)!={'index','uuid','name'}
            or type(value['index']) is not int or value['index']<0):
        raise ValueError('GPU requires explicit index, UUID and name')
    return dict(index=value['index'],uuid=_text(value['uuid'],'GPU UUID'),name=_text(value['name'],'GPU name'))


def _expected(value):
    if not isinstance(value,dict) or not value:raise ValueError('expected runner files are required')
    result={}
    for name,entry in value.items():
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*\.py',name):
            raise ValueError('runner names must be plain Python basenames')
        sha=entry.get('sha256') if isinstance(entry,dict) else entry
        if not isinstance(sha,str) or not _HEX.fullmatch(sha):raise ValueError('invalid expected runner SHA256')
        result[name]=sha
    return dict(sorted(result.items()))


def _connection(value):
    if (not isinstance(value,dict) or set(value)-{'host','port','user','key_path','work_dir','timeout'}
            or not {'host','work_dir'}<=set(value)):
        raise ValueError('explicit SSH host/work_dir required; unknown options refused')
    host=_text(value['host'],'SSH host')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',host) or host.startswith('-'):
        raise ValueError('invalid SSH host')
    user=value.get('user')
    if user is not None and (not re.fullmatch(r'[A-Za-z0-9_.-]+',_text(user,'SSH user')) or user.startswith('-') or '@' in host):
        raise ValueError('invalid or ambiguous SSH user')
    port=value.get('port',22)
    if type(port) is not int or not 1<=port<=65535:raise ValueError('invalid SSH port')
    work=_text(value['work_dir'],'remote runtime directory')
    if not work.startswith('/') or work=='/' or '..' in PurePosixPath(work).parts:
        raise ValueError('dedicated absolute remote runtime directory required')
    key=value.get('key_path')
    if key is not None:key=os.path.abspath(os.path.expanduser(_text(key,'SSH key path')))
    timeout=value.get('timeout',300.)
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=660:
        raise ValueError('invalid SSH observation timeout')
    return dict(host=host,port=port,user=user,key_path=key,work_dir=work,timeout=timeout)


def fixed_identity(receipt):
    """Fields that must remain unchanged; a later collected_at is expected."""
    return {key:receipt[key] for key in ('connection','gpu','runtime','runner_files','expected_runner_files')}


def validate(receipt,gpu,expected_runner_files):
    """Validate a retained observation without SSH, imports or new facts."""
    expected=_expected(expected_runner_files);gpu=_gpu(gpu)
    required={'schema','provenance','collected_at','connection','gpu','runtime','runner_files',
              'expected_runner_files','identity_sha256','kernel_executed','runtime_execution_verified','gpu_acceptance'}
    if not isinstance(receipt,dict) or set(receipt)!=required or receipt.get('schema')!=SCHEMA:
        raise ValueError('not a complete study instance receipt')
    if (receipt['provenance']!='ssh_read_only' or receipt['kernel_executed'] is not False
            or receipt['runtime_execution_verified'] is not False or receipt['gpu_acceptance'] is not False):
        raise ValueError('instance observation cannot claim GPU/runtime execution acceptance')
    stamp=receipt['collected_at']
    try:
        parsed=datetime.fromisoformat(stamp.replace('Z','+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds()!=0:raise ValueError()
    except (TypeError,AttributeError,ValueError):raise ValueError('UTC collection timestamp required') from None
    if receipt['connection']!=_connection(receipt['connection']):raise ValueError('noncanonical SSH binding')
    actual=receipt['gpu']
    if (not isinstance(actual,dict) or set(actual)!={'index','uuid','name','driver_version'}
            or type(actual.get('index')) is not int
            or any(actual.get(k)!=v for k,v in gpu.items())):
        raise ValueError('observed GPU index/UUID/name differs from the frozen GPU')
    _version(actual['driver_version'],'NVIDIA driver version')
    runtime=receipt['runtime']
    if not isinstance(runtime,dict) or set(runtime)!=set(_RUNTIME):raise ValueError('incomplete runtime versions')
    for name in _RUNTIME:
        _version(runtime[name],name+' version')
    if receipt['runner_files']!=expected or receipt['expected_runner_files']!=expected:
        raise ValueError('observed remote runner differs from local expected payload')
    if receipt['identity_sha256']!=digest(fixed_identity(receipt)):
        raise ValueError('instance identity hash mismatch')
    return receipt


def capture(connection,gpu,expected_runner_files):
    """Explicitly inspect an already-prepared SSH instance; never provision it."""
    connection=_connection(connection);gpu=_gpu(gpu);expected=_expected(expected_runner_files)
    payload=json.dumps(dict(work_dir=connection['work_dir'],gpu_index=gpu['index'],runner_names=list(expected)),
                       sort_keys=True,separators=(',',':'))
    python=str(PurePosixPath(connection['work_dir'])/'.venv/bin/python')
    remote=shlex.join([python,'-I','-B','-c',REMOTE_SCRIPT,payload])
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p',str(connection['port'])]
    if connection['key_path']:argv+=['-i',connection['key_path']]
    target=(connection['user']+'@' if connection['user'] else '')+connection['host']
    argv+=['--',target,remote]
    try:
        completed=subprocess.run(argv,capture_output=True,text=True,timeout=connection['timeout'],check=False)
    except (OSError,subprocess.SubprocessError):
        raise ValueError('read-only SSH instance observation failed; no receipt created') from None
    if completed.returncode!=0:
        # stderr may contain SSH/account paths or remote package diagnostics.
        raise ValueError('read-only SSH observation exited unsuccessfully; no receipt created')
    if len(completed.stdout.encode('utf-8'))>65536:raise ValueError('instance response exceeds bound')
    try:
        raw=json.loads(completed.stdout)
    except (ValueError,TypeError):raise ValueError('instance response is not JSON') from None
    if not isinstance(raw,dict) or set(raw)!={'runtime','gpu','runner_files'}:
        raise ValueError('unexpected instance response fields')
    receipt=dict(schema=SCHEMA,provenance='ssh_read_only',
        collected_at=datetime.now(timezone.utc).isoformat(),connection=connection,
        gpu=raw['gpu'],runtime=raw['runtime'],runner_files=raw['runner_files'],expected_runner_files=expected,
        kernel_executed=False,runtime_execution_verified=False,gpu_acceptance=False)
    receipt['identity_sha256']=digest(fixed_identity(receipt))
    return validate(receipt,gpu,expected)


def check(config):
    """Reobserve before each real cell/final; fail closed on instance drift."""
    if config.get('backend')=='fake':
        return dict(test_only=True,provenance='not_collected',reason='fake_backend_no_instance_observation',
                    gpu_acceptance=False)
    if config.get('backend')!='ssh':raise ValueError('study instance check requires fake or SSH backend')
    study=config.get('study')
    if not isinstance(study,dict) or not study.get('instance_receipt'):
        raise ValueError('study instance receipt required before dispatch')
    retained=study['instance_receipt']
    if isinstance(retained,str):retained=json.loads(Path(retained).read_text(encoding='utf-8'))
    expected=study.get('expected_runner_files')
    if expected is None and isinstance(retained,dict):expected=retained.get('expected_runner_files')
    validate(retained,config['gpu'],expected)
    connection=_connection(config['connection'])
    if retained['connection']!=connection:raise ValueError('SSH connection differs from frozen instance receipt')
    current=capture(connection,config['gpu'],expected)
    if fixed_identity(current)!=fixed_identity(retained):
        raise ValueError('instance GPU/runtime/driver/runner identity drift; freeze a new plan')
    return current
