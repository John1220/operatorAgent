"""A/C release-interface surfaces; core is never mounted into DSH's jail."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys

from benchmarks.experiments.s5 import s5_arm_env as legacy, s5_seatbelt as seatbelt

ROOT=Path(__file__).resolve().parents[3]
ARMS=('A','C')


def _python_files():
    return seatbelt.python_files()


def stage_runtime(dest):
    root=Path(dest).resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    sources={**{'plugin/'+name:ROOT/'integrations/dsh-operator'/name for name in seatbelt.PLUGIN_FILES},
             'proxy.py':ROOT/'benchmarks/experiments/s5/s5_plugin_proxy.py',
             'check.py':ROOT/'operator_agent/execute/s5_verify_client.py'}
    files={}
    for name,source in sources.items():
        target=root/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,target);files[name]=seatbelt.checksum(target)
    return dict(schema='s5-plugin-runtime-v1',root=str(root),files=files)


def verify_runtime(runtime):
    root=Path(runtime['root'])
    expected={**{'plugin/'+name:ROOT/'integrations/dsh-operator'/name for name in seatbelt.PLUGIN_FILES},
              'proxy.py':ROOT/'benchmarks/experiments/s5/s5_plugin_proxy.py',
              'check.py':ROOT/'operator_agent/execute/s5_verify_client.py'}
    if runtime.get('schema')!='s5-plugin-runtime-v1' or not root.is_dir() or root.is_symlink():
        raise ValueError('plugin runtime missing')
    if any(f.is_symlink() for f in root.rglob('*')):raise ValueError('plugin runtime symlink')
    actual={str(f.relative_to(root)):seatbelt.checksum(f)for f in root.rglob('*')if f.is_file()}
    if actual!=runtime['files'] or actual!={k:seatbelt.checksum(v)for k,v in expected.items()}:
        raise ValueError('plugin runtime/source drift')


def build(arm,*,runtime,relay=None,plugin_tool_timeout_ms=None,**kwargs):
    if arm not in ARMS:raise ValueError('only A/C are allowed')
    if arm=='C' and relay is None:raise ValueError('C requires the real MCP relay')
    if 'isolation_runtime'in kwargs:raise ValueError('legacy isolation cannot authorize plugin-only mode')
    tools=kwargs.pop('tools',True)
    if tools is not True:raise ValueError('plugin-only mode requires DSH tools')
    development=kwargs.get('development')
    if arm=='C':
        if plugin_tool_timeout_ms is None:
            if development is None:raise ValueError('C requires a bounded MCP tool timeout')
            plugin_tool_timeout_ms=(int(development['timeout'])+30)*1000
        if type(plugin_tool_timeout_ms)is not int or not 1<=plugin_tool_timeout_ms<=2147483647:
            raise ValueError('invalid MCP tool timeout')
    verify_runtime(runtime)
    # Reuse identical model settings and public stdlib client from the old A
    # surface, without ever adding its B/C PYTHONPATH or core CLI shim.
    built=legacy.build('A',tools=True,**kwargs)
    built['arm']=arm
    built['cli_shim']=None
    built['surface']=dict(arm=arm,label='native DSH'if arm=='A'else'DSH + plugin interface',
                          core_cli=False,core_source=False,plugin=arm=='C',mcp=arm=='C')
    built['protocol_mode']='plugin-only-ac-v1'
    home,work=Path(built['home']),Path(built['work'])
    for folder in(home,work,home.parent/'tmp'):folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Both groups get the exact same neutral test material. No product CLI.
    shutil.copyfile(Path(runtime['root'])/'check.py',work/'check.py')
    for name in('s5-verify','s5_public_verify.py'):
        (home.parent/'bin'/name).unlink(missing_ok=True)
    if arm=='C':
        config=home.parent/'mcp-client.json'
        with config.open('x')as f:json.dump(relay.client_config(),f)
        config.chmod(0o600)
        proxy=str(Path(runtime['root'])/'proxy.py')
        plugin=str(Path(runtime['root'])/'plugin/index.mjs')
        built['patch'].append(dict(insert=[
            dict(id='mcp-operator',name=str(legacy.MCP_CLIENT),config=dict(
                serverName='operator',transport='stdio',command=str(Path(sys.executable).resolve()),
                args=[proxy,'--config',str(config),'--stdio'],
                toolCallTimeoutMs=plugin_tool_timeout_ms,
                failOnStartupError=True,env=dict(PYTHONNOUSERSITE='1'))),
            dict(id='operator-tools',name=plugin,config=dict(command=str(Path(sys.executable).resolve()),
                args=[proxy,'--config',str(config)],env=dict(PYTHONNOUSERSITE='1')))]))
        built['mcp_client_config']=str(config)
    built['patch_sha256']=hashlib.sha256(json.dumps(built['patch'],sort_keys=True,
                                                  separators=(',',':')).encode()).hexdigest()
    configure(built,runtime)
    return built


def configure(built,runtime):
    if sys.platform!='darwin' or not Path(seatbelt.SANDBOX).is_file():
        raise RuntimeError('macOS Seatbelt required; no unconfined fallback')
    verify_runtime(runtime)
    root=Path(built['home']).parent
    if root.resolve().is_relative_to(ROOT):raise ValueError('participant must be outside source repository')
    reads=[Path(legacy.DSH_ROOT).parents[1],root/'bin']
    files=[*_python_files(),Path(legacy.NODE),root/'patch.json']
    development_config=built['env'].get('OPERATOR_AGENT_S5_DEV_CONFIG')
    if development_config:
        files.append(Path(development_config))
    if built['arm']=='C':
        reads.append(Path(runtime['root'])/'plugin')
        files.extend([Path(runtime['root'])/'proxy.py',Path(built['mcp_client_config'])])
    files.extend(Path(p)for p in('/private/etc/hosts','/private/etc/resolv.conf',
        '/private/etc/localtime','/private/etc/services','/private/var/select/sh'))
    writes=[Path(built['home']),Path(built['work']),root/'tmp']
    built['env'].update(TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1',
        DSH_PERMISSION_MODE='danger-full-access',OPERATOR_AGENT_S5_OUTER_REQUIRED='1')
    built['env'].pop('PYTHONPATH',None)
    text=seatbelt.profile(reads,files,writes,python_prefix=sys.base_prefix)
    path=root/'isolation.sb'
    with path.open('x')as f:f.write(text)
    # Reuse the proven wrapping implementation, while marking the different
    # runtime format so old ABC proofs cannot authorize this surface.
    built['isolation']=dict(schema='s5-plugin-seatbelt-v1',kind='macos-file-read-whitelist',
        profile=str(path),profile_sha256=seatbelt.checksum(path),runtime=runtime,
        read_roots=list(map(str,reads)),read_files=list(map(str,files)),write_roots=list(map(str,writes)),
        file_probe_passed=False,network_isolation_verified=False,
        inner_policy_delegated_to_outer=True)
    built['outer_isolation_required']=True


def wrap(built,command):
    policy=built.get('isolation')
    if not policy or policy.get('schema')!='s5-plugin-seatbelt-v1':raise ValueError('plugin isolation missing')
    if sys.platform!='darwin' or not Path(seatbelt.SANDBOX).is_file():raise RuntimeError('required sandbox unavailable')
    verify_runtime(policy['runtime'])
    p=Path(policy['profile'])
    if (p.is_symlink()or not p.is_file()or seatbelt.checksum(p)!=policy['profile_sha256']
        or p.read_text()!=seatbelt.profile(policy['read_roots'],policy['read_files'],policy['write_roots'],python_prefix=sys.base_prefix)):
        raise ValueError('plugin isolation policy drift')
    return [seatbelt.SANDBOX,'-f',str(p),*command]
