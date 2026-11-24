from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tarfile

import pytest

from benchmarks.delivery import install_dsh as installer
from benchmarks.delivery.dsh_profile import archive_identity
from operator_agent.app import mcp_server

MANIFEST = mcp_server.build_manifest()


def test_yaml_comments_and_other_settings_are_preserved_and_rerun_is_idempotent():
    original='# my model\n- id: agent-default-model\n  config: {model: custom}\n'
    entries=installer.configs(Path('/path with spaces/bin/python'))
    result=installer.patch_text(original,entries)
    assert result.startswith(original)
    assert installer.patch_text(result,entries)==result
    assert result.count(installer.BEGIN)==result.count(installer.END)==1
    assert 'hostModelBudget' in result and 'operator-agent-mcp' in result


def test_json_patch_preserves_other_plugins_and_replaces_our_overrides():
    unrelated=dict(id='agent-default-model',config=dict(provider='mine',model='model'))
    old=[unrelated,dict(id='operator-tools',config=dict(modelRelay=dict(provider='legacy')))]
    entries=installer.configs(Path('/env/bin/python'))
    value=json.loads(installer.patch_text(json.dumps(old),entries))
    assert value==[unrelated,*entries]
    assert json.loads(installer.patch_text(json.dumps(value),entries))==value


def test_fresh_dsh_profile_comment_followed_by_empty_array():
    original='# Your patch layer\n# expressions allowed\n[]\n'
    entries=installer.configs(Path('/env/bin/python'))
    result=installer.patch_text(original,entries)
    assert result.startswith('# Your patch layer\n# expressions allowed\n')
    assert json.loads('\n'.join(result.splitlines()[2:]))==entries
    assert installer.patch_text(result,entries)==result


@pytest.mark.parametrize('content',[
    installer.BEGIN+'\n',installer.END+'\n',
    installer.BEGIN+'\n'+installer.END+'\n'+installer.BEGIN+'\n'+installer.END,
    installer.END+'\n'+installer.BEGIN,
])
def test_broken_managed_block_is_refused(content):
    with pytest.raises(ValueError):installer.patch_text(content,[])


def fake_install(tmp_path,monkeypatch):
    python=tmp_path/'env/bin/python';python.parent.mkdir(parents=True);python.touch()
    plugin=tmp_path/'plugin.tgz';plugin.touch()
    dsh=tmp_path/'dsh/lib/bin.js';dsh.parent.mkdir(parents=True);dsh.touch()
    home=tmp_path/'home'
    args=SimpleNamespace(python=str(python),plugin=str(plugin),dsh_bin=str(dsh),
        dsh_home=str(home),profile='test',dry_run=False)
    monkeypatch.setattr(installer,'check_python',lambda _python,_manifest: '0.7.2rc1')
    monkeypatch.setattr(installer.shutil,'which',lambda _: '/usr/bin/node')
    monkeypatch.setattr(installer,'archive_identity',lambda _:dict(sha256='a'*64,
        files={'host-model-budget.mjs':'b','host-model-ledger.mjs':'c',
               'host-session-lifecycle.mjs':'d','manifest.json':'e'},manifest=MANIFEST))
    return args,home


def test_dry_run_does_not_create_profile_or_backup(tmp_path,monkeypatch):
    args,home=fake_install(tmp_path,monkeypatch);args.dry_run=True
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**k:pytest.fail('dry-run changed DSH'))
    installer.install(args)
    assert not home.exists()


@pytest.mark.parametrize('original',[None,b'# important model config\n- id: my-model\n'])
def test_failed_composition_restores_exact_original_patch(tmp_path,monkeypatch,original):
    args,home=fake_install(tmp_path,monkeypatch)
    profile=home/'profiles/test';profile.mkdir(parents=True)
    patch=profile/'cordis.patch.yml'
    if original is not None:patch.write_bytes(original)
    manifest=b'{"dependencies":{}}\n';(profile/'package.json').write_bytes(manifest)
    def run(argv,**kwargs):
        return subprocess.CompletedProcess(argv,0,b'[]\n',b'')
    monkeypatch.setattr(installer.subprocess,'run',run)
    with pytest.raises(ValueError,match='composition differs'):installer.install(args)
    assert (patch.read_bytes() if patch.exists() else None)==original
    root,=list((home/'operator-install-receipts').iterdir())
    assert (root/'package.json.before').read_bytes()==manifest
    if original is not None:assert (root/'cordis.patch.yml.before').read_bytes()==original
    assert root.stat().st_mode&0o077==0


def test_missing_host_modules_are_refused_before_any_profile_write(tmp_path,monkeypatch):
    args,home=fake_install(tmp_path,monkeypatch)
    monkeypatch.setattr(installer,'archive_identity',lambda _:dict(files={},sha256='a'*64))
    with pytest.raises(ValueError,match='lacks'):installer.install(args)
    assert not home.exists()


def test_desktop_requires_electron_install_before_configuration(tmp_path,monkeypatch):
    args,home=fake_install(tmp_path,monkeypatch);args.profile='desktop'
    with pytest.raises(ValueError,match='managed by Electron'):installer.install(args)
    assert not home.exists()


def test_desktop_configure_only_never_calls_restricted_cli(tmp_path,monkeypatch):
    args,home=fake_install(tmp_path,monkeypatch);args.profile='desktop';args.configure_only=True
    profile=home/'profiles/desktop';profile.mkdir(parents=True)
    monkeypatch.setattr(installer,'profile_state',lambda *a:dict(dependency=True,
        bundle_registered=True,package_installed=True,installed_files_match=True))
    commands=[]
    def run(argv,**kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv,0,json.dumps(installer.configs(Path(args.python))).encode(),b'')
    monkeypatch.setattr(installer.subprocess,'run',run)
    installer.install(args)
    assert len(commands)==1 and '--input-type=module' in commands[0]
    assert (profile/'cordis.patch.yml').exists()


def python_environment(tmp_path):
    python=tmp_path/'env/bin/python';python.parent.mkdir(parents=True)
    python.touch();(python.parent/'operator-agent').touch();(python.parent/'operator-agent-mcp').touch()
    return python


def handshake(manifest, *, tools=None, version=None, tool_version=None):
    tools=deepcopy(mcp_server._tool_definitions(host_mode=True) if tools is None else tools)
    capabilities=dict(ok=True,version=version or manifest['package_version'],
                      tool_version=manifest['tool_version'] if tool_version is None else tool_version)
    replies=[
        dict(jsonrpc='2.0',id=1,result=dict(serverInfo=dict(
            name='operator',version=version or manifest['package_version']))),
        dict(jsonrpc='2.0',id=2,result=dict(tools=tools)),
        dict(jsonrpc='2.0',id=3,result=dict(content=[dict(type='text',text=json.dumps(capabilities))])),
    ]
    return subprocess.CompletedProcess([],0,''.join(json.dumps(row)+'\n' for row in replies),'')


def test_manifest_is_the_installers_only_tool_list():
    assert 'operator_normalize' in MANIFEST['host_tools']
    assert 'operator_profile' in MANIFEST['host_tools']
    assert installer.validate_manifest(MANIFEST) is MANIFEST
    assert not hasattr(installer,'TOOLS')


def test_manifest_missing_normalize_is_refused():
    manifest=deepcopy(MANIFEST)
    manifest['tools']=[row for row in manifest['tools'] if row['name']!='operator_normalize']
    with pytest.raises(ValueError,match='tools and host_tools differ'):
        installer.validate_manifest(manifest)


def test_python_handshake_accepts_the_manifest_surface(tmp_path,monkeypatch):
    python=python_environment(tmp_path)
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**k:handshake(MANIFEST))
    assert installer.check_python(python,MANIFEST)==MANIFEST['package_version']


def test_same_tool_names_with_a_changed_schema_are_refused(tmp_path,monkeypatch):
    python=python_environment(tmp_path)
    tools=deepcopy(mcp_server._tool_definitions(host_mode=True))
    tools[0]['inputSchema']['description']='same version, different contract'
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**k:handshake(MANIFEST,tools=tools))
    with pytest.raises(ValueError,match='schemas differ'):
        installer.check_python(python,MANIFEST)


def test_same_package_version_with_a_changed_tool_contract_is_refused(tmp_path,monkeypatch):
    python=python_environment(tmp_path)
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**k:handshake(
        MANIFEST,tool_version=MANIFEST['tool_version']+1))
    with pytest.raises(ValueError,match='tool contract differs'):
        installer.check_python(python,MANIFEST)


def test_python_package_version_must_match_the_manifest(tmp_path,monkeypatch):
    python=python_environment(tmp_path)
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**k:handshake(MANIFEST,version='0.0.0'))
    with pytest.raises(ValueError,match='package version differs'):
        installer.check_python(python,MANIFEST)


def test_archive_without_manifest_is_refused_before_profile_write(tmp_path,monkeypatch):
    args,home=fake_install(tmp_path,monkeypatch)
    monkeypatch.setattr(installer,'archive_identity',lambda _:dict(sha256='a'*64,
        files={'host-model-budget.mjs':'b','host-model-ledger.mjs':'c','host-session-lifecycle.mjs':'d'}))
    with pytest.raises(ValueError,match='manifest'):
        installer.install(args)
    assert not home.exists()


def test_archive_identity_reads_the_bundled_manifest(tmp_path):
    plugin=tmp_path/'plugin.tgz'
    files={
        'package.json':dict(name='dsh-operator-agent',version='0.2.5'),
        'manifest.json':MANIFEST,
        'index.mjs':'', 'cordis.patch.yml':'', 'LICENSE':'', 'NOTICE':'',
    }
    with tarfile.open(plugin,'w:gz') as archive:
        for name,body in files.items():
            data=body if isinstance(body,str) else json.dumps(body)
            payload=data.encode();info=tarfile.TarInfo('package/'+name);info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
    artifact=archive_identity(plugin)
    assert artifact['manifest']==MANIFEST
    assert len(artifact['files']['manifest.json'])==64
