import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from benchmarks.delivery import install_dsh as installer


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
    monkeypatch.setattr(installer,'check_python',lambda _: '0.7.2rc1')
    monkeypatch.setattr(installer.shutil,'which',lambda _: '/usr/bin/node')
    monkeypatch.setattr(installer,'archive_identity',lambda _:dict(sha256='a'*64,
        files={'host-model-budget.mjs':'b','host-model-ledger.mjs':'c','host-session-lifecycle.mjs':'d'}))
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
