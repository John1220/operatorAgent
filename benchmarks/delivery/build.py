"""Build a local candidate from an explicit working-tree snapshot, never HEAD only.

Run with the chosen Python build environment. This does not publish packages.
The output includes untracked implementation files and excludes runtime evidence,
local credentials, development environments and vendored checkouts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

ROOT=Path(__file__).resolve().parents[2]
if (Path(__file__).resolve().parent.parent/'source/pyproject.toml').is_file():
    ROOT=Path(__file__).resolve().parent.parent/'source'
FILES=('README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','MANIFEST.in',
       'pyproject.toml','config.example.yaml','.python-version','docs/testing/v07-delivery.md')
DIRECTORIES=('operator_agent','remote_scripts','integrations/dsh-operator','benchmarks/delivery')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2)


def command(argv,cwd,log,env=None):
    with Path(log).open('x',encoding='utf-8') as stream:
        result=subprocess.run(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('build failed; inspect '+str(log))


def build(output,python,source=ROOT):
    source=Path(source).resolve()
    root=Path(output).expanduser().resolve()
    if root==source or source in root.parents:raise ValueError('candidate must be outside source tree')
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    snapshot=root/'source';snapshot.mkdir()
    ignored=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','.DS_Store','node_modules',
        '*.egg-info','build','dist','.pytest_cache')
    for name in FILES:
        (snapshot/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,snapshot/name)
    for name in DIRECTORIES:
        shutil.copytree(source/name,snapshot/name,ignore=ignored)
    source_files={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    write(root/'source-manifest.json',source_files)
    artifacts=root/'artifacts';artifacts.mkdir()
    command([str(python),'-I','-m','build','--no-isolation','--sdist','--wheel','--outdir',str(artifacts)],
            snapshot,root/'python-build.log')
    env=dict(os.environ,npm_config_cache=str(root/'npm-cache'),npm_config_ignore_scripts='true')
    command(['npm','pack','--ignore-scripts','--json','--pack-destination',str(artifacts)],
            snapshot/'integrations/dsh-operator',root/'npm-pack.log',env)
    wheel,=artifacts.glob('*.whl');tgz,=artifacts.glob('*.tgz');sdist,=artifacts.glob('*.tar.gz')
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist()
        required=['operator_agent/execute/fake_backend/'+n+'.py' for n in ('__init__','backend','config','model')]
        required+=['operator_agent/prompts/coder_generate_triton.md','operator_agent/skills/triton_skill.md',
                   'remote_scripts/run_kernel.py']
        if any(n not in names for n in required):raise ValueError('wheel missing runtime files')
        if any(n.startswith(('tests/','benchmarks/','output/','logs/')) for n in names):
            raise ValueError('wheel contains excluded runtime/development files')
        if not all(any(n.endswith('/'+legal) for n in names) for legal in ('LICENSE','NOTICE')):
            raise ValueError('wheel missing legal files')
        for name in names:
            if name in source_files and hashlib.sha256(archive.read(name)).hexdigest()!=source_files[name]:
                raise ValueError('wheel source differs from snapshot: '+name)
    with tarfile.open(tgz) as archive:
        npm_names=archive.getnames()
        required=['package/'+n for n in ('index.mjs','cordis.patch.yml','package.json','LICENSE','NOTICE',
            'examples/v07/offline_smoke.py','examples/v07/profile-absolute-paths.patch.yml')]
        if any(n not in npm_names for n in required):raise ValueError('plugin missing delivery files')
        if any('__pycache__' in n or n.endswith('.pyc') for n in npm_names):
            raise ValueError('plugin contains generated caches')
    plugin=root/'plugin';plugin.mkdir()
    with tarfile.open(tgz) as archive:archive.extractall(plugin,filter='data')
    guides=root/'guide';guides.mkdir()
    shutil.copy2(source/'docs/testing/v07-delivery.md',guides/'v07-delivery.md')
    tools=root/'tools';tools.mkdir()
    for name in ('build.py','installed_smoke.py','dsh_profile.py'):
        if (source/'benchmarks/delivery'/name).is_file():shutil.copy2(source/'benchmarks/delivery'/name,tools/name)
    project=tomllib.loads((snapshot/'pyproject.toml').read_text())['project']
    npm=json.loads((snapshot/'integrations/dsh-operator/package.json').read_text())
    manifest=dict(schema='operator-v07-local-candidate-v1',product='v0.7',status='candidate_unreleased',
        python_package=project['version'],plugin_package=npm['version'],python_default='3.12',
        source_manifest_sha256=sha(root/'source-manifest.json'),
        artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in (wheel,sdist,tgz)},
        wheel_members=names,plugin_members=npm_names,
        formal_comparison_credit=False,gpu_acceptance=False,public_release=False)
    write(root/'candidate.json',manifest)
    (root/'SHA256SUMS').write_text(''.join(sha(p)+'  artifacts/'+p.name+'\n' for p in (wheel,sdist,tgz)))
    return dict(candidate=str(root),wheel=str(wheel),plugin=str(tgz),unpacked=str(plugin/'package'))


if __name__=='__main__':
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--source',default=str(ROOT),help='repository or retained source snapshot')
    args=parser.parse_args()
    print(json.dumps(build(args.output,Path(args.python).absolute(),args.source),ensure_ascii=False))
