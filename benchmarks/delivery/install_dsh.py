"""Install a local DSH bundle and configure its existing Python environment.

No model requests, GPU execution or provider configuration. DSH's own plugin
manager enforces the Desktop profile lock. Receipts and exact config backups
are private, and a failed configuration check restores the pre-install patch.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

if __package__:
    from .dsh_profile import archive_identity, state as profile_state
else:
    from dsh_profile import archive_identity, state as profile_state

DEFAULT_DSH = Path('/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai/dsh/lib/bin.js')
BEGIN = '# BEGIN operator-agent host installer v1'
END = '# END operator-agent host installer v1'
TOOLS = sorted('operator_'+x for x in ('prepare','context','verify','resume','status','cancel','capabilities'))


def configs(python):
    return [dict(id='mcp-operator', disabled=False, config=dict(serverName='operator',
        transport='stdio',command=str(python.parent/'operator-agent-mcp'),
        toolCallTimeoutMs=3600000,failOnStartupError=True,env=dict(OPERATOR_AGENT_WORKFLOW='host'))),
        dict(id='operator-tools',disabled=False,config=dict(command=str(python.parent/'operator-agent'),
            timeoutMs=60000,hostModelBudget={}))]


def patch_text(original, entries):
    """Preserve YAML comments/tags verbatim; last overrides are DSH patch semantics."""
    lines=original.splitlines(keepends=True)
    first=next((i for i,line in enumerate(lines) if line.strip() and not line.lstrip().startswith('#')),len(lines))
    prefix=''.join(lines[:first]);body=''.join(lines[first:])
    if body.lstrip().startswith('['):
        values=json.loads(body)
        if not isinstance(values,list):raise ValueError('profile patch must be a list')
        ids={item['id'] for item in entries}
        return prefix+json.dumps([v for v in values if not isinstance(v,dict) or v.get('id') not in ids]
                          +entries,ensure_ascii=False,indent=2)+'\n'
    if original.count(BEGIN)!=original.count(END) or original.count(BEGIN)>1:
        raise ValueError('managed configuration markers are incomplete or duplicated')
    if BEGIN in original:
        start=original.index(BEGIN);finish=original.index(END)+len(END)
        if finish<start:raise ValueError('managed configuration markers are out of order')
        original=original[:start]+original[finish:].lstrip('\r\n')
    block='\n'.join('- '+json.dumps(e,ensure_ascii=False) for e in entries)
    # Keep all previous entries. Our two final overrides replace their full config.
    return original.rstrip()+'\n\n'+BEGIN+'\n'+block+'\n'+END+'\n'


def atomic(path, data):
    fd,name=tempfile.mkstemp(prefix='.operator-install-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def private(path,data):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data)


def check_python(python):
    for name in ('operator-agent','operator-agent-mcp'):
        if not (python.parent/name).is_file():raise ValueError('Python environment lacks '+name)
    env={k:v for k,v in os.environ.items() if k in ('HOME','PATH','LANG','SYSTEMROOT')}
    env['OPERATOR_AGENT_WORKFLOW']='host'
    requests=[dict(jsonrpc='2.0',id=1,method='initialize',params=dict(protocolVersion='2025-11-25',
        capabilities={},clientInfo=dict(name='operator-installer',version='1'))),
        dict(jsonrpc='2.0',id=2,method='tools/list',params={})]
    result=subprocess.run([str(python),'-I','-m','operator_agent.app.mcp_server'],
        input=''.join(json.dumps(x)+'\n' for x in requests),capture_output=True,text=True,
        env=env,cwd=tempfile.gettempdir(),timeout=20,check=True)
    replies=[json.loads(x) for x in result.stdout.splitlines()]
    names=sorted(x['name'] for x in next(x['result']['tools'] for x in replies if x.get('id')==2))
    if names!=TOOLS:raise ValueError('Python does not expose the seven v0.7.2 host tools')
    return next(x['result']['serverInfo']['version'] for x in replies if x.get('id')==1)


def install(args):
    python=Path(os.path.abspath(Path(args.python).expanduser())) # retain venv symlink
    plugin=Path(args.plugin).expanduser().resolve()
    dsh=Path(args.dsh_bin).expanduser().resolve()
    node=shutil.which('node')
    if not node or not dsh.is_file():raise ValueError('Node or the specified DSH CLI is unavailable')
    if not plugin.is_file() or plugin.suffix!='.tgz':raise ValueError('supply an existing local plugin .tgz')
    artifact=archive_identity(plugin)
    if not {'host-model-budget.mjs','host-model-ledger.mjs','host-session-lifecycle.mjs'} <= artifact['files'].keys():
        raise ValueError('plugin archive lacks the v0.7.2 host model budget modules')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.profile):raise ValueError('invalid profile name')
    version=check_python(python)
    home=Path(args.dsh_home or os.environ.get('DSH_HOME',str(Path.home()/'.dsh'))).expanduser().absolute()
    profile=home/'profiles'/args.profile;patch=profile/'cordis.patch.yml'
    entries=configs(python)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,python_version=version,plugin=str(plugin),
            profile=str(profile),plugin_sha256=artifact['sha256'],entries=entries),ensure_ascii=False,indent=2))
        return
    configure_only=getattr(args,'configure_only',False)
    if args.profile.lower()=='desktop' and not configure_only:
        raise ValueError('Desktop is managed by Electron: install the local archive in DSH Plugins, '
                         'then rerun with --configure-only. Quitting Desktop does not lift this CLI restriction.')
    if configure_only:
        installed=profile_state(profile,artifact)
        if not all(installed[k] for k in ('dependency','bundle_registered','package_installed','installed_files_match')):
            raise ValueError('Install this exact plugin archive in the target DSH profile before configuring it')
    receipts=home/'operator-install-receipts';receipts.mkdir(parents=True,exist_ok=True,mode=0o700)
    root=Path(tempfile.mkdtemp(prefix=args.profile+'-',dir=receipts))
    saved={}
    for name in ('cordis.patch.yml','package.json'):
        path=profile/name
        if path.is_symlink():raise ValueError('profile file must not be a symlink: '+name)
        saved[name]=path.read_bytes() if path.exists() else None
        if saved[name] is not None:private(root/(name+'.before'),saved[name])
    env=dict(os.environ,DSH_HOME=str(home))
    def call(argv,label):
        result=subprocess.run([node,str(dsh),*argv],env=env,capture_output=True,timeout=180)
        private(root/(label+'.stdout'),result.stdout);private(root/(label+'.stderr'),result.stderr)
        if result.returncode:
            raise RuntimeError(label+' failed; see '+str(root/(label+'.stderr'))+
                               ' (desktop profile requires DSH Desktop to be fully closed)')
        return result.stdout
    modified=False
    try:
        if not configure_only:
            call(['plugin','--profile',args.profile,'add',str(plugin),'--offline'],'plugin-add')
        current=patch.read_text() if patch.exists() else ''
        content=patch_text(current,entries).encode()
        private(root/'cordis.patch.yml.proposed',content)
        atomic(patch,content);modified=True
        # Desktop's CLI also forbids read-only dumps. Validate its overlay with
        # DSH's YAML parser, then require a live /op capabilities check in the UI.
        # Custom profiles additionally validate the fully composed configuration.
        if args.profile.lower()=='desktop':
            data=patch.read_bytes()
            check_scope='desktop_overlay_and_python_tools; live_ui_check_required'
        else:
            result=subprocess.run([node,str(dsh),'--profile',args.profile,'--dump-config'],
                env=env,capture_output=True,timeout=60)
            if result.returncode:
                private(root/'config-check.stderr',result.stderr)
                raise RuntimeError('DSH config check failed; see '+str(root/'config-check.stderr'))
            data=result.stdout
            check_scope='composed_profile_and_python_tools'
        parser='''import{createRequire}from'node:module';
const yaml=createRequire(process.argv[1])('js-yaml');
const schema=yaml.DEFAULT_SCHEMA.extend([new yaml.Type('tag:yaml.org,2002:js',{kind:'scalar',construct:x=>x})]);
let input='';for await(const c of process.stdin)input+=c;
const entries=yaml.load(input,{schema});
console.log(JSON.stringify(entries.filter(x=>['mcp-operator','operator-tools'].includes(x.id))));'''
        boot=dsh.parents[2]/'dsh-app-boot/package.json'
        inspected=subprocess.run([node,'--input-type=module','-e',parser,str(boot)],
            input=data,capture_output=True,timeout=15,check=True)
        actual={x['id']:x for x in json.loads(inspected.stdout)}
        for e in entries:
            a=actual.get(e['id'],{})
            if a.get('disabled') is True or a.get('config')!=e['config']:
                raise ValueError('DSH composition differs for '+e['id'])
        receipt=dict(installed=True,profile=str(profile),python_version=version,
            plugin=str(plugin),plugin_sha256=artifact['sha256'],tools=TOOLS,config_backup=str(root),model_calls=0,gpu_executions=0,
            config_check_scope=check_scope,
            completed_at=datetime.now(timezone.utc).isoformat())
        private(root/'result.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
        print('安装与路径配置检查通过。在 DSH 执行 /op capabilities 核对版本；直接发送新任务需求即可试跑。')
        print('配置：'+str(patch));print('备份和回执：'+str(root))
    except Exception:
        if modified:
            if saved['cordis.patch.yml'] is None:patch.unlink(missing_ok=True)
            else:atomic(patch,saved['cordis.patch.yml'])
        print('安装未完成；原配置备份：'+str(root))
        print('如已完成 plugin add，插件包可能仍已安装；仅配置文件自动恢复，不声称完整卸载回退。')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',required=True,help='installed Python environment executable')
    parser.add_argument('--plugin',required=True,help='local dsh-operator-agent .tgz')
    parser.add_argument('--profile',default='desktop')
    parser.add_argument('--dsh-home')
    parser.add_argument('--dsh-bin',default=str(DEFAULT_DSH))
    parser.add_argument('--dry-run',action='store_true',help='check prerequisites and print config; no profile writes')
    parser.add_argument('--configure-only',action='store_true',help='configure an exact archive already installed through DSH (required for desktop)')
    args=parser.parse_args()
    try:install(args)
    except Exception as error:
        print('失败：'+type(error).__name__+': '+str(error));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
