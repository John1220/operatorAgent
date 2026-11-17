"""A/C 工作流轮的驱动件：**先证明臂面与终评这两环，再谈跑格**。

这个文件现在只承担 P0-4 的两个可验证半环，不假装自己是完整驱动：

1. ``surface_check`` —— 这一臂**实际加载的 bundle**是什么。归档里出过一次事故：版本不齐时
   bundle 会**静默 skip**，而 prompt 还在按"你有插件"写。所以先 `--dump-config` 看真东西：
   A 臂必须 `tool-bash`/`tool-fs` 在且没被 `disabled`，且**不能**出现 `mcp-operator`/`operator-tools`；
   C 臂反过来。
2. ``final_evaluation`` —— A 臂没有插件，它的"独立终评"只能由 supervisor 做：
   先 `prepare(..., host_budget=…)` 建 host task，再用**同一份源码**跑一次完整 L2
   （`final` 的前置：`host_final_source` 拒绝没有 L2 通过的源码），最后
   `verify(purpose='final')`。三步缺一步就只剩一句主张。

``cell(...)`` 把结果拼成 `ac_compare` 读的那一格（`task_id/arm/state/final_receipt/source/…`）。
真正的两臂会话编排留到下一步：它要起 DSH 子进程、喂 prompt、回收提交源码，属于另一拍。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

CELL_FIELDS = ('task_id', 'arm', 'state', 'final_receipt', 'source',
               'wall_clock_to_first_full_L2_s', 'wall_clock_total_s',
               'model_dispatches', 'verifications', 'stop_reason',
               'qualified', 'delivery_class', 'fusion_static',
               'final_speedup', 'final_validity')


class SurfaceError(ValueError):
    """臂面与计划声明不一致：这一格不能开，开了也判不了。"""


def _dump(arm, built, *, node, entry, timeout=180):
    patch = Path(built['home']).parent / 'patch.json'
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(json.dumps(built['patch'], ensure_ascii=False), encoding='utf-8')
    proc = subprocess.run([str(node), str(entry), '--profile', 'headless',
                           '--patch', str(patch), '--dump-config'],
                          cwd=str(built['work']), env=built['env'],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise SurfaceError(f'{arm}: --dump-config 失败 rc={proc.returncode}: '
                           f'{(proc.stderr or "")[-400:]}')
    return proc.stdout


def _entries(text):
    """(id, disabled, name) —— 只看 `- id:` 行与紧跟的下一行。"""
    rows, lines = {}, text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('- id:'):
            continue
        ident = stripped.split(':', 1)[1].strip()
        nxt = lines[index + 1].strip() if index + 1 < len(lines) else ''
        rows[ident] = dict(disabled=nxt.startswith('disabled:'),
                          name=nxt.split(':', 1)[1].strip() if nxt.startswith('name:') else nxt)
    return rows


def surface_check(arm, *, runtime, builder, node, entry, base_url='http://127.0.0.1:9/v1',
                  model='fixture-model', arm_dir=None, relay=None, development=None,
                  max_tokens=1024, context_window=65536):
    """构建臂面、dump 实际配置、按臂核对。返回事实，不合法就抛 `SurfaceError`。"""
    if arm not in ('A', 'C'):
        raise SurfaceError(f'unknown arm {arm!r}')
    if arm_dir is None:
        import tempfile
        arm_dir = Path(tempfile.mkdtemp(prefix=f'ac-surface-{arm}-')) / 'arm'
    kwargs = dict(runtime=runtime, base_url=base_url, model=model, tools=True,
                  max_tokens=max_tokens, context_window=context_window, arm_dir=arm_dir)
    if arm == 'C':
        if relay is None and development is None:
            raise SurfaceError('C 臂需要 MCP relay 或 development 配置')
        kwargs['relay'] = relay
        if development is not None:
            kwargs['development'] = development
    built = builder(arm, **kwargs)
    text = _dump(arm, built, node=node, entry=entry)
    rows = _entries(text)
    tools = {name: (name in rows and not rows[name]['disabled'])
             for name in ('tool-bash', 'tool-fs', 'tool-fs-search')}
    operator = sorted(name for name in rows if name in ('mcp-operator', 'operator-tools'))
    layers = [line.strip() for line in text.splitlines() if line.startswith('# ==')]
    facts = dict(arm=arm, layers=layers, tools=tools,
                 operator_entries=operator, patch_sha256=built.get('patch_sha256'),
                 surface=built.get('surface'))
    if arm == 'A':
        missing = [name for name, enabled in tools.items() if not enabled]
        if missing:
            raise SurfaceError(f'A 臂缺工具（或工具被 disabled）: {missing} —— 无插件臂靠 shell/fs 干活')
        if operator:
            raise SurfaceError(f'A 臂不该有插件入口，实际出现: {operator}')
    else:
        for name in ('mcp-operator', 'operator-tools'):
            if name not in operator:
                raise SurfaceError(f'C 臂缺插件入口 {name!r}：prompt 会按"有插件"写，实际没有')
        missing = [name for name, enabled in tools.items() if not enabled]
        if missing:
            raise SurfaceError(f'C 臂的工具被关掉了: {missing}（宿主仍要能读项目、写文件）')
    return facts


def final_evaluation(run_dir, code, *, declaration, host_budget, backend, stop_after='L2',
                     host=None, port=None, user=None, key_path=None, work_dir=None,
                     timeout=300, persistent=False):
    """A 臂的独立终评：prepare（含终评预留）→ 同源码完整 L2 → `purpose=final`。

    顺序不是风格问题：`host_final_source` 会拒绝"没有 L2 通过的源码"，
    所以"直接 final"这条路在机制上不存在。
    """
    from operator_agent.app import hitl_tools as ht

    run = Path(run_dir)
    made = False
    if not (run / 'contract.json').is_file():
        ht.prepare(declaration, run, host_budget=dict(host_budget))
        made = True
    kwargs = dict(backend=backend, stop_after=stop_after, host=host, port=port, user=user,
                  key_path=key_path, work_dir=work_dir, timeout=timeout,
                  persistent=persistent)
    explore = ht.verify(run, code, **kwargs)
    if not (explore.get('levels') or {}).get('L2'):
        return dict(prepared=made, delivered=False,
                    explore=explore, final=None,
                    reason='prior full L2 did not pass: a final evaluation would be refused '
                           '(host_final_source) and this arm did not deliver')
    final = ht.verify(run, code, verification_purpose='final', **kwargs)
    return dict(prepared=made, delivered=bool((final.get('levels') or {}).get('L2')),
                explore=explore, final=final)


def cell(task_id, arm, *, state, final=None, source=None, **facts):
    """拼 `ac_compare.cell_summary` 读的那一格。字段名只有一处来源（CELL_FIELDS）。"""
    if state not in ('completed', 'budget_exhausted', 'deadline', 'crashed', 'not_executed'):
        raise SurfaceError(f'unknown cell state {state!r}')
    body = dict(task_id=task_id, arm=arm, state=state,
                final_receipt=final.get('final') if isinstance(final, dict) else final,
                source=str(source) if source is not None else None,
                wall_clock_to_first_full_L2_s=None, model_dispatches=None,
                verifications=None, stop_reason=None)
    body.update({key: value for key, value in facts.items() if key in CELL_FIELDS})
    unknown = set(facts) - set(CELL_FIELDS)
    if unknown:
        raise SurfaceError(f'unknown cell fields: {sorted(unknown)}')
    return body


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog='ac_driver', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)

    surf = sub.add_parser('surface', help='dump 这一臂实际加载的 bundle 并按臂核对')
    surf.add_argument('--arm', required=True, choices=['A', 'C'])
    surf.add_argument('--json-out')

    fin = sub.add_parser('final', help='supervisor 侧独立终评（prepare → L2 → final）')
    fin.add_argument('--run-dir', required=True)
    fin.add_argument('--code-file', required=True)
    fin.add_argument('--backend', default='fake', choices=['mock', 'local', 'ssh', 'fake'])
    fin.add_argument('--host')
    fin.add_argument('--work-dir')
    fin.add_argument('--json-out')

    args = parser.parse_args(argv)
    if args.action == 'surface':
        import tempfile

        from benchmarks.experiments.s5 import s5_plugin_env
        from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE

        class _RelayStub:
            """只为 dump：C 臂的 patch 需要一份 MCP client 配置，但这里不启动 relay。"""
            def client_config(self):
                return dict(serverName='operator', transport='stdio',
                            command='/usr/bin/false', args=[], toolCallTimeoutMs=60000)

        runtime = s5_plugin_env.stage_runtime(Path(tempfile.mkdtemp(prefix='ac-surface-')) / 'runtime')
        extra = dict(relay=_RelayStub(),
                     development=dict(url='http://127.0.0.1:9/development/verify',
                                      token='stub-token', contract='0' * 64,
                                      backend='mock', timeout=30)) if args.arm == 'C' else {}
        facts = surface_check(args.arm, runtime=runtime, builder=s5_plugin_env.build,
                              node=NODE, entry=Path(DSH_ROOT) / 'dsh' / 'lib' / 'bin.js', **extra)
        payload = json.dumps(facts, ensure_ascii=False, indent=2)
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(payload + '\n', encoding='utf-8')
        print(payload)
        return 0
    raise SystemExit('只有 surface 有 CLI；final 由驱动调用（它需要声明与预算）')

if __name__ == '__main__':
    raise SystemExit(main())
