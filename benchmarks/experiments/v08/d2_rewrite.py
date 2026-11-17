"""D2 参数层：冻结起点核的 launch 参数字面量机械注入（零模型调用）。

只改 wrapper 里 launch 调用的 BLOCK_SIZE / num_warps 两个数，核体（@triton.jit 函数）
一个字节不动。AST 定位 Constant 的精确 span 做替换，其余文本逐字保留。模式不符即拒
（fail-closed）：不猜第二个 launch、不猜非常量、不猜多处赋值——改写器宁可炸，也不许
把「没找到旋钮」读成「改好了」。

起点核（cb851b67…）的实际形状：wrapper 里 `BLOCK_SIZE = 512` 赋值后经
`BLOCK_SIZE=BLOCK_SIZE` 传入 launch，`num_warps=4` 是 launch 关键字字面量。
两条解析路径都覆盖，但每处只认唯一常量。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass


class RewriteError(ValueError):
    """源码不符合可机械改写的冻结模式；拒绝改写。"""


@dataclass(frozen=True)
class LaunchKnobs:
    block_size: int
    num_warps: int


def _launch_calls(tree):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)]


def _kernel_func(tree):
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == 'kernel']
    if len(funcs) != 1:
        raise RewriteError('源码必须恰有一个顶层 kernel wrapper')
    return funcs[0]


def _constant(node):
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
        return node
    return None


def _resolve_assignment(func, name):
    hits = []
    for node in func.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            hits.append(node)
    if len(hits) != 1:
        raise RewriteError(f'kernel 内 {name} 必须恰有一处赋值，实际 {len(hits)} 处')
    constant = _constant(hits[0].value)
    if constant is None:
        raise RewriteError(f'{name} 的赋值不是正整数字面量，拒绝猜值')
    return constant


def resolve_launch(source: str) -> LaunchKnobs:
    """读源码当前 launch 参数（不改写）；模式不符即拒。"""
    tree = ast.parse(source)
    calls = _launch_calls(tree)
    if len(calls) != 1:
        raise RewriteError(f'源码必须恰有一个 Triton launch 调用，实际 {len(calls)} 个')
    func = _kernel_func(tree)
    keywords = {}
    for kw in calls[0].keywords:
        if kw.arg in ('BLOCK_SIZE', 'num_warps'):
            if kw.arg in keywords:
                raise RewriteError(f'launch 关键字 {kw.arg} 重复')
            keywords[kw.arg] = kw.value
    if set(keywords) != {'BLOCK_SIZE', 'num_warps'}:
        raise RewriteError('launch 必须同时带 BLOCK_SIZE 与 num_warps 关键字')
    values = {}
    for name, node in keywords.items():
        constant = _constant(node)
        if constant is None:
            if not isinstance(node, ast.Name):
                raise RewriteError(f'{name} 既不是字面量也不是可解析的变量名')
            constant = _resolve_assignment(func, node.id)
        values[name] = constant
    return LaunchKnobs(block_size=values['BLOCK_SIZE'].value,
                       num_warps=values['num_warps'].value)


def _jit_bodies(tree):
    return sorted(ast.dump(node) for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and any(isinstance(d, ast.Attribute) and d.attr == 'jit'
                          for d in node.decorator_list))


def _splice(source: str, edits):
    lines = source.splitlines(keepends=True)
    for node, text in sorted(edits, key=lambda item: (item[0].lineno, item[0].col_offset),
                             reverse=True):
        if node.end_lineno != node.lineno:
            raise RewriteError('目标字面量跨行，冻结模式不支持')
        line = lines[node.lineno - 1]
        lines[node.lineno - 1] = line[:node.col_offset] + text + line[node.end_col_offset:]
    return ''.join(lines)


def rewrite(source: str, *, block_size: int, num_warps: int) -> str:
    """把 launch 参数改成目标值；其余字节逐字不动，核体 AST 不变。"""
    for name, value in (('block_size', block_size), ('num_warps', num_warps)):
        if type(value) is not int or value <= 0:
            raise RewriteError(f'{name} 必须是正整数，实际 {value!r}')
    if block_size & (block_size - 1):
        raise RewriteError('BLOCK_SIZE 必须是 2 的幂（tl.arange 约束）')
    if num_warps & (num_warps - 1):
        raise RewriteError('num_warps 必须是 2 的幂')
    before = ast.parse(source)
    calls = _launch_calls(before)
    if len(calls) != 1:
        raise RewriteError(f'源码必须恰有一个 Triton launch 调用，实际 {len(calls)} 个')
    func = _kernel_func(before)
    edits = []
    for kw in calls[0].keywords:
        if kw.arg == 'BLOCK_SIZE':
            node = _constant(kw.value)
            if node is None:
                if not isinstance(kw.value, ast.Name):
                    raise RewriteError('BLOCK_SIZE 既不是字面量也不是可解析的变量名')
                node = _resolve_assignment(func, kw.value.id)
            edits.append((node, str(block_size)))
        elif kw.arg == 'num_warps':
            node = _constant(kw.value)
            if node is None:
                raise RewriteError('num_warps 不是字面量，拒绝猜值')
            edits.append((node, str(num_warps)))
    if len(edits) != 2:
        raise RewriteError('launch 必须同时带 BLOCK_SIZE 与 num_warps 关键字')
    after_source = _splice(source, edits)
    after = ast.parse(after_source)
    if resolve_launch(after_source) != LaunchKnobs(block_size=block_size, num_warps=num_warps):
        raise RewriteError('改写后回读与目标值不一致')
    if _jit_bodies(before) != _jit_bodies(after):
        raise RewriteError('改写触碰了 @triton.jit 核体')
    old_lines, new_lines = source.splitlines(), after_source.splitlines()
    if len(old_lines) != len(new_lines):
        raise RewriteError('改写改变了行数')
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    if len(changed) > 2:
        raise RewriteError('改写差分超过两处字面量')
    for i in changed:
        if ''.join(ch for ch in old_lines[i] if not ch.isdigit()) != \
                ''.join(ch for ch in new_lines[i] if not ch.isdigit()):
            raise RewriteError('改写差分不只是数字替换')
    return after_source


def grid_points(block_sizes, num_warps):
    """确定性枚举：BLOCK_SIZE 外轴、num_warps 内轴（轴序即枚举序）。"""
    return [dict(block_size=bs, num_warps=nw) for bs in block_sizes for nw in num_warps]