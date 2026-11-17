"""Static fusion check for the A/C ``residual add + LayerNorm`` task.

The core verifier judges correctness and speed; it cannot tell whether the timed
path is one fused kernel or two Triton kernels chained inside a wrapper.  For
this round's "qualified delivery" we additionally require:

1. exactly **one** ``@triton.jit`` kernel in the submitted source;
2. the wrapper ``kernel()`` (or the module) reaches that kernel through exactly
   **one** call site.

A two-kernel wrapper (separate ``add`` and ``layernorm``) fails by construction.
This is a plan-level check, deliberately kept out of the core verifier: it is a
research gate for one task, not a new global correctness rule.  The dynamic
backstop is the frozen ``min_speedup=1.5`` threshold (two-kernel wrappers land
near 1.0x against the eager baseline).
"""
from __future__ import annotations

import ast

FUSION_CHECK_SCHEMA = 'v08-fusion-static-v1'


class FusionCheckError(ValueError):
    """The source cannot be parsed as a Python/Triton candidate."""


def _decorator_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Name):
            return f'{base.id}.{node.attr}'
        if isinstance(base, ast.Attribute):
            inner = _decorator_name(base.value)
            return f'{inner}.{node.attr}' if inner else node.attr
    return ''


def _is_triton_jit(node):
    for decorator in getattr(node, 'decorator_list', ()) or ():
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = _decorator_name(target)
        if name == 'triton.jit' or name.endswith('.jit'):
            return True
    return False


def _callee_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        # Triton launch syntax: ``_kernel[grid](...)`` -> callee is the subscript value.
        return _callee_name(node.value)
    return ''


def check_fused_source(source, *, entry='kernel'):
    """Return the static fusion reading; never raises for a well-typed str."""
    if not isinstance(source, str) or not source.strip():
        raise FusionCheckError('source must be a non-empty string')
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise FusionCheckError(f'source is not parseable Python: {exc.msg}') from exc

    jit = [node for node in tree.body
           if isinstance(node, ast.FunctionDef) and _is_triton_jit(node)]
    jit_names = [node.name for node in jit]
    wrappers = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == entry]
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node.func) in jit_names:
            calls.append(getattr(node, 'lineno', None))
    result = dict(schema=FUSION_CHECK_SCHEMA, jit_kernels=jit_names,
                  wrapper_present=bool(wrappers), launch_sites=len(calls),
                  passed=False, reason='')
    if len(jit_names) != 1:
        result['reason'] = f'expected exactly one @triton.jit kernel, found {len(jit_names)}'
    elif not wrappers:
        result['reason'] = f"missing entry wrapper def {entry}(...)"
    elif len(calls) != 1:
        result['reason'] = (f'expected exactly one launch of {jit_names[0]}, '
                            f'found {len(calls)} call site(s)')
    else:
        result['passed'] = True
        result['reason'] = 'single triton kernel with one launch site'
    return result


def require_fused(source, *, entry='kernel'):
    """Raise ``FusionCheckError`` unless the static fusion requirement holds."""
    result = check_fused_source(source, entry=entry)
    if not result['passed']:
        raise FusionCheckError(result['reason'])
    return result
