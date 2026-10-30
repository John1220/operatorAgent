"""Extract the same final Python candidate for either S5 comparison arm.

This module does not search previous candidates or consult verification results.
The caller retains the raw answer before calling it. Extraction establishes an
unambiguous source artifact, not that its kernel is correct or safe to execute.
"""
from __future__ import annotations

import ast
import re


class FinalSourceError(ValueError):
    """The final answer does not contain one identifiable Python candidate."""


_FENCE_LINE = re.compile(r'^[ \t]*(`{3,}|~{3,})([^\r\n]*)$')


def _python_candidate(source):
    """Check syntax and a visible module-level kernel binding, without execution."""
    if not source.strip():
        raise FinalSourceError('empty final candidate')
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise FinalSourceError('final candidate is not valid Python syntax') from exc

    bound = False
    kernel_definitions = 0
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.FunctionDef):
            bound = bound or statement.name == 'kernel'
            kernel_definitions += statement.name == 'kernel'
        elif isinstance(statement, ast.AsyncFunctionDef):
            kernel_definitions += statement.name == 'kernel'
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            bound = bound or any(isinstance(target, ast.Name) and target.id == 'kernel'
                                 for target in targets)
        # A heading such as `Done` is legal Python, but is not source delivery.
        # Preserve a real initial docstring and ordinary top-level calls.
        elif isinstance(statement, ast.Expr):
            value = statement.value
            docstring = (index == 0 and isinstance(value, ast.Constant)
                         and isinstance(value.value, str))
            if not docstring and not isinstance(value, ast.Call):
                raise FinalSourceError('ambiguous top-level expression in final candidate')
    if kernel_definitions > 1:
        raise FinalSourceError('multiple top-level kernel definitions in final candidate')
    if not bound:
        raise FinalSourceError('final candidate has no visible kernel binding')


def extract_final_source(text):
    """Return ``(source, normalization)`` from pure Python or one Python fence.

    Pure Python is returned byte-for-byte (as UTF-8 text), including its line
    endings. A fenced answer may have surrounding prose, but must have exactly
    one pair of three-backtick fence lines labelled ``python`` or ``py``. Its
    content is returned exactly as supplied between those lines. In particular,
    no terminal newline is added or removed to match a previous candidate.

    Valid pure Python takes precedence over Markdown detection: a Python string
    containing backticks is source, not a second candidate. For fenced answers,
    additional fence lines, nested fences and alternative languages are refused
    rather than selected heuristically.
    """
    if not isinstance(text, str) or not text.strip():
        raise FinalSourceError('nonempty final answer text required')
    try:
        _python_candidate(text)
    except FinalSourceError:
        pass
    else:
        return text, 'plain_python'

    lines = text.splitlines(keepends=True)
    fences = []
    for index, line in enumerate(lines):
        match = _FENCE_LINE.fullmatch(line.rstrip('\r\n'))
        if match:
            fences.append((index, match.group(1), match.group(2).strip()))
    if len(fences) != 2:
        raise FinalSourceError('final answer must contain one Python code block')
    opening, closing = fences
    if (opening[1] != '```' or opening[2].lower() not in ('python', 'py')
            or closing[1] != '```' or closing[2]):
        raise FinalSourceError('ambiguous or unsupported final code fence')
    source = ''.join(lines[opening[0] + 1:closing[0]])
    _python_candidate(source)
    return source, 'single_python_code_block'
