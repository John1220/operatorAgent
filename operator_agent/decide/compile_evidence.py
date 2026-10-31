"""Bounded, typed provenance for one native Triton JIT compilation failure.

This shared module deliberately never reads exception prose or executes source.
Locations are full-source Python AST coordinates: one-based lines and zero-based
UTF-8 byte columns. Missing or ambiguous provenance stays unknown.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re


SCHEMA = "candidate-jit-compile-failure-v1"
PRODUCER = "triton.compiler.errors.CompilationError"
PHASE = "candidate_launch"
_FIELDS = {"schema", "producer", "phase", "source_hash", "helper_name", "line",
           "column", "end_line", "end_column", "node_ast_hash", "evidence_hash"}
_LOCATION = ("lineno", "col_offset", "end_lineno", "end_col_offset")


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal(payload):
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False))


def _parse(source):
    if type(source) is not str or not source or len(source.encode("utf-8")) > 32768:
        return None
    try:
        tree = ast.parse(source)
        pending, count = [(tree, 0)], 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 4096 or depth > 32:
                return None
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        return tree
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError):
        return None


def _jit(node):
    return (isinstance(node, ast.FunctionDef) and len(node.decorator_list) == 1
            and isinstance(node.decorator_list[0], ast.Attribute)
            and isinstance(node.decorator_list[0].value, ast.Name)
            and node.decorator_list[0].value.id == "triton"
            and node.decorator_list[0].attr == "jit")


def _function_key(node):
    # Work only on a fresh parser tree, never on the compiler's AST object.
    decorators = node.decorator_list
    node.decorator_list = []
    try:
        return ast.dump(node, include_attributes=False)
    finally:
        node.decorator_list = decorators


def _paths(node, path=()):
    """The same AST-field path maps locations despite decorator/indent changes."""
    yield path, node
    for field, value in ast.iter_fields(node):
        if field == "decorator_list":
            continue
        if isinstance(value, ast.AST):
            yield from _paths(value, path + (field,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield from _paths(child, path + (field, index))


def _location(node):
    values = tuple(getattr(node, name, None) for name in _LOCATION)
    if (any(type(value) is not int for value in values)
            or values[0] < 1 or values[1] < 0 or values[2] < values[0]
            or values[3] < 0 or (values[0] == values[2] and values[3] <= values[1])):
        return None
    return values


def _node_hash(node):
    return _sha(ast.dump(node, include_attributes=False))


def _helpers(tree, name):
    # Duplicate bindings cannot identify the function that was actually launched.
    named = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name == name]
    return named if len(named) == 1 and _jit(named[0]) else []


def capture_compile_failure(error, code, phase):
    """Capture native compiler AST provenance only during the candidate call."""
    if type(phase) is not str or phase != PHASE:
        return None
    try:
        from triton.compiler.errors import CompilationError
    except ImportError:
        return None
    if type(error) is not CompilationError:
        return None
    try:
        source_tree = _parse(code)
        attributes = vars(error)
        fragment_tree = _parse(attributes.get("src"))
        failed_node = attributes.get("node")
        if (source_tree is None or fragment_tree is None
                or len(fragment_tree.body) != 1
                or not isinstance(fragment_tree.body[0], ast.FunctionDef)
                or not isinstance(failed_node, ast.AST)):
            return None
        fragment = fragment_tree.body[0]
        if fragment.decorator_list and not _jit(fragment):
            return None
        helpers = _helpers(source_tree, fragment.name)
        if len(helpers) != 1 or _function_key(helpers[0]) != _function_key(fragment):
            return None
        # Inspect only a bounded compiler node. Its AST must be exactly the node
        # at the compiler's typed fragment coordinates, not a text guess.
        failed_location = _location(failed_node)
        if failed_location is None:
            return None
        pending, count = [(failed_node, 0)], 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 4096 or depth > 32:
                return None
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        failed_hash = _node_hash(failed_node)
        matched = [(path, node) for path, node in _paths(fragment)
                   if type(node) is type(failed_node) and _location(node) == failed_location
                   and _node_hash(node) == failed_hash]
        if len(matched) != 1:
            return None
        full_node = dict(_paths(helpers[0]))[matched[0][0]]
        full_location = _location(full_node)
        if full_location is None:
            return None
        line, column, end_line, end_column = full_location
        payload = dict(schema=SCHEMA, producer=PRODUCER, phase=PHASE,
                       source_hash=_sha(code), helper_name=helpers[0].name,
                       line=line, column=column, end_line=end_line,
                       end_column=end_column, node_ast_hash=_node_hash(full_node))
        payload["evidence_hash"] = _seal(payload)
        return validate_compile_failure(payload, code)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError, UnicodeError):
        return None


def validate_compile_failure(payload, code):
    """Recheck a strict payload against the original full candidate source."""
    if type(payload) is not dict or set(payload) != _FIELDS:
        return None
    try:
        tree = _parse(code)
        if (tree is None or any(type(payload[key]) is not str for key in ("schema", "producer", "phase"))
                or payload["schema"] != SCHEMA or payload["producer"] != PRODUCER
                or payload["phase"] != PHASE or payload["source_hash"] != _sha(code)
                or type(payload["helper_name"]) is not str
                or not payload["helper_name"].isidentifier()
                or any(type(payload[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", payload[key])
                       for key in ("source_hash", "node_ast_hash", "evidence_hash"))
                or any(type(payload[key]) is not int
                       for key in ("line", "column", "end_line", "end_column"))):
            return None
        sealed = {key: value for key, value in payload.items() if key != "evidence_hash"}
        if payload["evidence_hash"] != _seal(sealed):
            return None
        helpers = _helpers(tree, payload["helper_name"])
        if len(helpers) != 1:
            return None
        location = tuple(payload[key] for key in ("line", "column", "end_line", "end_column"))
        matched = [node for _, node in _paths(helpers[0])
                   if _location(node) == location and _node_hash(node) == payload["node_ast_hash"]]
        return dict(payload) if len(matched) == 1 else None
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError, UnicodeError):
        return None
