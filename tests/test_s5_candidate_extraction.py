"""Shared A/C final delivery rules; no execution of candidate source."""
import pytest

from benchmarks.s5_candidate_extraction import FinalSourceError, extract_final_source


SOURCE = 'def kernel(X):\n    return X\n'


@pytest.mark.parametrize('source', [SOURCE, SOURCE.rstrip('\n'), SOURCE.replace('\n', '\r\n')])
def test_pure_source_preserves_all_bytes_and_terminal_newline(source):
    assert extract_final_source(source) == (source, 'plain_python')


@pytest.mark.parametrize('language', ['python', 'py', 'Python', 'PY'])
def test_one_explicit_python_block_accepts_prose_without_selecting_history(language):
    text = 'Both checks passed. Final source:\n\n```' + language + '\n' + SOURCE + '```\nDone.'
    assert extract_final_source(text) == (SOURCE, 'single_python_code_block')


def test_fenced_crlf_is_not_rewritten_and_final_fence_needs_no_newline():
    source = SOURCE.replace('\n', '\r\n')
    assert extract_final_source('```python\r\n' + source + '```') == (
        source, 'single_python_code_block')


def test_plain_python_string_containing_fence_lines_stays_plain_source():
    source = '"""Literal Markdown:\n```python\nexample\n```\n"""\n' + SOURCE
    assert extract_final_source(source) == (source, 'plain_python')


def test_inline_backticks_inside_fenced_python_string_are_source():
    source = 'MARKER = "```python"\n' + SOURCE
    assert extract_final_source('```python\n' + source + '```') == (
        source, 'single_python_code_block')


def test_import_and_string_backticks_do_not_become_markdown_wrappers():
    source = 'import importlib\nMODULE_NAME = "```python"\n' + SOURCE
    assert extract_final_source(source) == (source, 'plain_python')


@pytest.mark.parametrize('binding', [
    'kernel = lambda X: X\n',
    'def candidate(X):\n    return X\nkernel = candidate\n',
    'kernel: object = lambda X: X\n',
])
def test_visible_kernel_alias_is_not_a_reason_to_discard_valid_source(binding):
    assert extract_final_source(binding) == (binding, 'plain_python')


@pytest.mark.parametrize('wrap', [False, True])
def test_duplicate_top_level_kernels_are_not_selected_as_one_final_candidate(wrap):
    source = SOURCE + '\ndef kernel(X):\n    return X + 1\n'
    text = ('Both alternatives passed; final sources:\n```python\n' + source + '```'
            if wrap else source)
    with pytest.raises(FinalSourceError):
        extract_final_source(text)


def test_distinct_triton_helpers_and_one_kernel_wrapper_are_one_candidate():
    source = ('import triton\nimport triton.language as tl\n'
              '@triton.jit\ndef _load_tile(X):\n    return tl.load(X)\n'
              '@triton.jit\ndef _store_tile(Y, values):\n    tl.store(Y, values)\n'
              + SOURCE)
    assert extract_final_source(source) == (source, 'plain_python')


def test_rebinding_one_kernel_to_its_compiled_wrapper_is_not_a_second_definition():
    source = 'import torch\n' + SOURCE + 'kernel = torch.compile(kernel)\n'
    assert extract_final_source(source) == (source, 'plain_python')


@pytest.mark.parametrize('text', [
    None, 12, '', ' \r\n', 'Done', 'Final source follows.',
    'import torch\n',
    'Done\n' + SOURCE,
    SOURCE + '\n"checks passed"\n',
    'def kernel(X)\n    return X\n',
    'async def kernel(X):\n    return X\n',
    '```\n' + SOURCE + '```',
    '```javascript\n' + SOURCE + '```',
    '~~~python\n' + SOURCE + '~~~',
    '````python\n' + SOURCE + '````',
    '```python\n' + SOURCE,
    '```python\n```',
    '```python\n' + SOURCE + '```python',
    '```python\n' + SOURCE + '```\n```python\n' + SOURCE + '```',
    '```json\n{}\n```\n```python\n' + SOURCE + '```',
    '```python\nMARKER = """\n```\n"""\n' + SOURCE + '```',
])
def test_ambiguous_missing_mixed_or_invalid_delivery_is_refused(text):
    with pytest.raises(FinalSourceError):
        extract_final_source(text)


def test_source_is_never_executed_by_extraction(tmp_path):
    marker = tmp_path / 'must-not-exist'
    source = 'from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n' + SOURCE
    assert extract_final_source(source) == (source, 'plain_python')
    assert not marker.exists()
