"""Portable case projections do not copy untrusted free text or secrets."""
import hashlib
import json

import pytest

from benchmarks.s4_case_bundle import number, project_event
from benchmarks.s4_case_bundle_verify import sensitive


def test_event_projection_keeps_case_evidence_without_untrusted_free_text(tmp_path):
    original = dict(level='L2', passed=False, full_l2_passed=False,
        contract_hash='a'*64, source_hash='b'*64,
        timing=dict(completed_at=10, completed_monotonic=8),
        model_watermark=dict(reserved_calls=0,dispatched_calls=0,completed_calls=0),
        private_endpoint='https://example.invalid/private',
        verdicts=[dict(level='L2', passed=False, message='at ' + '/Users/' + 'private/path',
            metrics=dict(correctness=dict(verdict='failed',cases=[dict(
                case_id='case-1',shape_key='X=8x32',pattern='random',layout='strided',
                verdict='failed', reason='secret path /home/private',
                metrics=dict(bad_elements=256,compared_elements=256,max_error=4.7,
                    matched_ratio=0.0,input_fidelity_verified=True))])))])
    path=tmp_path/'event.json'; path.write_text(json.dumps(original))
    projection=project_event(path)
    assert projection['origin_evidence_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert projection['correctness']['cases'][0]['bad_elements']==256
    rendered=json.dumps(projection)
    assert 'example.invalid' not in rendered and '/Users/' not in rendered and '/home/' not in rendered


def test_projection_rejects_unsupported_level(tmp_path):
    data=dict(level='L4',passed=False,full_l2_passed=None,contract_hash='a'*64,
        source_hash='b'*64,timing=dict(completed_at=1,completed_monotonic=1),
        verdicts=[dict(level='L4',metrics={})])
    path=tmp_path/'event.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='unsupported'): project_event(path)


@pytest.mark.parametrize('text',[
    'https://example.invalid/private', '/Users/' + 'private/file', '/home/private/file',
    'algorithm-h20', 'Bearer abcdef', 'sk-' + '12345678901234567890',
    '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----',
])
def test_sensitive_detector_rejects_known_connection_and_secret_shapes(text):
    assert sensitive(text)


def test_nonfinite_and_boolean_values_are_not_numeric_evidence():
    assert number(float('nan')) is None
    assert number(float('inf')) is None
    assert number(True) is None
    assert number(12)==12
