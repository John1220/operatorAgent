"""Explicit installed-artifact validation, never substitutes a source import."""
import json
import os
from pathlib import Path

import pytest

from benchmarks.delivery.installed_smoke import run


@pytest.mark.skipif(not (os.environ.get('OPERATOR_AGENT_DELIVERY_PYTHON') and
                         os.environ.get('OPERATOR_AGENT_DELIVERY_PLUGIN')),
                    reason='requires explicitly supplied clean Python 3.12 venv and extracted plugin artifact')
def test_clean_wheel_and_packed_plugin_mcp_workflow(tmp_path):
    root=tmp_path/'installed'
    result=run(os.environ['OPERATOR_AGENT_DELIVERY_PYTHON'],os.environ['OPERATOR_AGENT_DELIVERY_PLUGIN'],
               root,dsh=os.environ.get('OPERATOR_AGENT_DELIVERY_DSH')=='1')
    assert result['passed'] is True
    assert result['test_only'] is True and result['gpu_acceptance'] is False
    assert result['paid_model_calls']==result['ssh_executions']==0
    provenance=json.loads((root/'provenance.json').read_text())
    assert provenance['python'].startswith('3.12.') and 'site-packages' in Path(provenance['module']).parts
    records=json.loads((root/'mcp-transcript.json').read_text())
    names={r['request'].get('params',{}).get('name') for r in records}
    assert {'operator_prepare','operator_verify','operator_run','operator_status','operator_cancel'}<=names
    manifest=json.loads((root/'export/manifest.json').read_text())
    assert manifest['product_export_command'] is False
    assert {r['file'] for r in manifest['files']}=={'code.py','report.md'}
    assert all(r['sha256']==r['source_sha256'] for r in manifest['files'])
    assert result['cancel_status']['cleanup']['worker_reaped'] is True
    if os.environ.get('OPERATOR_AGENT_DELIVERY_DSH')=='1':
        assert result['dsh']['passed'] is True
        assert result['dsh']['outer_calls']>=3
