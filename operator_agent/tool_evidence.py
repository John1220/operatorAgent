"""Write-once tool evidence. Storage completeness never grants candidate credit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from .run_contract import audit_value, code_hash, protocol_identity
from .run_layout import write_once_artifact


def verification_data(results):
    """Keep complete stage facts, including errors and unprojected metrics."""
    return [dict(level=result.level.value, passed=result.passed,
                 message=result.message, metrics=result.metrics,
                 error_log=result.error_log) for result in results]


def json_bytes(value):
    return (json.dumps(audit_value(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + '\n').encode('utf-8')


class RoundEvidence:
    """Reserve a unique numbered directory; interrupted reservations are not reused.

    Old flat JSON/log files remain readable and count toward allocation. mkdir
    arbitrates concurrent allocations; every artifact uses exclusive creation.
    A complete.json receipt is written last, after compatibility files too.
    This protects against accidental overwrite, not hostile same-UID mutation.
    """

    def __init__(self, run, kind, declaration, contract, *, tool_version,
                 stop_after=None, now=None):
        if kind not in ('verify', 'repair', 'bench', 'step'):
            raise ValueError('unknown evidence kind')
        self.run = Path(run).resolve()
        self.kind = kind
        pattern = re.compile(re.escape(kind) + r'-(\d+)(?:\..*)?$')
        while True:
            numbers = [int(match.group(1)) for path in self.run.iterdir()
                       if (match := pattern.fullmatch(path.name))]
            self.index = max(numbers, default=0) + 1
            self.path = self.run / f'{kind}-{self.index}'
            try:
                self.path.mkdir(mode=0o700)
                break
            except FileExistsError:
                continue
        self.complete = False
        self.created_at = time.time() if now is None else now
        self.contract = {'hash': contract.hash, 'contract': contract.as_dict()}
        self.declaration = declaration
        self.implementation = protocol_identity()
        self.binding = dict(tool_version=tool_version, contract_hash=contract.hash,
                            source_hash=None, stop_after=stop_after)
        self.write_json('started.json', dict(kind=kind, index=self.index,
            created_at=self.created_at, binding=self.binding))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not self.complete:
            try:
                self.write_json('incomplete.json', dict(evidence_complete=False,
                    error_type=exc_type.__name__, binding=self.binding))
            except (OSError, ValueError, TypeError):
                pass  # Never replace the original backend/storage exception.
        return False

    @staticmethod
    def _write_once(path, data):
        write_once_artifact(path, data)

    def write_bytes(self, name, data):
        if self.complete:
            raise ValueError('completed evidence cannot be extended')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe evidence member')
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_once(path, data)
        return str(path)

    def write_text(self, name, text):
        return self.write_bytes(name, text.encode('utf-8'))

    def write_json(self, name, value):
        return self.write_bytes(name, json_bytes(value))

    def source(self, name, code):
        path = self.write_text(name, code)
        return dict(source=path, source_hash=code_hash(code))

    def finish(self, full, *, legacy_payload=None, legacy_log=None,
               legacy_source=None):
        """Persist complete facts separately from the compact legacy/tool reply."""
        after = protocol_identity()
        payload = dict(full, schema='operator-tool-evidence-v1',
                       evidence_complete=True, created_at=self.created_at,
                       binding=dict(self.binding), contract=self.contract,
                       declaration=self.declaration, implementation=self.implementation,
                       implementation_stable=after == self.implementation,
                       implementation_after=after if after != self.implementation else None,
                       implementation_scope='On-disk hashes before/after; not proof of loaded module bytes.')
        self.write_json('result.json', payload)
        if legacy_payload is not None:
            self._write_once(self.run / f'{self.kind}-{self.index}.json',
                             json_bytes(dict(legacy_payload, created_at=self.created_at)))
        if legacy_log is not None:
            self._write_once(self.run / f'{self.kind}-{self.index}.log', legacy_log.encode('utf-8'))
        if legacy_source is not None:
            self._write_once(self.run / f'{self.kind}-{self.index}.py', legacy_source.encode('utf-8'))
        members = {str(path.relative_to(self.path)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in self.path.rglob('*') if path.is_file()}
        self.write_json('complete.json', dict(schema='operator-tool-evidence-complete-v1',
            files_sha256=members, binding=dict(self.binding), evidence_complete=True,
            scope='Storage integrity only; does not establish correctness or performance.'))
        self.complete = True


def persist_conductor_round(layout, task, code, results, round_num, *, admitted):
    """Retain post-gate round facts using the existing layout's declared writers.

    Refused executable source stays hash-only. Unique event names also retain
    earlier evidence when a caller resumes/reuses a run root or round number.
    No independent verification, confirmation or corpus admission occurs here.
    """
    if layout is None:
        return None
    from .contract import freeze_from_task
    from .report import build_report
    from .run_layout import ROLE_CANDIDATE, ROLE_ORACLE, ROLE_REPORT_WRITER
    from .schemas import KernelArtifact, VerificationLevel

    token = f'round-{round_num}-{uuid.uuid4().hex}'
    source = (layout.write_for(ROLE_CANDIDATE, f'candidate/{token}.py', code)
              if admitted else None)
    contract = freeze_from_task(task)
    l3 = next((r for r in results if r.level is VerificationLevel.L3_PERFORMANCE), None)
    report = layout.write_for(ROLE_REPORT_WRITER, f'report/{token}.md', build_report(
        artifact=KernelArtifact(code=code if admitted else '', verification_results=results,
                                benchmark_data=dict(l3.metrics or {}) if l3 else {}), task=task))
    payload = dict(schema='operator-conductor-round-evidence-v1', round=round_num,
        source=source, source_hash=code_hash(code),
        source_retention='full' if admitted else 'hash_only',
        contract={'hash': contract.hash, 'contract': contract.as_dict()},
        implementation=protocol_identity(), report=report,
        implementation_scope='Post-verification on-disk hashes; in-flight checks belong to run_binding.',
        levels={r.level.value: r.passed for r in results}, verdicts=verification_data(results),
        scope='Recorded round facts only; does not grant confirmation or corpus credit.')
    result = layout.write_for(ROLE_ORACLE, f'measurements/{token}.json', json_bytes(payload))
    paths = [path for path in (source, report, result) if path is not None]
    receipt = layout.write_for(ROLE_ORACLE, f'measurements/{token}.complete.json', json_bytes(
        dict(files_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
             source_hash=payload['source_hash'], contract_hash=contract.hash,
             evidence_complete=True, scope='Storage integrity, not candidate acceptance.')))
    return dict(source=source, source_hash=payload['source_hash'], evidence=result,
                report=report, receipt=receipt, source_retention=payload['source_retention'])
