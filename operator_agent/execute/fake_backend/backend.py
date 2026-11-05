"""CPU candidate execution and scripted faults, never an accelerator simulator.

Python/NumPy candidates run through exec_guard, preserving the normal L2 input
readback. Triton/CUDA source is refused rather than replaced with a reference.
Timings are fixed test fixtures. The import restrictions prevent ordinary GPU
paths; exec_guard remains a resource guard, not a hostile-code security sandbox.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ...core.backend_caps import Backend, Cap
from ..exec_guard import (MODE_COMPILE, MODE_RUN, STATUS_CRASHED, candidate_error,
                          limits_for, refusal_from, run_candidate, run_compile)
from ..mock_gpu import ExecutionResult
from .config import FakeConfig, execution_context


_IO_NAMES = frozenset({'ctypeslib', 'ctypes', 'DataSource', 'open', 'load', 'loads',
                       'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt',
                       'genfromtxt', 'fromfile', 'tofile', 'memmap', 'f2py'})
_DYNAMIC_NAMES = frozenset({'__import__', 'eval', 'exec', 'compile', 'getattr',
                          'setattr', 'globals', 'locals', 'vars', 'breakpoint'})


class FakeBackend(Backend):
    def __init__(self, config=None, *, event_path=None):
        self.config = config if isinstance(config, FakeConfig) else FakeConfig.from_dict(config)
        self.event_path = Path(event_path) if event_path is not None else None
        self._fault_source = None
        self._event_index = 0

    @property
    def capabilities(self):
        return frozenset({Cap.TEST_ONLY, Cap.EMULATED, Cap.SIMULATED_PERF,
                          Cap.FULL_OUTPUT_TENSOR})

    @property
    def label(self):
        return "fake:" + self.config.profile

    @property
    def name(self):
        return self.label

    @property
    def generation_instructions(self):
        return (
            "[execution-backend]\n"
            "FakeBackend executes CPU Python with NumPy ndarray inputs. "
            "Import only numpy/math. Do not import torch, triton, cupy or other libraries. "
            "Use NumPy operations and preserve the declared input/output dtype and actual strides. "
            "All execution and timing evidence remains FAKE / test_only.\n"
            "[/execution-backend]"
        )

    def _event(self, operation, source=None, **fields):
        self._event_index += 1
        if self.event_path is not None:
            row = dict(index=self._event_index, operation=operation,
                       scenario=self.config.scenario, profile=self.config.profile,
                       evidence_kind="test_only", monotonic=time.monotonic(), **fields)
            if source is not None:
                row["source_hash"] = hashlib.sha256(source.encode()).hexdigest()
            with self.event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def get_hardware_info(self):
        self._event("hardware")
        if self.config.scenario == "hardware-missing":
            raise RuntimeError("fake hardware-missing: injected hardware probe failure")
        # These profiles exercise resource adaptation, not real GPU identities.
        large = self.config.profile == "large"
        report = {} if self.config.profile == "unknown" else dict(
            shared_memory_per_sm_kb=228 if large else 48,
            shared_memory_per_block_optin_bytes=(227 if large else 48) * 1024,
            max_threads_per_sm=2048, sm_count=16 if large else 4)
        return dict(name="FakeGPU-" + self.config.profile, device="cpu",
                    cuda_available=False, triton_available=False,
                    compute_units=16 if large else 4, warp_size=32,
                    hardware_report=report,
                    execution_context=execution_context(self.config),
                    memory={"global": {"size": "1GB"},
                            "local": ({"size": str(report["shared_memory_per_sm_kb"]) + "KB/SM"}
                                      if report else {})})

    @staticmethod
    def _check_source(source):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] if not node.level else [""]
                if any(alias.name in _IO_NAMES or alias.name.startswith('_')
                       or alias.name == '*' for alias in node.names):
                    raise ValueError("dynamic code/IO is unsupported by the fake backend")
            else:
                modules = []
            if any(module not in {"numpy", "math"} for module in modules):
                raise ValueError("fake backend supports CPU Python with numpy/math imports only")
            if isinstance(node, ast.Name) and node.id in _DYNAMIC_NAMES | _IO_NAMES:
                raise ValueError("dynamic code/IO is unsupported by the fake backend")
            if isinstance(node, ast.Attribute) and (node.attr.startswith("_")
                                                    or node.attr in _IO_NAMES):
                raise ValueError("dynamic code/IO is unsupported by the fake backend")

    def _injected(self, source, scenario):
        if self.config.scenario != scenario:
            return False
        digest = hashlib.sha256(source.encode()).hexdigest()
        if self._fault_source is None:
            self._fault_source = digest
        return digest == self._fault_source

    def compile(self, kernel_code):
        self._event("compile", kernel_code)
        self._check_source(kernel_code)
        if self.config.scenario == "timeout":
            self._event("blocked", kernel_code)
            # The ordinary supervisor must enforce wall time/cancellation.
            while True:
                time.sleep(.25)
        if self._injected(kernel_code, "compile-repair"):
            raise SyntaxError("fake compile-repair: injected first-source compile failure")
        guard = run_compile(kernel_code, limits=limits_for(MODE_COMPILE, None))
        if not guard.ok:
            if guard.status == STATUS_CRASHED:
                raise candidate_error(guard)
            raise refusal_from(guard, where="fake CPU compile")
        if not guard.metrics.get("has_kernel") or not guard.metrics.get("kernel_callable"):
            raise SyntaxError("fake backend requires a callable kernel")
        return True

    def correctness_input_support(self, kernel_code, layout):
        try:
            self._check_source(kernel_code)
        except (SyntaxError, ValueError) as exc:
            return False, str(exc)
        return (layout in ("contiguous", "strided", "broadcast"),
                "CPU candidate with exec_guard storage/view readback; test evidence only")

    def execute(self, kernel_code, inputs):
        self._event("execute", kernel_code)
        result = ExecutionResult(device="cpu", metrics=execution_context(self.config))
        self._check_source(kernel_code)
        if self.config.scenario == "transport-unknown":
            result.success = False
            result.error = "fake transport-unknown: injected lost reply; no network was used"
            result.metrics["transport_outcome"] = "outcome_unknown"
            return result
        guard = run_candidate(kernel_code, inputs, preserve_inputs=True,
                              limits=limits_for(MODE_RUN, None))
        result.metrics.update(guard.metrics, exec_guard_executor=guard.executor)
        if not guard.fully_enforced:
            result.metrics["exec_guard_not_enforced"] = list(guard.not_enforced)
        result.success = guard.ok
        if not guard.ok:
            result.error = guard.error or guard.detail
            result.metrics.update(exec_guard_status=guard.status, exec_guard_kind=guard.kind)
            return result
        result.output = np.asarray(guard.output)
        result.execution_time_ms = guard.elapsed_ms  # CPU observation, never GPU timing.
        if self._injected(kernel_code, "correctness-repair"):
            result.output = result.output + np.array(1, dtype=result.output.dtype)
            result.metrics["injected_fault"] = "wrong_output"
        return result

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        self._event("benchmark", kernel_code, warmup=warmup, repeat=repeat)
        self._check_source(kernel_code)
        candidate = dict(avg_time_ms=1., median_time_ms=1., min_time_ms=1., max_time_ms=1.,
                         std_time_ms=0., dispersion_iqr_over_median=0.)
        baseline_ms = 1. if self.config.scenario == 'no-improvement' else 2.
        return dict(execution_context(self.config), success=True, mode="simulated", device="cpu",
                    **candidate, candidate=candidate, gflops=1., memory_bandwidth_gbs=1.,
                    baseline_selected='fake_fixture',
                    baselines={'fake_fixture': dict(median_time_ms=baseline_ms,
                                                    evidence_kind='test_only')})
