"""算子产物持久化存储
将生成的算子代码、Sketch、验证结果、性能数据保存到本地文件系统

目录结构:
  output/<kernel_type>/<hardware>/<kernel_id>_<timestamp>/
    code.py          - 生成的算子代码
    sketch.json     - 设计草图
    result.json     - 完整产物 (含验证结果和性能数据)
    benchmark.json  - 性能基准数据 (如果有)支持从目录加载历史产物，用于回溯和对比"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from .schemas import KernelArtifact


class ArtifactStore:
    """算子产物持久化存储"""

    def __init__(self, output_dir: Optional[str] = None):
        # OPERATOR_AGENT_OUTPUT_DIR 覆盖默认落盘位置。为什么要能覆盖: 单元测试每次跑
        # 都会往仓库的 output/ 里灌产物目录, 于是「历史语料」里绝大多数是测试现场 ——
        # 2026-09-26 盘点: 5326 份 execution_snapshot 里 5316 份是 mock_gpu, 2872 份
        # result.json 里 2822 份是 mock_gpu (docs/research/43 第三节)。真实语料因此
        # 不可读。与 logs/ 那条 (e2b455e) 是同一个缺陷的另一面。
        if not output_dir:
            output_dir = os.environ.get("OPERATOR_AGENT_OUTPUT_DIR", "").strip() or None
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: KernelArtifact, task=None) -> Path:
        kernel_type = task.source.kernel_type if task else "unknown"
        hardware = task.target.hardware if task else "unknown"
        language = task.target.dsl if task else "python"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{artifact.kernel_id}"
        artifact_dir = self.output_dir / kernel_type / hardware / dir_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ext = self._lang_ext(language)

        # 1. 保存代码文件
        code_path = artifact_dir / f"code.{ext}"
        code_path.write_text(artifact.code, encoding="utf-8")

        # 2. 保存 Sketch
        sketch_path = artifact_dir / "sketch.json"
        if artifact.sketch:
            sketch_path.write_text(
                json.dumps(artifact.sketch.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # 3. 保存完整产物
        result_data = {
            "kernel_id": artifact.kernel_id,
            "task_id": artifact.task_id,
            "language": language,
            "all_passed": artifact.all_passed,
            "created_at": datetime.fromtimestamp(artifact.created_at).isoformat(),
            "saved_at": datetime.now().isoformat(),
        }
        if task:
            result_data["task"] = task.to_dict()
        if artifact.iteration_history:
            result_data["iteration_history"] = artifact.iteration_history
        # 环境指纹与漂移比较 (doc 49 §3.5 第 1 处)。落在这里而不是 `metadata`:
        # `metadata` 整个不进 result.json (llm_cost 也一样只在内存里), 而指纹的
        # **读者是下一次运行** —— 不落盘就等于没算。
        if artifact.environment_fingerprint:
            result_data["environment_fingerprint"] = artifact.environment_fingerprint
        if artifact.environment_drift is not None:
            result_data["environment_drift"] = artifact.environment_drift
        # 续跑状态 (doc 50)。同上一条的理由: 「这次到底续上了没有」的读者在下一次运行与
        # 事后复盘里, 只在内存里等于没记。`resume_from` 被请求而没续上时, 这条是唯一
        # 留下「为什么」的地方 —— 以前那件事只发生在 stdout 上的一行 print。
        if artifact.resume_status is not None:
            result_data["resume"] = artifact.resume_status
        # LLM 账单 (doc 53 §五.1)。写侧 `conductor.py:926` 把它塞进 `metadata`, 而
        # `metadata` 整个不进 `result.json` —— 所以**每次生产运行的产物里没有这份成本**。
        # 今天能读到它的两处都读的是同一次调用链里的内存对象
        # (`harness_pass_rate.py:176 cost_from_metadata`、`golden_regression.py:357`),
        # 后者再抄两列进 `tests/golden_registry.json` —— 那是 golden 路径的出口, 不是产物。
        # 预算/成本出口的读者在**下一次运行**与事后复盘里, 那里没有内存对象可读,
        # 所以同指纹、同 `resume` 一样单独立顶层键。
        # 只走**登记过的键**，不把整个 `metadata` 倒出来。我原先在这里写的理由是
        # 「metadata 里还放着含真实路径的执行快照外挂」—— 那句**没核过**：今天
        # `artifact.metadata` 只有两个写者 (`conductor.py:219` 的 `stop` 与 `:982` 的
        # `llm_cost`，都是登记键)，执行快照是**另一个文件** `execution_snapshot.json`。
        # 门仍然要留，但它守的是「未登记的键不许过界」这个**未来**契约，
        # 与当前有没有这种键无关（见 CHECKLOG I8）。
        # 缺账单时**不写这个键** —— 「没有成本记录」与「成本为 0」是两件事。
        _llm_cost = (artifact.metadata or {}).get("llm_cost")
        if _llm_cost:
            result_data["llm_cost"] = _llm_cost
        # 「循环为什么停」(doc 53 §7.1)。以前只有 stdout 上的一行 print, 于是产物里
        # 「没练够」「练到顶了」「修坏了」三者完全同形 —— rounds-to-pass 那类统计要的正是它。
        _stop = (artifact.metadata or {}).get("stop")
        if _stop:
            result_data["stop"] = _stop
        result_path = artifact_dir / "result.json"
        result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 4. 保存验证结果
        if artifact.verification_results:
            verify_data = []
            for v in artifact.verification_results:
                verify_data.append({
                    "level": v.level.value,
                    "passed": v.passed,
                    "message": v.message,
                    "metrics": v.metrics,
                })
            verify_path = artifact_dir / "verification.json"
            verify_path.write_text(json.dumps(verify_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 5. 保存性能数据
        if artifact.benchmark_data:
            bench_path = artifact_dir / "benchmark.json"
            bench_path.write_text(
                json.dumps(artifact.benchmark_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return artifact_dir

    #: 往回翻多少个产物目录找「上一次」。仓库 output/ 里存着上千份**没有指纹**的
    #: 历史产物 (2026-09-26 盘点: 2872 份 result.json), 不设上限的话每次运行都要把它们
    #: 全读一遍才承认「没有上一次」。设了上限之后, 找不到就如实说找不到。
    PREVIOUS_SCAN_LIMIT = 200

    def previous_environment(self, kernel_type: str, hardware: str) -> Optional[dict]:
        """最近一次同 `(kernel_type, hardware)` 运行留下的环境指纹

        返回 `{"fingerprint": {...}, "artifact_dir": "...", "saved_at": "..."}`。
        找不到就返回 **None**, 不返回一个空指纹 —— `compare_fingerprints` 把空的一侧
        读成「无从比较」, 而一个 `{}` 的「上一次」会被读成「上一次什么都没报」,
        于是每一个键都变成漂移。

        ⚠️ 调用方必须在**本次落盘之前**问 (Conductor 就是这么调的): 这个方法不知道
        「本次」是谁, 落盘之后再问就会把自己当成上一次。

        按 (kernel_type, hardware) 分桶而不是全库找最近一份: 跨算子比较会把
        「换了一个任务」误报成「换了环境」。目录名以时间戳开头, 所以按 mtime 排就够。
        """
        if not kernel_type or not hardware:
            return None
        base = self.output_dir / kernel_type / hardware
        try:
            if not base.is_dir():
                return None
            candidates = sorted((p for p in base.iterdir() if p.is_dir()),
                                key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return None
        for artifact_dir in candidates[:self.PREVIOUS_SCAN_LIMIT]:
            result_file = artifact_dir / "result.json"
            if not result_file.exists():
                continue          # 循环内预创建的 snap_* 目录只有 execution_snapshot.json
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            fp = data.get("environment_fingerprint")
            if fp:
                return {"fingerprint": fp,
                        "artifact_dir": str(artifact_dir),
                        "saved_at": data.get("saved_at", "")}
        return None

    def list_all(self, kernel_type: str = "", hardware: str = "") -> list[dict]:
        results = []
        base = self.output_dir
        if kernel_type:
            base = base / kernel_type
        if hardware and kernel_type:
            base = base / hardware
        if not base.exists():
            return results
        for hw_dir in sorted(base.iterdir()) if not hardware and kernel_type and hw_dir.is_dir() else [base]:
            if not hw_dir.is_dir():
                continue
            for artifact_dir in sorted(hw_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not artifact_dir.is_dir():
                    continue
                result_file = artifact_dir / "result.json"
                if not result_file.exists():
                    continue
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    data["_dir"] = str(artifact_dir)
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        return results

    def load(self, artifact_dir: str) -> Optional[KernelArtifact]:
        dir_path = Path(artifact_dir)
        if not dir_path.is_dir():
            return None
        result_file = dir_path / "result.json"
        if not result_file.exists():
            return None
        from .schemas import UnifiedSketch, TilingStrategy, MemoryPlan, PipelineStage, VerificationResult, VerificationLevel
        data = json.loads(result_file.read_text(encoding="utf-8"))
        code_path = dir_path / self._find_code_file(dir_path)
        code = code_path.read_text(encoding="utf-8") if code_path else ""
        sketch = None
        sketch_file = dir_path / "sketch.json"
        if sketch_file.exists():
            sketch_data = json.loads(sketch_file.read_text(encoding="utf-8"))
            sketch = UnifiedSketch(
                algorithm=sketch_data.get("algorithm", ""),
                version=sketch_data.get("version", 1),
                tiling=TilingStrategy(**sketch_data.get("tiling", {})),
                memory_plan=MemoryPlan(**sketch_data.get("memory_plan", {})),
                pipeline=[PipelineStage(**p) for p in sketch_data.get("pipeline", [])],
                target_dsl=sketch_data.get("target_dsl", "python"),
                optimization_hints=sketch_data.get("optimization_hints", []),
            )
        verification_results = []
        verify_file = dir_path / "verification.json"
        if verify_file.exists():
            for v in json.loads(verify_file.read_text(encoding="utf-8")):
                verification_results.append(VerificationResult(
                    level=VerificationLevel(v["level"]),
                    passed=v["passed"],
                    message=v.get("message", ""),
                    metrics=v.get("metrics", {}),
                ))
        benchmark_data = {}
        bench_file = dir_path / "benchmark.json"
        if bench_file.exists():
            benchmark_data = json.loads(bench_file.read_text(encoding="utf-8"))
        return KernelArtifact(
            kernel_id=data.get("kernel_id", ""),
            task_id=data.get("task_id", ""),
            code=code,
            sketch=sketch,
            verification_results=verification_results,
            benchmark_data=benchmark_data,
        )

    @staticmethod
    def _lang_ext(language: str) -> str:
        ext_map = {"python": "py", "triton": "py", "cuda": "cu", "hlsl": "hlsl", "mlir": "mlir"}
        return ext_map.get(language.lower(), language.lower())

    @staticmethod
    def _find_code_file(dir_path: Path) -> Optional[Path]:
        for name in ("code.py", "code.cu", "code.triton.py"):
            p = dir_path / name
            if p.exists():
                return p
        for p in dir_path.iterdir():
            if p.suffix in (".py", ".cu", ".c", ".cc") and p.name != "__init__.py":
                return p
        return None

    def __repr__(self) -> str:
        return f"ArtifactStore(dir={self.output_dir})"
