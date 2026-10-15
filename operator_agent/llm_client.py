"""
LLM 客户端 - 统一的 LLM 调用封装
支持 OpenAI 兼容 API (私有部署模型 / DeepSeek)
支持主模型失败后自动降级到 fallback 模型
通过环境变量配置:
  LLM_BASE_URL  - API 地址 (必填, 仓库内不保存真实地址)
  LLM_API_KEY   - API Key (可选)
  LLM_MODEL     - 模型名称 (默认 Qwen3.8-27B)
  HY_API_KEY    - Fallback 模型 API Key (可选)

=== 成本监控 Harness (Usage Tracker) ===
每次 chat/chat_json 调用都自动记录 usage (prompt_tokens / completion_tokens / total_tokens / 估算费用),
可通过 client.cost_tracker.summary() 获取账单摘要, 或 client.cost_tracker.reset() 清零。
API 未返回 usage 时, 退化为基于字符数的轻量估算 (4 chars ≈ 1 token)。
"""

import os
import json
import re
import time
import urllib.request
import urllib.error
import urllib.parse
import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

from .llm_budget import LLMCallBudget, LLMCallRefused, strict_object, validated_usage
from .llm_transport import dispatch as bounded_dispatch


# ─── 常见模型的每 M tokens 参考价格 (单位: USD / 1M tokens, input / output) ───
# 数据为粗略参考, 不承诺精度; 对未知模型使用 conservative 默认值 (3$/M input, 15$/M output)
_MODEL_PRICING = {
    # 开源自托管模型按"电+带宽"近似估算, 统一用一个很低的默认
    "default_self_hosted": {"input_per_m": 0.30, "output_per_m": 1.00},
    # 常见 API 模型 (更新时只改此表)
    "qwen/qwen3.5-9b": {"input_per_m": 0.20, "output_per_m": 0.60},
    "qwen/qwen3.5-32b": {"input_per_m": 0.80, "output_per_m": 2.40},
    "deepseek-v3": {"input_per_m": 0.14, "output_per_m": 0.28},
    "deepseek-chat": {"input_per_m": 0.14, "output_per_m": 0.28},
}


def parse_usage(usage: dict) -> dict:
    """把两种 usage 约定归一成同一个口径, 并把约定本身记下来

    外部一手 (arXiv:2609.11987v1 行 78 与行 1108-1111, 逐字):
        "input-token fields are cache-inclusive on some SDKs and cache-exclusive on others,
         and a normalizer that assumes one convention doubles the other."
    那篇为这件事付过一次代价: 它八月版的**全部成本结论**与 stop-loss 用的钱都被这个归一器
    污染了 (§5.1 与行 1073), 而**没有影响任何判定** —— 所以「钱算错而结果不变」是可以整篇
    存在的失效形态。我方 #71 的预算出口正要拿这份账单当货币, 因此这里先把约定本身记进账本。

    两种形状:
      · inclusive (OpenAI chat/completions 与 Responses、LangChain UsageMetadata):
        输入总量**已含**缓存命中, 缓存数在 `prompt_tokens_details.cached_tokens`
        或 `input_tokens_details.cached_tokens`;
      · exclusive (Anthropic Messages): `input_tokens` **不含** `cache_read_input_tokens`
        与 `cache_creation_input_tokens`, 三者相加才是这次送进去的输入。

    ⚠️ 判别信号是**哪个字段出现了**, 不是键名 —— `input_tokens` 这一个键在 OpenAI Responses 上
    是包含、在 Anthropic 上是排除 (外部那篇行 1108-1111 把三家并列就在这儿)。第一版我按
    「有 `input_tokens` 就走 exclusive」判, 于是 Responses 的缓存被读成 0 —— 恰好是那句
    "a normalizer that assumes one convention doubles the other" 的镜像: 假设错了方向,
    不是翻倍就是漏掉, 而两种都表现为「一个像样的数字」。

    Returns:
        {"prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cache_semantics"}
        cache_semantics ∈ {"inclusive", "exclusive", "absent"} (没有 usage 时 "absent")
    """
    usage = usage or {}
    total_in = usage.get("prompt_tokens")
    if total_in is None:
        total_in = usage.get("input_tokens")
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens")
    total_in, completion = int(total_in or 0), int(completion or 0)

    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)

    details_cached = 0
    # 只认这两种 details 键 (OpenAI chat / Responses)。LangChain 的 `input_token_details`
    # 本仓不跑那个栈, 真接进来时这里会退成 absent —— 少报缓存不会把钱算成两倍, 方向是安全的
    for holder in ("prompt_tokens_details", "input_tokens_details"):
        d = usage.get(holder) or {}
        if isinstance(d, dict) and d.get("cached_tokens"):
            details_cached = int(d["cached_tokens"])
            break

    if details_cached:
        # inclusive: 缓存已经在总量里, **不许再加一次** (那就是那句 doubles)
        return {"prompt_tokens": total_in, "completion_tokens": completion,
                "cached_prompt_tokens": min(details_cached, total_in),
                "cache_semantics": "inclusive"}
    if read or write:
        # exclusive: 读/写缓存要加进总量, 否则账单会少掉一整段
        return {"prompt_tokens": total_in + read + write, "completion_tokens": completion,
                "cached_prompt_tokens": read, "cache_semantics": "exclusive"}
    return {"prompt_tokens": total_in, "completion_tokens": completion,
            "cached_prompt_tokens": 0, "cache_semantics": "absent"}


def _estimate_tokens_from_text(text: str) -> int:
    """当 API 没有返回 usage 时, 用 4 chars ≈ 1 token 做保守估算 (中文字符按 1 char=1.7 tokens 修正)"""
    if not text:
        return 0
    # 粗略区分中英: 统计中文字符占比, 平均中文字符 1.7 tokens, 英文 0.25 tokens
    n_chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    n_other = len(text) - n_chinese
    return int(n_chinese * 1.7 + n_other * 0.25) + 1


@dataclass
class _LLMCallRecord:
    """单次 LLM 调用的账本条目"""
    ts: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    caller_tag: str           # 调用方标签, 如 "Designer" / "Coder"
    usage_source: str         # "api_usage" | "estimated"
    cached_prompt_tokens: int = 0
    cache_semantics: str = "absent"   # inclusive | exclusive | absent
    cost_basis: str = "list_price"    # list_price | upper_bound_cache_unpriced
    # 端点**自报**的模型标识 (响应里的 `model` 字段), 与请求的那个并排存 (doc 58 §2.4 / #89)。
    # None = 响应里根本没有这个字段 —— 那是「不知道」而不是「一致」。
    reported_model: Optional[str] = None
    # 本次**真发出去**的采样配置 (#98 / doc 59 §四)。三个值必须在造 body 的那一处一次性取好
    # 再分给 body 与账本两个读者 —— 让报表另读一遍构造参数会印成 `client.temperature` (0.3),
    # 而逐调用覆盖在这里是常态 (Designer 0.3 / Coder 0.1 / Fixer 0.1 / planner 0.2 …),
    # 那就是 #52「记录与事实不同源」的同一条。
    # None = 这条记录没带采样配置 (假账本、或写这三个字段之前的记录), **不是**「值为 0」
    # 也不是「取默认」。
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_thinking: Optional[bool] = None


def _or_dash(value):
    """`None` 印成 `—` —— 「这一项没记到」不是一个取值"""
    return "—" if value is None else value


def format_sampling_configs(sampling: dict) -> list:
    """把 `CostTracker.sampling_configs` 的结果印成行 (#98 / doc 59 §四)

    `pretty()` 与 `benchmarks/harness_pass_rate.py` 两个读者共用这一份渲染: 各印一遍就会
    各漂移各的, 而这张表的用途是让两个人对同一批数字得出同一个读数条件。
    """
    out: list = []
    for caller, configs in (sampling.get("by_caller") or {}).items():
        for c in configs:
            out.append(
                f"{caller}: model={_or_dash(c['model'])} "
                f"temperature={_or_dash(c['temperature'])} "
                f"max_tokens={_or_dash(c['max_tokens'])} "
                f"enable_thinking={_or_dash(c['enable_thinking'])} ×{c['calls']} 次")
    missing = sampling.get("n_records_without_sampling") or 0
    if missing:
        out.append(f"⚠️ 另有 {missing} 条调用没记采样配置 ⇒ 上面这张表不全, "
                   f"「没记」不等于「用了默认值」")
    return out


class CostTracker:
    """LLM 调用账本: 累计 tokens / 估算费用 / 调用次数, 支持按 caller 分解"""

    def __init__(self):
        self.records: list[_LLMCallRecord] = []

    # ----- 写入 -----
    def record(self, *, model: str, prompt_tokens: int, completion_tokens: int,
               usage_source: str, caller_tag: str = "Unknown",
               cached_prompt_tokens: int = 0, cache_semantics: str = "absent",
               reported_model: Optional[str] = None,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               enable_thinking: Optional[bool] = None) -> None:
        total = prompt_tokens + completion_tokens
        # 查价格表: 小写匹配 model 前缀
        key_lower = model.lower()
        pricing = _MODEL_PRICING.get(key_lower)
        if pricing is None:
            # 模糊匹配: 取第一个 key 是 model 前缀
            for k, p in _MODEL_PRICING.items():
                if k != "default_self_hosted" and (key_lower.startswith(k) or k in key_lower):
                    pricing = p
                    break
        if pricing is None:
            pricing = _MODEL_PRICING["default_self_hosted"]
        cost_usd = (
            prompt_tokens * pricing["input_per_m"] / 1_000_000
            + completion_tokens * pricing["output_per_m"] / 1_000_000
        )
        # 价格表里没有缓存输入的折扣价 —— 这不是疏忽, 是本仓不抄没核过的数。
        # 于是「有缓存命中的那些调用」这条记录只能标成**按全价算的**, 不能标成账单:
        # 若供应商按折扣计 (外部那篇给 Anthropic ~0.1x 读、OpenAI ~50% 缓存折扣, 行 1100-1101),
        # 这份数就是上界; 方向不明时也只说「没单独计价」, 不说它是上界还是下界。
        cost_basis = "list_price_cache_unpriced" if cached_prompt_tokens > 0 else "list_price"
        self.records.append(_LLMCallRecord(
            ts=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=round(cost_usd, 6),
            caller_tag=caller_tag,
            usage_source=usage_source,
            cached_prompt_tokens=cached_prompt_tokens,
            cache_semantics=cache_semantics,
            cost_basis=cost_basis,
            reported_model=reported_model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        ))

    def reset(self) -> None:
        self.records.clear()

    # ----- 查询 -----
    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    def breakdown_by_caller(self, records=None) -> dict:
        """按 caller 分组汇总 {caller: {calls, prompt, completion, total, cost_usd}}"""
        out = {}
        for r in self.records if records is None else records:
            bucket = out.setdefault(r.caller_tag, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "cost_usd": 0.0,
            })
            bucket["calls"] += 1
            bucket["prompt_tokens"] += r.prompt_tokens
            bucket["completion_tokens"] += r.completion_tokens
            bucket["total_tokens"] += r.total_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + r.cost_usd, 6)
        return out

    def sampling_configs(self, records=None) -> dict:
        """把「真发出去的采样配置」按 caller 归并成**组合**计数 (#98 / doc 59 §四)

        三条规则，各对应一种已经犯过的读法：

        1. 逐调用覆盖在这里是常态（Designer 0.3 / Coder 0.1+40960 / Fixer 0.1+1024 /
           planner 0.2），一个标量代表不了 ⇒ 按 (model, temperature, max_tokens,
           enable_thinking) 的**组合**计数。把各轴分开取集合会配出四种根本不存在的
           配置（把「0.1 配 1024」和「0.3 配 40960」混成笛卡尔积）。
        2. 三项**全缺**的条目单独计数成 `n_records_without_sampling`，不折进任何组合：
           「没记」既不是「温度为 0」也不是「取默认」（#53 那一族：缺证据当否定证据）。
        3. 只缺其中一两项的条目照常归组，缺的那一格留 None ⇒ 读者印 `—`。
        """
        rows = self.records if records is None else records
        combos: dict = {}
        without = 0
        for r in rows:
            if r.temperature is None and r.max_tokens is None and r.enable_thinking is None:
                without += 1
                continue
            key = (r.caller_tag, r.model, r.temperature, r.max_tokens, r.enable_thinking)
            combos[key] = combos.get(key, 0) + 1
        by_caller: dict = {}
        for (caller, model, temp, mx, think), n in combos.items():
            by_caller.setdefault(caller, []).append({
                "model": model, "temperature": temp, "max_tokens": mx,
                "enable_thinking": think, "calls": n,
            })
        for configs in by_caller.values():
            configs.sort(key=lambda c: (-c["calls"], str(c["temperature"]),
                                        str(c["max_tokens"])))
        return {
            "n_records_with_sampling": sum(combos.values()),
            "n_records_without_sampling": without,
            "by_caller": dict(sorted(by_caller.items())),
        }

    def _summarize(self, records) -> dict:
        semantics: dict = {}
        # 端点自报的模型标识 vs 请求的那个 (doc 58 §2.4 / #89)。刻意**不做归一化**:
        # 外部那篇按日期快照归一 (`model-2026-08-01` 与 `model` 视为同一个), 那是它自己
        # 端点上的规则 —— 本仓不抄没核过的映射表, 所以这里连「同族不同日期」也算不同。
        # 与之配套的是 `n_reported_model_absent`: 响应里没这个字段是**不知道**,
        # 不是「一致」—— 缺证据不许算进「对上了」那一侧。
        mismatches: dict = {}
        absent = 0
        for r in records:
            semantics[r.cache_semantics] = semantics.get(r.cache_semantics, 0) + 1
            if not r.reported_model:
                absent += 1
            elif r.reported_model != r.model:
                key = f"{r.model} -> {r.reported_model}"
                mismatches[key] = mismatches.get(key, 0) + 1
        return {
            "total_calls": len(records),
            "total_prompt_tokens": sum(r.prompt_tokens for r in records),
            "total_completion_tokens": sum(r.completion_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "total_cached_prompt_tokens": sum(r.cached_prompt_tokens for r in records),
            "cache_semantics": semantics,
            "n_cache_unpriced_calls": sum(1 for r in records
                                          if r.cost_basis == "list_price_cache_unpriced"),
            "model_mismatches": mismatches,
            "n_model_mismatch_calls": sum(mismatches.values()),
            "n_reported_model_absent": absent,
            "total_cost_usd": round(sum(r.cost_usd for r in records), 6),
            "breakdown_by_caller": self.breakdown_by_caller(records),
            "n_estimated_usage": sum(1 for r in records if r.usage_source == "estimated"),
            # 「这次跑到底是在哪个采样配置下测的」(#98)。报表把它当成读数条件之一,
            # 所以它必须和 body 同源 —— 见 `_LLMCallRecord.temperature` 那条注释。
            "sampling": self.sampling_configs(records),
        }

    def summary(self) -> dict:
        """返回结构化账单摘要 (供 Conductor 日志输出)

        ⚠️ 这是**整个账本**的累计值。账本挂在 client 上、`reset()` 全仓没人调，所以一个
        client 跑多个任务时，第二个任务的 `summary()` 里含着第一个任务的量
        (`harness_pass_rate.run_one` 的 docstring 记过一次真实事故：六次运行的 token
        被当成六份独立样本再求和，总量高估 3.25 倍)。要「这一次运行花的」用
        `summary_since(watermark())`。
        """
        return self._summarize(self.records)

    def watermark(self) -> int:
        """本次运行开始时的账本长度；配 `summary_since` 用。"""
        return len(self.records)

    def summary_since(self, n: int) -> dict:
        """只汇总第 `n` 条之后的记录 —— 即「这一次运行花的」，不是「这个进程到现在花的」。"""
        return self._summarize(self.records[n:])

    def pretty(self) -> str:
        """人可读的账单摘要 (终端 print 用)"""
        s = self.summary()
        lines = []
        lines.append(f"LLM 账单摘要: 调用次数={s['total_calls']}, "
                     f"tokens={s['total_prompt_tokens']}in / {s['total_completion_tokens']}out "
                     f"(合计 {s['total_tokens']}), 估算≈${s['total_cost_usd']:.4f}")
        if s["n_estimated_usage"]:
            lines.append(f"  * 其中 {s['n_estimated_usage']} 次 API 未返回 usage, 使用字符估算")
        if s.get("n_cache_unpriced_calls"):
            lines.append(
                f"  * 其中 {s['n_cache_unpriced_calls']} 次的输入含缓存命中 "
                f"({s['total_cached_prompt_tokens']} tokens, 约定 {s['cache_semantics']}), "
                f"价格表没有缓存价 ⇒ 这个钱是**按全价算的**, 不是账单")
        if s["n_model_mismatch_calls"]:
            lines.append(f"  ⚠️ {s['n_model_mismatch_calls']} 次调用里端点**自报**的模型与请求的不同"
                         f" (逐族计数, 不做同族归一):")
            for pair, n in sorted(s["model_mismatches"].items(), key=lambda kv: -kv[1]):
                lines.append(f"    - {pair}: {n} 次")
            lines.append("    自报只作参考 —— 它可能答的是路由器后面的另一条线路 (doc 58 §2.4)")
        if s["n_reported_model_absent"]:
            lines.append(f"  * {s['n_reported_model_absent']} 次的响应里没有 model 字段 "
                         f"⇒ 记为「不知道」, 不算「对上了」")
        if s["breakdown_by_caller"]:
            lines.append("  按调用方分解:")
            for caller, b in sorted(s["breakdown_by_caller"].items(), key=lambda kv: -kv[1]["cost_usd"]):
                lines.append(
                    f"    - {caller}: {b['calls']} 次, "
                    f"{b['prompt_tokens']}in / {b['completion_tokens']}out, "
                    f"≈${b['cost_usd']:.4f}"
                )
        # #98: 账单旁边印「这次是在什么采样配置下跑的」。放在这里而不是只进报表,
        # 因为一次普通 run 的读者只有这份日志 —— 只落盘不印出来就又是「算了没人读」。
        sampling_lines = format_sampling_configs(s["sampling"])
        if sampling_lines:
            lines.append("  采样配置 (逐调用覆盖之后, 真发出去的那些):")
            lines.extend(f"    - {x}" for x in sampling_lines)
        return "\n".join(lines)


class LLMClient:
    """OpenAI 兼容 API 的 LLM 客户端 (纯 urllib，无第三方依赖)"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        timeout: int = 120,
        fallback_model: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        call_budget: Optional[LLMCallBudget] = None,
    ):
        # 端点不写进仓库: 由 base_url 参数或 LLM_BASE_URL 环境变量提供
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "未配置 LLM 端点: 请传入 base_url 或设置环境变量 LLM_BASE_URL "
                "(形如 http://host:port/v1)。仓库内不保存真实地址。"
            )
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "Qwen3.8-27B")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 之前这条只写在 body 的字面量里 (硬编码 False), 于是「我方所有通过率数字都是在
        # 关掉思考的配置下测出来的」这件事在报表里没有出处 (#98 / doc 59 §四)。挪到这里
        # ⇒ 发出去的那个值和报表里印的那个值是同一个属性。
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self._chat_url = f"{self.base_url}/chat/completions"

        # Fallback 模型配置
        self.fallback_model = fallback_model or os.environ.get("LLM_FALLBACK_MODEL", "")
        self.fallback_base_url = (fallback_base_url or os.environ.get(
            "LLM_FALLBACK_BASE_URL", ""
        )).rstrip("/") if (fallback_base_url or os.environ.get("LLM_FALLBACK_BASE_URL", "")) else ""
        self.fallback_api_key = fallback_api_key or os.environ.get("HY_API_KEY", "")

        # 追踪实际使用的模型 (主模型/降级模型)
        self._active_model: str = self.model
        self._active_base_url: str = self.base_url
        self._fallback_triggered: bool = False

        # ── 成本监控 Harness: 账本实例, 每次 LLMClient 创建都带独立账本
        self.cost_tracker: CostTracker = CostTracker()

        # Explicit bounded campaigns are opt-in. Legacy callers retain their
        # old behavior; they are NOT claimed to have a pre-request hard cap.
        self._call_budget = call_budget
        self._bounded_key = self.api_key
        if call_budget is not None:
            if not isinstance(call_budget, LLMCallBudget):
                raise TypeError('LLMCallBudget required')
            self._validate_bounded_endpoint()
            call_budget.bind(self._bounded_identity())

    @property
    def call_budget(self):
        return self._call_budget

    def _validate_bounded_endpoint(self):
        try:
            parsed = urllib.parse.urlsplit(self.base_url)
            port = parsed.port
            valid = (parsed.scheme in ('https', 'http') and parsed.hostname
                     and not parsed.username and not parsed.password
                     and not parsed.query and not parsed.fragment)
        except ValueError:
            valid = False
        if not valid or type(self.model) is not str or not self.model:
            raise LLMCallRefused('invalid bounded model endpoint/identity')

    def _bounded_identity(self):
        return dict(endpoint_sha256=hashlib.sha256(self.base_url.encode()).hexdigest(),
                    model=self.model, credential_present=bool(self.api_key),
                    enable_thinking=self.enable_thinking, timeout=self.timeout,
                    default_temperature=self.temperature, default_max_tokens=self.max_tokens)

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
        raw: bool = False,
        caller_tag: str = "Unknown",
    ) -> str:
        """
        发送聊天请求并返回模型回复文本。
        自动处理 Qwen3 的 Thinking Process 前缀。
        主模型失败时自动切换到 fallback 模型。

        Args:
            caller_tag: 调用方标识, 写入 CostTracker 账本, 便于按 Agent 类型分解账单
        """
        if self.call_budget is not None:
            # Bypass the fallback handler entirely. Every agent shares this
            # ledger; changing caller or resetting CostTracker grants no credit.
            self._active_model, self._active_base_url = self.model, self.base_url
            self._fallback_triggered = False
            return self._call_model(self.base_url, self.api_key, self.model,
                                    messages, temperature, max_tokens, raw, caller_tag)
        try:
            self._active_model = self.model
            self._active_base_url = self.base_url
            self._fallback_triggered = False
            return self._call_model(
                self.base_url, self.api_key, self.model,
                messages, temperature, max_tokens, raw, caller_tag,
            )
        except Exception as primary_error:
            # 主模型失败，尝试 fallback
            if self.fallback_model:
                print(f"  [LLM] 主模型 [{self.model}] 调用失败: {primary_error}")
                print(f"  [LLM] 自动降级到 [{self.fallback_model}] @ {self.fallback_base_url or self.base_url}")
                try:
                    self._active_model = self.fallback_model
                    self._active_base_url = self.fallback_base_url or self.base_url
                    self._fallback_triggered = True
                    return self._call_model(
                        self._active_base_url,
                        self.fallback_api_key or self.api_key,
                        self.fallback_model,
                        messages, temperature, max_tokens, raw, caller_tag,
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"主模型 [{self.model}] 失败: {primary_error}; "
                        f"降级模型 [{self.fallback_model}] 也失败: {fallback_error}"
                    )
            raise primary_error

    def _call_model(
        self,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[float],
        raw: bool,
        caller_tag: str,
    ) -> str:
        """实际发起一次 LLM API 调用的内部方法, 调用结束后写入 usage 账本"""
        chat_url = f"{base_url}/chat/completions"
        # 逐调用覆盖在这里是常态 (`agents.py` 里 Designer 0.3/2048、Coder 0.1/40960,
        # `fixer.py` 0.1/1024, `planner.py` 0.2/2048), 所以「本次真发出去的几个值」只能
        # 在这一处算一次, 再分给 body、错误文案、账本三个读者。报表若另读
        # `client.temperature` 就会把 0.1 的那一半印成 0.3 —— #52 同一条 (#98)。
        sent_temperature = self.temperature if temperature is None else temperature
        sent_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        body = {
            "model": model,
            "messages": messages,
            "temperature": sent_temperature,
            "max_tokens": sent_max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        if self.call_budget is not None:
            return self._call_bounded_model(base_url, api_key, model, body, raw, caller_tag)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            chat_url, data=data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM API 请求失败 [{e.code}]: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"LLM API 连接失败: {e.reason}"
            ) from e

        message = result["choices"][0]["message"]
        finish_reason = result["choices"][0].get("finish_reason", "")
        content = message.get("content")

        # ── 成本监控 Harness: 从响应里取 usage, 没有则回退估算 ──
        usage = result.get("usage") or {}
        norm = parse_usage(usage)
        prompt_tokens = norm["prompt_tokens"]
        completion_tokens = norm["completion_tokens"]
        usage_source = "api_usage"
        if prompt_tokens <= 0 and completion_tokens <= 0:
            # API 未返回 usage, 基于请求/响应文本做字符级估算
            usage_source = "estimated"
            try:
                flat_input = json.dumps(messages, ensure_ascii=False)
            except Exception:
                flat_input = "".join(m.get("content", "") for m in messages if isinstance(m, dict))
            prompt_tokens = _estimate_tokens_from_text(flat_input)
            completion_tokens = _estimate_tokens_from_text(content or "")

        self.cost_tracker.record(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_source=usage_source,
            caller_tag=caller_tag,
            cached_prompt_tokens=norm["cached_prompt_tokens"],
            cache_semantics=norm["cache_semantics"],
            # 端点自报的标识 (OpenAI 兼容层通常都会带 `model`)。取**原始值**, 不裁剪、不映射:
            # 观测阶段要的是「它今天说自己是谁」, 归一化规则没核过就不写 (doc 58 §2.4 / #89)
            reported_model=(result.get("model") or None),
            # 记账用**同一个局部变量**, 不是再算一遍 `temperature or self.temperature`:
            # 两处各写一遍就是两条会各自漂移的路径 (doc 33 `runs` / doc 37 `llm_cost` /
            # doc 48 `candidate_signature` 那一族「值算出来死在搬运路上」的反面写法)。
            temperature=sent_temperature,
            max_tokens=sent_max_tokens,
            enable_thinking=self.enable_thinking,
        )

        if content is None or (isinstance(content, str) and content.strip() == ""):
            if finish_reason == "length":
                raise RuntimeError(
                    f"LLM 回复被截断 (finish_reason=length): "
                    f"思考过程耗尽 max_tokens={sent_max_tokens}，"
                    f"未产生最终输出。请增大 max_tokens 或简化请求。"
                )
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                content = reasoning
            else:
                raise RuntimeError("LLM 返回空内容 (content 和 reasoning_content 均为空)")

        if raw:
            return content

        content = self._strip_thinking(content)
        return content.strip()

    def _call_bounded_model(self, base_url, api_key, model, body, raw, caller_tag):
        budget = self.call_budget
        self._validate_bounded_endpoint()
        if (base_url != self.base_url or api_key != self._bounded_key
                or self.api_key != self._bounded_key or model != self.model):
            raise LLMCallRefused('frozen model configuration drift')
        if (type(body['temperature']) not in (int, float)
                or not math.isfinite(body['temperature']) or not 0 <= body['temperature'] <= 2
                or type(self.enable_thinking) is not bool):
            raise LLMCallRefused('invalid bounded sampling configuration')
        try:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            raise LLMCallRefused('invalid bounded request') from None
        call, seconds = budget.reserve(self._bounded_identity(), data, caller_tag, self.timeout)
        start = time.monotonic()
        response_text, outcome, usage_source = None, 'failed', None
        try:
            response_text = bounded_dispatch(dict(url=base_url + '/chat/completions',
                key=api_key, body_text=data, timeout=seconds,
                max_response_bytes=budget.limits.max_response_bytes), seconds)
            if (type(response_text) is not str
                    or len(response_text.encode()) > budget.limits.max_response_bytes):
                response_text = None  # Never retain an unbounded transport result.
                raise ValueError('response_size_limit')
            result = strict_object(response_text)
            choices = result['choices']
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('exactly one response choice required')
            choice = choices[0]
            content = choice['message'].get('content')
            usage, usage_source = validated_usage(result.get('usage'))
            norm = parse_usage(usage)
            self.cost_tracker.record(model=model,
                prompt_tokens=norm['prompt_tokens'] if usage_source == 'api_usage' else _estimate_tokens_from_text(data),
                completion_tokens=norm['completion_tokens'] if usage_source == 'api_usage' else _estimate_tokens_from_text(content if isinstance(content, str) else ''),
                usage_source=usage_source, caller_tag=caller_tag,
                cached_prompt_tokens=norm['cached_prompt_tokens'], cache_semantics=norm['cache_semantics'],
                reported_model=result.get('model'), temperature=body['temperature'],
                max_tokens=body['max_tokens'], enable_thinking=self.enable_thinking)
            if (result.get('model') != model or choice.get('finish_reason') != 'stop'
                    or type(content) is not str or not content.strip()
                    or norm['completion_tokens'] > body['max_tokens']
                    or time.monotonic() - start > seconds):
                raise ValueError('invalid/truncated/over-budget model output')
            outcome = 'answered'
            # No reasoning substitution, JSON repair or mechanical code rewrite.
            return content if raw else content.strip()
        except Exception:
            raise LLMCallRefused('bounded model call failed; no fallback/retry') from None
        finally:
            budget.finish(call, response=response_text, outcome=outcome,
                          wall_seconds=time.monotonic() - start, usage_source=usage_source)

    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
        caller_tag: str = "Unknown",
    ) -> dict:
        """聊天并尝试将回复解析为 JSON dict"""
        text = self.chat(messages, temperature, max_tokens, caller_tag=caller_tag)
        if self.call_budget is not None:
            try:
                return strict_object(text)
            except (ValueError, TypeError):
                self.call_budget.abort('invalid_json')
                raise LLMCallRefused('strict JSON output required; no repair/retry') from None
        return self._extract_json(text)

    # ── 内部方法 ──────────────────────────────────────────

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        剥离 Qwen3 等模型输出的 Thinking Process。
        处理多种格式:
         标签
        2. "Thinking Process:" 纯文本 (可能跨多段)
        3. 无明确分隔的 thinking (直接找 JSON 或实际内容)
        """
        # 1) <think>...</think> 标签
        m = re.search(r"<think>\s*.*?\s*</think>\s*\n?", text, re.DOTALL)
        if m:
            return text[m.end():].strip()

        # 2) "Thinking Process:" 纯文本模式
        #    Qwen3 的 thinking 可能包含多个段落，不能简单在第一个空行截断
        #    策略: 找到 thinking 开始位置，然后找最后一个看起来像 thinking 的段落
        if re.search(r"Thinking\s*Process:", text, re.IGNORECASE):
            # 将文本按双换行分段
            paragraphs = re.split(r"\n\s*\n", text)
            # 找到第一个非 thinking 段落
            found_thinking = False
            content_start = -1
            for i, para in enumerate(paragraphs):
                is_thinking = (
                    re.search(r"Thinking\s*Process:", para, re.IGNORECASE)
                    or para.strip().startswith(("-", "*", "1.", "2.", "3.", "4.", "5."))
                    or para.strip() == ""
                    or para.strip().startswith("Analy")
                    or para.strip().startswith("Let")
                    or para.strip().startswith("The user")
                    or para.strip().startswith("I need")
                    or para.strip().startswith("Looking")
                    or para.strip().startswith("Based on")
                )
                if is_thinking:
                    found_thinking = True
                    continue
                if found_thinking and para.strip():
                    content_start = text.find(para)
                    break
            if content_start >= 0:
                return text[content_start:].strip()

        # 3) 如果文本以 "Thinking" 开头但无法用上述方法分离
        #    尝试找 JSON 或代码块的起始位置
        if text.strip().lower().startswith("thinking"):
            for marker in ["{", "```", "import ", "def ", "#"]:
                idx = text.find(marker)
                if idx > 0:
                    candidate = text[idx:].strip()
                    if candidate:
                        return candidate

        # 4) 如果文本已经以代码开头，直接返回
        if text.strip().startswith(("import ", "@triton", "def ", "{", "```")):
            return text.strip()

        return text

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从文本中提取第一个 JSON 块并解析，自动修复常见 JSON 错误"""
        # 尝试 ```json ... ``` 代码块
        m = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试找最后一个 { ... } 块 (thinking 可能包含 JSON-like 片段)
        all_matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
        for m in reversed(all_matches):
            candidate = m.group(0)
            try:
                result = json.loads(candidate)
                # 确保是有效的 dict (不是空 dict 或碎片)
                if result and len(result) > 0:
                    return result
            except json.JSONDecodeError:
                # 尝试修复常见错误
                fixed = LLMClient._fix_json(candidate)
                if fixed:
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        continue

        # 最后尝试: 找第一个 { 到最后一个 } 之间的内容
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidate = text[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                fixed = LLMClient._fix_json(candidate)
                if fixed:
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"无法从 LLM 回复中提取 JSON: {text[:300]}")

    @staticmethod
    def _fix_json(text: str) -> str:
        """修复常见的 JSON 格式错误"""
        # 移除尾逗号
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        # 移除注释
        text = re.sub(r"//.*?\n", "\n", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        # 修复未引用的键
        text = re.sub(r"([{,]\s*)([a-zA-Z_]\w*)\s*:", r'\1"\2":', text)
        # 修复单引号为双引号
        text = re.sub(r"'([^']*)':", r'"\1":', text)
        return text

    def __repr__(self) -> str:
        if self.call_budget is not None:
            return f'LLMClient(model={self.model!r}, bounded=True, endpoint=redacted)'
        if self._fallback_triggered:
            return (
                f"LLMClient(主模型={self.model!r} [失败→降级], "
                f"实际使用={self._active_model!r} @ {self._active_base_url!r})"
            )
        return (
            f"LLMClient(model={self.model!r}, "
            f"base_url={self._chat_url!r})"
        )
