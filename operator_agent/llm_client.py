"""
LLM 客户端 - 统一的 LLM 调用封装
支持 OpenAI 兼容 API (私有部署模型 / DeepSeek)
支持主模型失败后自动降级到 fallback 模型
通过环境变量配置:
  LLM_BASE_URL  - API 地址 (默认 http://152.136.252.171:8092/v1)
  LLM_API_KEY   - API Key (可选)
  LLM_MODEL     - 模型名称 (默认 Qwen/Qwen3.5-9B)
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
from dataclasses import dataclass, field
from typing import Optional


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


class CostTracker:
    """LLM 调用账本: 累计 tokens / 估算费用 / 调用次数, 支持按 caller 分解"""

    def __init__(self):
        self.records: list[_LLMCallRecord] = []

    # ----- 写入 -----
    def record(self, *, model: str, prompt_tokens: int, completion_tokens: int,
               usage_source: str, caller_tag: str = "Unknown") -> None:
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
        self.records.append(_LLMCallRecord(
            ts=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=round(cost_usd, 6),
            caller_tag=caller_tag,
            usage_source=usage_source,
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

    def breakdown_by_caller(self) -> dict:
        """按 caller 分组汇总 {caller: {calls, prompt, completion, total, cost_usd}}"""
        out = {}
        for r in self.records:
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

    def summary(self) -> dict:
        """返回结构化账单摘要 (供 Conductor 日志输出)"""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "breakdown_by_caller": self.breakdown_by_caller(),
            "n_estimated_usage": sum(1 for r in self.records if r.usage_source == "estimated"),
        }

    def pretty(self) -> str:
        """人可读的账单摘要 (终端 print 用)"""
        s = self.summary()
        lines = []
        lines.append(f"LLM 账单摘要: 调用次数={s['total_calls']}, "
                     f"tokens={s['total_prompt_tokens']}in / {s['total_completion_tokens']}out "
                     f"(合计 {s['total_tokens']}), 估算≈${s['total_cost_usd']:.4f}")
        if s["n_estimated_usage"]:
            lines.append(f"  * 其中 {s['n_estimated_usage']} 次 API 未返回 usage, 使用字符估算")
        if s["breakdown_by_caller"]:
            lines.append("  按调用方分解:")
            for caller, b in sorted(s["breakdown_by_caller"].items(), key=lambda kv: -kv[1]["cost_usd"]):
                lines.append(
                    f"    - {caller}: {b['calls']} 次, "
                    f"{b['prompt_tokens']}in / {b['completion_tokens']}out, "
                    f"≈${b['cost_usd']:.4f}"
                )
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
        timeout: int = 120,
        fallback_model: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get(
            "LLM_BASE_URL", "http://152.136.252.171:8092/v1"
        )).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-9B")
        self.temperature = temperature
        self.max_tokens = max_tokens
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
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

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
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
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
        )

        if content is None or (isinstance(content, str) and content.strip() == ""):
            if finish_reason == "length":
                raise RuntimeError(
                    f"LLM 回复被截断 (finish_reason=length): "
                    f"思考过程耗尽 max_tokens={max_tokens or self.max_tokens}，"
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

    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
        caller_tag: str = "Unknown",
    ) -> dict:
        """聊天并尝试将回复解析为 JSON dict"""
        text = self.chat(messages, temperature, max_tokens, caller_tag=caller_tag)
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
        if self._fallback_triggered:
            return (
                f"LLMClient(主模型={self.model!r} [失败→降级], "
                f"实际使用={self._active_model!r} @ {self._active_base_url!r})"
            )
        return (
            f"LLMClient(model={self.model!r}, "
            f"base_url={self._chat_url!r})"
        )
