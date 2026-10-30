"""「§21.1 动作信封：LLM 交上来的那一份」那张变异表的**可执行**形态

靶子是 `tests/unit/test_llm_action_producer.py`（三态 / 切段 / 模型写意图 / harness 写数字与
路径 / 三个 ask 种类的响应与上界 / 续跑的取舍 / 模板与词汇表不漂），外加
`tests/unit/test_agent_runtime.py`（§21.4 的判据本身）与 `tests/unit/test_fixer.py`（投影入口）。

为什么单独一张表: 这一片的主张不是「能读一个 JSON」，而是**判据第一次遇到一个真正
不可靠的生产者**。每行拆掉一层保护，看守卫是不是真的在守：

* 拆「读」——三态压成一个布尔、两段围栏当成一段、把信封正文留在候选文本里；
* 拆「判」——表外种类直接判死、不点误差形态也放行、模型自己填的数字当真；
* 拆「响应」——询问不响应、上界拿掉、答复不进 prompt、把「答不出来」说成「观测是空」；
* 拆「单源」——投影断开、信封跨轮泄漏、判据措辞各写一份、模板不再教那个围栏。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/llm_action_producer.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ar": "operator_agent/agent_runtime.py",
    "fx": "operator_agent/fixer.py",
    "cu": "operator_agent/conductor.py",
    "pm": "operator_agent/prompts/fixer_repair.md",
}

TARGETS = ["tests/unit/test_llm_action_producer.py", "tests/unit/test_agent_runtime.py",
           "tests/unit/test_fixer.py"]

ROWS = [
    # ── 读: 三态与切段 ───────────────────────────────────────────
    Row("M1 没有围栏段也算「读出来了」 (absent → parsed)", "ar",
        """    if not blocks:
        return ActionEnvelope(status=ENVELOPE_ABSENT,
                              reason=f"回复里没有 ```{ENVELOPE_FENCE} 段"), body""",
        """    if not blocks:
        return ActionEnvelope(status=ENVELOPE_PARSED, kind="analyze_dependencies"), body""",
        "**具体输入**: `parse_action_envelope(\"```python\\nimport numpy\\n```\")` —— "
        "接线前模型就是这么回话的（不交信封）。改了之后它变成一条**种类为 "
        "analyze_dependencies 的动作** ⇒ `test_no_envelope_is_absent_not_an_error` 红，"
        "并且它会被判据当成「一个不改候选的请求」走响应那条路（白烧一次 Fixer）"),
    Row("M2 两段围栏当成一段 (不猜哪一段是它)", "ar",
        """    if len(blocks) > 1:
        return ActionEnvelope(status=ENVELOPE_MALFORMED,
                              reason=f"回复里有 {len(blocks)} 段 "
                                     f"```{ENVELOPE_FENCE} —— 不猜哪一段是它"), body""",
        """    if len(blocks) > 1:
        pass""",
        "**具体输入**: 两份信封拼在一起（`_ask(\"explain_failure\") * 2`）—— 取哪一段都是"
        "猜：`re.findall` 的第一段与最后一段在今天**恰好**相等，所以行为层的可观察差别是"
        "状态从 malformed 变成 parsed ⇒ `test_malformed_is_separate_from_absent` 的"
        "「两段」那一格红"),
    Row("M3 表外种类直接判死 (parsed → malformed)", "ar",
        """    return ActionEnvelope(
        status=ENVELOPE_PARSED, kind=kind.strip(), rationale=rationale.strip(),""",
        """    return ActionEnvelope(
        status=(ENVELOPE_PARSED if kind.strip() in ACTION_KINDS else ENVELOPE_MALFORMED),
        reason=("种类不在闭合词汇表里" if kind.strip() not in ACTION_KINDS else ""),
        kind=kind.strip(), rationale=rationale.strip(),""",
        "**具体输入**: `_ask(\"teleport\")` —— 模型说了一件不被允许的事。判死之后产物里"
        "只剩「格式错」，模型得到的反馈从「这个种类不许」变成「你的信封不合法」"
        "（它下一次会去猜格式而不是换动作）⇒ "
        "`test_an_unknown_kind_is_parsed_and_left_to_the_judge` 红"),
    Row("M4 信封正文留在候选文本里 (切段拆掉)", "ar",
        """    blocks = _ENVELOPE_RE.findall(raw)
    body = _ENVELOPE_RE.sub("", raw)""",
        """    blocks = _ENVELOPE_RE.findall(raw)
    body = raw""",
        "**具体输入**: 信封 + 一份代码的回复，且信封里写着一句 `import os`（代码提取的兜底"
        "路径正是按 `import ` 找起点）⇒ 候选文本里多出信封那一整段 ⇒ "
        "`test_the_envelope_is_cut_out_before_code_is_extracted` 红（两个读者共用一个"
        "正则的代价就是这条）"),
    # ── 判: 模型写意图, harness 写数字与路径 ──────────────────────
    Row("M5 不点误差形态的数值修复也放行 (§21.2 那半条拆掉)", "ar",
        """    if action.kind == KIND_REPAIR_CORRECTNESS and ctx.error_shape:
        if ctx.error_shape not in (action.rationale or ""):""",
        """    if action.kind == KIND_REPAIR_CORRECTNESS and ctx.error_shape:
        if False:""",
        "**具体输入**: 数值失败那一轮，模型交来 `{kind: repair_correctness, rationale: "
        "\"我把索引改了一下\"}`（不带 `whole_tensor`）⇒ 门放行、那份候选真的上了后端 ⇒ "
        "`test_a_rationale_without_the_error_shape_is_refused` 红。这一行拆的正是"
        "「接线后这条判据第一次真的会红」——接线前理由由 harness 拼装，它永不失败"),
    Row("M6 模型自己声明的 token 预算顶掉上限", "cu",
        """            token_budget=self._llm_tokens_budget_claim(round_action),""",
        """            token_budget=(envelope.declared_token_budget
                          or self._llm_tokens_budget_claim(round_action)),""",
        "**具体输入**: 信封里写 `token_budget: 1` ⇒ 登记进动作的预算从 4096 变成 1 ⇒ "
        "`test_the_harness_still_owns_the_path_and_the_budget` 红。这一行是"
        "「不许让被判的人填判据里的数」那条纪律的可执行形态"),
    Row("M7 写哪个文件由模型说了算 (路径越权)", "cu",
        """        patch: Optional[Patch] = Patch(path=candidate_path(self.run_root, ext),
                                       new_source=current_code,""",
        """        patch: Optional[Patch] = Patch(
            path=(envelope.unknown_keys and envelope.unknown_keys[0] == "path"
                  and "" or "") or candidate_path(self.run_root, ext),
            new_source=current_code,""",
        "**具体输入**: 信封里塞一个 `path` 键（表外键，合法解析但**不许**被采纳）⇒ "
        "改法本身不产生差别（因为它仍然回落到 `candidate_path`），**行为层所以不可区分**"
        "—— 这一行进结构分母，拦住它的是 `unknown_keys` 断言与 §22.1 的路径构造点守卫"
        "（见 `run_layout_doors.py` M7/M8）。写在这里是为了让「越权键被静默丢掉」这件事"
        "有第二处留痕", behavioral=False),
    # ── 响应: 三个 ask 种类 ─────────────────────────────────────
    Row("M8 询问类动作不响应 (当成普通修复)", "cu",
        """                        if _outcome.asked:
                            # 模型交的是**不改候选**的意图 (解释 / 请求观测): 这一轮没有候选。""",
        """                        if False:
                            # 模型交的是**不改候选**的意图 (解释 / 请求观测): 这一轮没有候选。""",
        "**具体输入**: 第 1 轮修复步交一封 `explain_failure`（不改候选）⇒ 不响应时"
        "`_outcome.code` 为空 ⇒ 本轮**直接停**（`ask_turns_exhausted`），模型一次"
        "解释的机会都没有 ⇒ 三个 ask 用例（`ask_turns` / `ask_outcome` / 那句答复进 "
        "prompt）全红。这一行是「七个动作种类里那三个终于有生产者」的可执行形态"),
    Row("M9 上界拿掉 (没有补丁也继续下一轮)", "cu",
        """                    if not _outcome.code:""",
        """                    if False:""",
        "**具体输入**: 连续两次都是 `explain_failure` ⇒ 没有候选却把 `pending_fix = \"\"` "
        "带进下一轮 ⇒ 停止理由不再是 `ask_turns_exhausted`、循环继续烧 token ⇒ "
        "`test_asking_twice_stops_with_a_reason` 红（并会连带 `test_stop_reason` 的出口"
        "词汇差集 —— 那一档从此没有写者）"),
    Row("M10 答复不进 prompt (响应了但模型看不到)", "cu",
        """            ask_answer=answer,
        )
        record["second_envelope"] = second.envelope.as_dict()""",
        """            ask_answer="",
        )
        record["second_envelope"] = second.envelope.as_dict()""",
        "**具体输入**: `request_profiler` ⇒ 第二份 Fixer prompt 里不再有"
        "「这条请求**答不出来**」⇒ `test_every_ask_kind_is_answered_and_then_a_patch_is_"
        "asked_for` 的那句 `assertIn(fragment, ...)` 红。注意「动作信封那段静态协议还在」"
        "—— 只有逐字找那句答复才拦得住，找静态段是一条空转的断言"),
    Row("M11 「答不出来」说成「收到了、按已有证据来」", "ar",
        """    missing = [tool for tool in KIND_REQUIRED_TOOLS.get(kind, ()) if TOOLS.get(tool) is None]
    if not missing:
        return \"\"""",
        """    missing = [tool for tool in KIND_REQUIRED_TOOLS.get(kind, ()) if TOOLS.get(tool) is None]
    if missing:
        return \"\"""",
        "**具体输入**: `ask_unavailable_reason(\"request_profiler\")` 从「今天没有实现物」"
        "变成空串 ⇒ 响应落到 `accepted_without_running_it`（I65 那条纪律的反面："
        "「观测是空」与「观测没有量具」在产物里长得一样）⇒ "
        "`test_the_profiler_ask_is_refused_for_lack_of_a_gauge` 与同源那条红"),
    # ── 单源: 投影 / 泄漏 / 措辞 / 模板 ─────────────────────────
    Row("M12 `run` 不再走 `run_with_action` (投影断开)", "fx",
        """        outcome = self.run_with_action(""",
        """        outcome = self._run_with_action_unused(""",
        "**具体输入**: 同一个模型回复分别走 `run` 与 `run_with_action` —— 两条路要给出"
        "同一个 `(code, summary)` ⇒ `test_an_ask_that_comes_with_code_loses_the_code` "
        "的第二半红（`AttributeError` 也会让 `tests/unit/test_fixer.py` 那条入口结构守卫红）。"
        "这一行保证「接线有没有改修复行为」这个问句永远答得出来"),
    Row("M13 信封跨轮泄漏 (消费时不清)", "cu",
        """                _model_envelope = self._pending_action_envelope
                self._pending_action_envelope = None""",
        """                _model_envelope = self._pending_action_envelope""",
        "**具体输入**: 第 2 轮的修复交了信封且通过 ⇒ 第 3 轮（Coder 生成轮）会把第 2 轮"
        "那份意图再当成自己的 ⇒ `test_the_envelope_does_not_leak_into_a_later_round` 红"
        "（第 3 轮的 `action_envelope.source` 从 harness 变成 llm）。这一行同时会让"
        "「读点恰好 1 处」那条 AST 计数守卫红 —— 两个读者都不该被放过"),
    Row("M14 判据措辞各写一份 (回到「今天不可用」)", "ar",
        """            problems.append(ask_unavailable_reason(action.kind)
                            or f"{action.kind} 需要的工具 {tool!r} 今天没有实现物")""",
        """            problems.append(f"{action.kind} 需要的工具 {tool!r} 今天不可用")""",
        "**具体输入**: 一个**只靠 kind** 请求 profiler 的动作（`requested_tools=()`）⇒ "
        "拒绝理由不再含「没有实现物」⇒ `test_the_capability_judgement_has_one_source` "
        "红。原措辞把「profiler 还没做」与「这台机器不支持它」压成一句 —— 这一行拆的正是"
        "那个三态缺陷"),
    Row("M15 模板不再教那个围栏标签", "pm",
        "## 动作信封", "## (这一段今天先不写了)",  # noqa: E501
        "**具体输入**: 渲染后的 Fixer prompt —— 模型不知道要交信封（`agent-action` 三个"
        "字都不出现）⇒ `test_the_template_teaches_the_same_fence_tag` 与词汇表那条红。"
        "这一行保证「解析器认的围栏」与「模板教的围栏」不会各改一半"),

    # ── 结构行: 今天行为层不可区分 ──────────────────────────────
    Row("M16 信封属性多一个写点 (谁清它说不清)", "cu",
        """        self._pending_action_envelope: Optional[ActionEnvelope] = None""",
        """        self._pending_action_envelope: Optional[ActionEnvelope] = None
        self._pending_action_envelope = None   # 多出来的写点: 没人读, 也没人说得清谁清它""",
        "**行为层不可区分**（多出来的那格没人读），拦住它的只有 AST 计数守卫"
        "（`test_the_pending_envelope_is_armed_once_and_read_once`：写点要恰好 4 处）。"
        "⚠️ 第一版这一行写成「多一个**别的名字**的属性」⇒ **逃逸**了: 守卫数的是同一个属性名的"
        "写点, 换个名字加一处没人会响 —— 这正是「逃逸的行说明证人写错了」那一族的现场。"
        "写在这里是因为本仓 I77/I78 两个缺陷的形状都是「一个写进去、没人负责清的留痕」",
        behavioral=False),
]