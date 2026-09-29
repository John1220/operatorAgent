"""修复取向按「哪一级报的失败」分支 —— 吸纳 CUDA-Harness §4.3.1 (docs/research/55)

外部一手 (arXiv:2609.00058v1, 逐字核过):
    "The refinement branches based on which validation stage reports the failure.
     ... When the kernel fails functional validation, the agent is instructed to
     retreat to a more conservative implementation, prioritizing correctness
     above all else."
    "When optimization is built on a kernel that looks high-performance but is
     functionally flawed, subsequent refinement typically remains trapped in the
     same faulty pattern."

我方的先决事实 (不是推测, 是读来的): `conductor.py:779` 交给 Fixer 的是
`[r for r in verify_results if not r.passed]` —— 只要本轮有 L1/L2 失败就会**顺带**
把 L3 性能失败一起交进来。所以「同一份 prompt 里同时写着『误差超标』和『还不够快』」
是今天就会发生的事, 而 Fixer 的模板在此之前只有一条「修好它」, 从不许可退让。

变异标定 (8 条, 每条只改一处, 全绿=逃逸; 实测 8/8 拦住, 0 逃逸):
    M1 `stage_directive` 恒返回 `_L1_INPLACE`   → 数值/退让/优先级/重试 4 条红
    M2 `stage_directive` 恒返回 `_L2_RETREAT`   → 编译取向 + 词表差集 3 条红
    M3 分支条件 `"L2" in levels` 改成 `"L3"`     → 数值退让、重试、L1+L3 三条红
       (L3 是那个「顺带进来」的级别, 用它当条件等于让性能失败决定取向)
    M4 模板删掉 `{{stage_directive}}`            → 6 条红 (含 #67 的装配契约: 值算了没人读)
    M5 模板把它换成死文字                        → 6 条红
    M6 模板恢复「在编译 (L1) 或功能 (L2) 验证中失败」并删 `{{failure_stage}}`
                                                → 2 条红 (装配契约 + 那句被合写的旧话)
    M7 调用处把级别写死成 `{"L1"}`                → 数值退让、优先级、重试 3 条红
    M8 删掉 `_L2_RETREAT` 的末句「本轮**不要**为此优化」
                                                → 只剩 1 条红 (L2+L3 同框那条) —— 正是
                                                  这句话唯一负责的场合

第一轮跑这张表时**归属是错的**: 脚本只在最后统一复原文件, 于是改模板的 M4/M5/M6
一直留在盘上, 后面两条 fixer 变异的红其实是 M6 的红 (那句把两个占位符一起删掉,
装配契约必然红)。加了逐轮复原 + 复原后逐字节自检才是上面这份: M6 从 4 条红变 2 条,
M7/M8 不再挂着与它们无关的 `test_every_passed_name_appears_in_the_template`。
教训不是「全绿骗人」这一种, 而是**串味道的红会把账记到错的守卫头上** —— 照着它去
加固, 加固的是那条本来不需要加固的 (docs/research/55 §六, 陷阱族第 ㉒ 次)。

运行: python3 -m pytest tests/test_fixer_stage_branch.py -q
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operator_agent.fixer import (  # noqa: E402
    FixerAgent, stage_directive, _L1_INPLACE, _L2_RETREAT,
)
from operator_agent.schemas import (  # noqa: E402
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
    VerificationResult, VerificationLevel,
)

GOOD = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
BROKEN = "import numpy as np\n\ndef kernel(A, B:\n    return A @ B\n"


class _LLM:
    """记录每一次 chat 的 prompt; 每次都回一份能通过预检的代码。"""

    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return f"```python\n{GOOD}```"

    def chat_json(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return {}


def _task():
    return TaskSchema(
        task_id="t-stage",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, max_error=1e-5),
    )


def _res(level, msg):
    return VerificationResult(level=level, passed=False, message=msg, error_log=msg)


def _prompt_for(results):
    llm = _LLM()
    FixerAgent(llm=llm).run(BROKEN, _task(), results)
    return llm.prompts[0]


class TestStageBranch(unittest.TestCase):

    def test_a_numeric_failure_permits_retreating_to_a_conservative_form(self):
        p = _prompt_for([_res(VerificationLevel.L2_FUNCTION, "最大绝对误差 1.2e-1 超出容差")])
        self.assertIn("数值检查器", p)
        self.assertIn("correctness 优先于性能", p)
        # 编译分支那段不许同时出现 —— 两句并存等于没分支 (缺陷族: 一个状态词盖两件事)
        self.assertNotIn("编译器 / 预检", p)
        self.assertNotIn("保持既有的优化策略不变", p)

    def test_a_compile_failure_tells_the_model_to_keep_the_strategy(self):
        p = _prompt_for([_res(VerificationLevel.L1_COMPILE, "SyntaxError: invalid syntax")])
        self.assertIn("编译器 / 预检", p)
        self.assertIn("保持既有的优化策略不变", p)
        self.assertNotIn("correctness 优先于性能", p)
        self.assertNotIn("撤销可疑的激进改动", p)

    def test_a_numeric_failure_outranks_a_performance_one(self):
        # 这就是 conductor.py:779 今天真的会交进来的形状
        p = _prompt_for([
            _res(VerificationLevel.L2_FUNCTION, "最大绝对误差 1.2e-1 超出容差"),
            _res(VerificationLevel.L3_PERFORMANCE, "加速比 0.98 未达标"),
        ])
        self.assertIn("correctness 优先于性能", p)
        self.assertIn("本轮**不要**为此优化", p)
        self.assertNotIn("保持既有的优化策略不变", p)

    def test_the_stage_that_failed_is_reported_rather_than_a_fixed_sentence(self):
        p = _prompt_for([
            _res(VerificationLevel.L1_COMPILE, "SyntaxError"),
            _res(VerificationLevel.L3_PERFORMANCE, "加速比未达标"),
        ])
        self.assertIn("L1, L3", p)
        # 模板里那句被合写的旧话不许以任何形式回来
        self.assertNotIn("以下代码在编译 (L1) 或功能 (L2) 验证中失败", p)
        # 退让只由**数值**失败触发: L1+L3 里没有 L2, 就不该许可退回保守写法
        # (这条同时钉住「分支的条件是 L2 而不是 L3」—— 让性能失败决定取向正是
        #  原文说的那个陷阱的反方向)
        self.assertIn("保持既有的优化策略不变", p)
        self.assertNotIn("correctness 优先于性能", p)

    def test_the_directive_survives_the_retry_feedback(self):
        # 预检重试的 feedback 是**追加**在 base_prompt 后面的, 取向不能被它冲掉
        class _BadThenGood(_LLM):
            def chat(self, messages, **kw):
                self.prompts.append(messages[-1]["content"])
                if len(self.prompts) == 1:
                    return "这次没有代码块"
                return f"```python\n{GOOD}```"

        llm = _BadThenGood()
        FixerAgent(llm=llm).run(BROKEN, _task(), [
            _res(VerificationLevel.L2_FUNCTION, "最大绝对误差超出容差")])
        self.assertGreaterEqual(len(llm.prompts), 2)
        self.assertIn("correctness 优先于性能", llm.prompts[1])

    def test_no_orientation_is_never_an_option(self):
        # 空列表 / 只给了没预料到的级别: 取向必须仍然是一段实文字
        self.assertEqual({_L2_RETREAT, _L1_INPLACE},
                         {stage_directive(set()), stage_directive({"L1"}),
                          stage_directive({"L2"}), stage_directive({"L3"}),
                          stage_directive({"L2", "L3"}), stage_directive({"L1", "L3"})})
        for lv in (set(), {"L1"}, {"L2"}, {"L3"}):
            self.assertTrue(stage_directive(lv).strip(), f"levels={lv} 取向为空")

    def test_the_rendered_prompt_has_no_unsubstituted_placeholder(self):
        p = _prompt_for([_res(VerificationLevel.L2_FUNCTION, "误差超标")])
        self.assertNotIn("{{", p)
        self.assertNotIn("未知", p)   # failure_stage 的兜底值不该出现在真实路径里


if __name__ == "__main__":
    unittest.main(verbosity=2)
