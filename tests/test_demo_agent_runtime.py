"""演示脚本的守卫: `benchmarks/demo_agent_runtime.py` 说得对, 而且今天仍然跑得起来

为什么给一个**演示**配守卫 (本仓「每句 ✅ 都要有可运行的守卫」的同一条):
演示里的每一行读数都是一句判断 ("被拒的候选没到过后端" / "非标量指标进不去")。
判据改名、装配点搬走、字段顺序变了 —— 这些都不会让脚本报错, 只会让它**印出一张
读起来仍然通顺、但已经不成立的图**。那是 I15/I11 那一族 (守卫在不成立的前提下恒绿)
在文档侧的形态: 屏幕上的图比注释更容易被当成证据。

这里只问三件事:
1. 演示六个场景全部跑完, 且**自称的断言**没炸 (夹带 `import torch` 的候选若到了后端,
   `run_loop` 自己会 AssertionError);
2. §21.3 的八段一段不少 (设计原文的八项, 少一段就是上下文被偷偷改窄/改宽);
3. 每一条判据都在屏幕上有一行结论 (六条判据名逐个出现) —— 少一条说明这条判据
   在演示里"被跳过"而读者不会知道。

运行: python3 -m unittest tests.test_demo_agent_runtime -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import demo_agent_runtime as demo  # noqa: E402
from operator_agent import agent_runtime as ar  # noqa: E402


class TestTheAgentRuntimeDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """跑一次, 六个场景的输出留着重用 (每个场景都要跑真循环, 不重复付这个钱)"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo.main()
        cls.out = buf.getvalue()

    def test_it_runs_and_its_own_assertion_holds(self):
        """整脚本跑完 —— 关键是它**没有**抛 AssertionError (那正是它自带的判据)"""
        self.assertIn("场景 1", self.out)
        self.assertIn("场景 4", self.out)
        self.assertIn("场景 6", self.out)
        self.assertIn("**没到过后端**", self.out,
                      "夹带 torch 的候选到过后端了 —— run_loop 的断言应当已经炸了")

    def test_every_design_check_shows_up_with_a_state(self):
        """六条判据名逐个出现 (屏幕上缺一条 = 这条判据被静默跳过了)"""
        after = self.out.split("场景 4")[1]
        for rule in ar.CHECK_RULES:
            self.assertIn(rule, after, f"演示里没有 {rule} 这一行的结论")

    def test_the_eight_context_sections_are_all_rendered(self):
        """§21.3 的八项: 渲染出的 `###` 小节数必须**正好**是八个"""
        sections = [l for l in self.out.splitlines() if l.strip().startswith("### ")]
        self.assertEqual(8, len(sections), f"上下文小节数变了: {sections}")
        for design_item in ("当前契约摘要", "父候选 diff", "最近一次失败分类",
                            "首个错误与误差形态", "相关 profiler 指标",
                            "可用工具列表", "剩余预算", "允许修改的文件"):
            self.assertIn(design_item, self.out, f"§21.3 的「{design_item}」不在渲染里")

    def test_the_repeat_violation_scene_reports_every_round(self):
        """场景 5 (I78): 「重复提交也每轮都被拒」这句读数必须在屏幕上成立

        这一条的靶子是**基线取错对象**那个洞: 演示自己带断言 (被拒轮 == [2, 3] 且后端
        零 torch 载荷), 这里再钉一遍屏幕上的读数 —— 演示的说服力全在「读数与判据同一份」。
        """
        self.assertIn("场景 5", self.out)
        after = self.out.split("场景 5")[1]
        self.assertIn("被拒的轮 = [2, 3]", after)
        self.assertIn("含 'torch' 的 = 0 份", after)
        self.assertIn("dep_baseline_source", after, "洞的修法没有在演示里说出来")

    def test_the_model_authored_scene_reports_who_wrote_the_action(self):
        """场景 6: 「动作是谁写的」必须在屏幕上一轮一轮看得见

        这一条钉的是演示**没有**把生产者侧讲成「能读一个 JSON」: 屏幕上要能看到
        ① 模型写的那句理由逐字出现; ② 越权的键只留痕不采纳; ③ 答不出来时说的是
        「答不出来」而不是「观测结果是空」; ④ 退回 harness 那一档也在 (场景 C 第二份
        回复没带信封)。
        """
        self.assertIn("场景 6", self.out)
        after = self.out.split("场景 6")[1]
        self.assertIn("动作是谁写的=llm", after, "模型写的动作没有出现")
        self.assertIn("动作是谁写的=harness", after, "退回 harness 的那一档没有出现")
        self.assertIn("累加器 dtype 写错了", after, "模型写的理由没有逐字上屏")
        self.assertIn("信封里那些越权的键 (只留痕, 不采纳): ['path']", after)
        self.assertIn("refused_no_gauge", after)
        self.assertIn("这条请求**答不出来**", after,
                      "答不出来那句没有上屏 —— 会被读成「观测结果是空」")
        self.assertIn("误差形态=whole_tensor", after)

    def test_the_two_states_that_are_not_pass_are_named(self):
        """「判不了」与「不适用」必须各自出现 —— 演示的价值就在于把这两档摆出来"""
        self.assertIn("not_applicable", self.out)
        self.assertIn("violation", self.out)
        self.assertIn("不是「没问题」", self.out)


if __name__ == "__main__":
    unittest.main(verbosity=2)