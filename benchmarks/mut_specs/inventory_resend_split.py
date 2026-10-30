# -*- coding: utf-8 -*-
"""#102 第 3 片 (存量产物里 intent 重发的清点) 的变异表。

跑法:
    python3 -B -m benchmarks.mutation_calibrate \\
        --spec benchmarks/mut_specs/inventory_resend_split.py

十二行全在行为分母：这一片的判据都是「给一棵合成的产物树，看数出来的是哪几个数」，
每行都存在一个输入让差异可观察，所以没有结构行（按 #83, 没有可观察差异的行要写隔离理由）。
**2026-09-30 实跑：基线 28 passed，行为分母 12/12 拦住、结构分母 0/0，零 ESCAPED、零 ANCHOR。**

C9 那一行不是假想病：它是本片**第一版真写过的句子**。第一版 `render()` 里直接断言「多出来的
那些快照目录是一次运行里的不同轮」，而实测 `extra_by_new_round = 0`、`extra_by_same_round = 674`
—— 一个都不是换轮。所以这一行钉的是「被读数推翻的那句解释不许写回来」，它的证人
(`test_the_refuted_explanation_must_not_come_back`) 是一条 `assertNotIn`（I40 那一族：解释不改
任何数值断言，所以必须被单独断言）。

C4/C5 是一对：恒等式 `extra_by_new_round + extra_by_same_round == extra_dirs` 只挡「丢东西」，
不挡「两侧颠倒」(C4) 也不挡「两侧填同一个总数」(C5)。这两条都必须有用例，否则「恒等式成立」
听起来比它实际保证的东西多。

C3/C10 是**量具自己那两个空转分支**：盲区计数不看 `result.json` 在不在 (C3) 就永远印一个像
结论的警告；数到 0 个目录时返回 0 (C10) 就会被 cron 读成「没有重发」。两条都属「全绿可能只证
这条分支没人走」。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "inv": "benchmarks/drivers/inventory_intent_resend.py",
}

TARGETS = [
    "tests/unit/test_inventory_intent_resend.py",
]

ROWS = [
    Row("C1 第三格并进完整产物（「分母之外还有东西」那格消失）",
        "inv",
        "    if names == {SNAP}:\n        return KIND_STAGING\n    return KIND_OTHER",
        "    if names == {SNAP}:\n        return KIND_STAGING\n    return KIND_FULL",
        "存在具体输入可观察: 一棵只有 `code.py` 的半截写入目录 —— 改后被算成完整产物, 于是 "
        "`total` 一格不动、`full` 却比真的有 `result.json` 的份数多。这条分母之所以能说自己"
        "没骗人, 全靠第三格是单独数出来的", True),

    Row("C2 `_read_json` 失败时返回 `{}` 而不是 None（「读不出」被当成「读到一份空产物」）",
        "inv",
        "    except (OSError, ValueError):\n        return None",
        "    except (OSError, ValueError):\n        return {}",
        "存在具体输入可观察: 一份截断的 `result.json` —— 改后 `unreadable_result` 从 1 变 0, "
        "也就是这一格从账上蒸发。它与 #53「`speedup=None` 被印成『没有加速』」是同一件事的"
        "两种方向: 缺证据既不能当肯定证据, 也不能悄悄不算证据", True),

    Row("C3 盲区计数不看 `result.json` 在不在（凡带子目录就记一笔）",
        "inv",
        "        if dirnames and dirpath != root and RESULT in filenames:",
        "        if dirnames and dirpath != root:",
        "存在具体输入可观察: 一棵 `op/gpu/run` 里带 `variants/` 子目录的树 —— 改后 `op` 与 "
        "`op/gpu` 这两个中间层也被数进盲区, 读数从 1 变 3; 而只有叶子产物的那棵树改后从 0 变 2, "
        "于是 `render()` 印出一个**根本不存在的**「少算了 N 份」。这一行钉的是「那条假设自己"
        "有没有被数过」", True),

    Row("C4 两个来源颠倒（把换轮算成同轮、同轮算成换轮）",
        "inv",
        "    extra_by_new_round = len(staging_rounds) - len(staging_ids)\n"
        "    extra_by_same_round = sum(staging_rounds.values()) - len(staging_rounds)",
        "    extra_by_new_round = sum(staging_rounds.values()) - len(staging_rounds)\n"
        "    extra_by_same_round = len(staging_rounds) - len(staging_ids)",
        "存在具体输入可观察: 同一个 task_id 三个 `round_done=1` 的快照目录 —— 改后读成「换轮 "
        "2 个 · 同轮 0 个」, 而真值是 0 与 2。**恒等式在这条变异下依然成立**（两侧相加不变）, "
        "所以「两边相加正好等于它」那句不能当这条的证人, 必须有按方向的断言", True),

    Row("C5 两个来源填同一个总数（回到 doc 60 §三 那种「都是重复」的归因）",
        "inv",
        "    extra_by_new_round = len(staging_rounds) - len(staging_ids)\n"
        "    extra_by_same_round = sum(staging_rounds.values()) - len(staging_rounds)",
        "    extra_by_new_round = sum(staging_rounds.values()) - len(staging_ids)\n"
        "    extra_by_same_round = extra_by_new_round",
        "存在具体输入可观察: 同一条 C4 的树 —— 改后两格都印 2。恒等式这次也**破了** (2+2 != 2), "
        "所以它同时被「按方向的断言」和那条恒等式用例杀掉。C4/C5 一起写是因为单测恒等式会让人"
        "以为拆分已经安全", True),

    Row("C6 快照里取不到的那个轮次被当成第 1 轮",
        "inv",
        "                round_hist[str(rnd)] += 1",
        "                round_hist[str(rnd if rnd is not None else 1)] += 1",
        "存在具体输入可观察: 一份没有 `round_done` 键的旧快照 —— 改后 `round_hist` 里那一档 "
        "`None` 消失、第 1 轮多 1 个, 于是「这一代产物还没有这一格」被读成「它们都停在第 1 轮」"
        "（与 #64「`unknown` 被折进某一态」同族）", True),

    Row("C7 `staging_only` 算成 `len(staging_ids)`（不减交集）",
        "inv",
        '                    "staging_only": len(set(staging_ids) - set(result_ids))},',
        '                    "staging_only": len(staging_ids)},',
        "存在具体输入可观察: 一棵「既写了快照又写了产物」的树 —— 改后 `staging_only` 等于 "
        "`intents`, 读起来像「所有写过快照的运行都没写完产物」, 而真值是 1 个。这一格是那片"
        "更正的核心证据 (存量实测 6,865 / 9,608), 差集写成全量就把结论造反了", True),

    Row("C8 报表把两个分母合成一个「重复总数」",
        "inv",
        '    lines.append("⚠️ 这两个数**不能相加**: 前者是「同一件事被发了几次」(完整产物那一侧), "',
        '    lines.append(f"⚠️ 两处合起来 {f[\'extra_writes\'] + s[\'extra_dirs\']} 个重复目录: '
        '前者是「同一件事被发了几次」(完整产物那一侧), "',
        "存在具体输入可观察: 一棵 `extra_writes=1, extra_dirs=2` 的树 —— 改后印「合起来 3 个"
        "重复目录」, 而「不能相加」那句没了。这正是 doc 60 §三 犯的错被写回报表的样子", True),

    Row("C9 把那句被读数推翻的解释写回来",
        "inv",
        '                 f"「同一份 intent 被重新发过」, 不是「一轮一个目录」"',
        '                 f"「一次运行里的每一轮各占一个目录」"',
        "存在具体输入可观察: 同一 task_id、同一个 `round_done` 的三个快照目录 —— 数值两版完全"
        "相同 (都是「同轮 2 个」), 只有句子的解释不同, 所以**没有任何数值断言杀得掉它**, 只有 "
        "`assertNotIn(\"每一轮各占一个目录\")` 杀得掉。本片的真实红法: 第一版就是这么写的, 而"
        "存量实测 `extra_by_new_round` 是 0", True),

    Row("C10 一个叶子目录都没数到时返回 0（空输入读成「没有重发」）",
        "inv",
        '        print("❌ 一个叶子目录都没数到 —— 那是脚本没跑到该跑的地方, 不是「没有产物」")\n'
        "        return 2",
        '        print("❌ 一个叶子目录都没数到 —— 那是脚本没跑到该跑的地方, 不是「没有产物」")\n'
        "        return 0",
        "存在具体输入可观察: 一个空目录当 `--root` —— 改后退出码 0, 于是 cron/下一轮的人把它读"
        "成「跑过了, 没有重发」。与 #59「续跑不许静默从头重跑」、I39「门没放行 vs 这栏算错了」"
        "同族: 「没原料」必须是缺陷信号而不是一个 0", True),

    Row("C11 取不到 task_id 的快照目录被当成一个新 intent",
        "inv",
        "                else:\n                    staging_no_id += 1",
        "                else:\n                    staging_no_id += 1\n"
        '                    staging_ids["<无 id>"] += 1\n'
        '                    staging_rounds[("<无 id>", "None")] += 1',
        "存在具体输入可观察: 两个 staging 目录、其中一个快照里没有 `task_id` —— 改后 `intents` "
        "从 1 变 2, 于是「重发率」的分母被「取不到」那一格灌水 (存量实测这一格是 10 个)。"
        "`round_hist` 那一档同时从 `None` 挪走, 所以 C6 的证人也会红一条 —— 两行共享一个输入"
        "不是巧合, 是「取不到」这一族只有一个修法", True),

    Row("C12 staging 的判据从「只有快照」放宽成「带快照」",
        "inv",
        "    if names == {SNAP}:",
        "    if SNAP in names:",
        "存在具体输入可观察: 一个同时有 `execution_snapshot.json` 与 `code.py` 的目录 —— 改后被"
        "算成 staging, 而它是半截写入。这一格并进哪一边都会让「10,292 个快照目录」这个数变得"
        "不可比, 而它正是那片更正的分母", True),
]
