# S6 准备：S4 脱敏案例与离线演示

状态：**S4 CLI/core 的补充演示材料**。数据来自 2026-10-27 S4 的真实 CLI / H20 GPU 6 / DeepSeek 小批记录，不是新一次 GPU 测量。本轮 S6 的主要插件案例和离线验收改用 [S5 脱敏插件案例包](../../output/plugin_completion/20261030-s5-closeout-v1/shareable/s5-plugin-cases.zip)及[三分钟演示脚本](dsh-operator-s5-resume-demo-20261030.md)；整体交付见 [S6 验收](../status/dsh-operator-s6-acceptance-20261030.md)。

交付：[脱敏案例 ZIP](../../output/plugin_completion/20261027-s4/portable-case-bundle-v2.zip)，SHA256 `692e56d664ce61d4910746d5534ba48a7f1c50dbb41d391c0dbde67c8b1d484a`；展开目录为 [portable-case-bundle-v2](../../output/plugin_completion/20261027-s4/portable-case-bundle-v2/)。`output/` 被 Git 忽略，须单独备份 ZIP；原始私有记录仍在 `gpu6-8004/`，没有被覆盖。

## 讲解顺序（约 3 分钟）

1. **错误被拦住**：打开 `cases/layernorm-tail/initial.py` 与 `initial-l2.json`。初始核忽略列 stride，输入 `X=8x32`、非连续布局时，事件记录 256/256 元素错误，输入布局保真校验通过；整体完整 L2 不通过。这是一份历史实测投影，不是刚刚重跑。
2. **修复后完整验证**：打开 `selected.py` 与 `final-l2.json`。修改分别传递输入、输出列 stride；同一契约的 12 个 L2 case 均通过。模型账本 `calls.json` 显示 Fixer 1 次调用，服务端返回输入 7,076、输出 585、合计 7,661 tokens。只对该有限 suite 主张正确，不推断全域正确。
3. **不合格读数不算性能成绩**：打开 `final-l3.json`。该候选的有效性是 `uninformative`，launch 地板约占 78.2%，`can_gate=false`；性能结论为未知，没有合格首次测量可用于独立确认。展示时不把探索加速比当作达标成绩。

其余三项 MatMul / Softmax 生成案例也各自保留首个 L2、最终 L2、最终 L3、相关候选和 API usage 投影。汇总结果：最终完整 L2 4/4；性能结果未知 4/4；14 次模型调用，API 报告合计 94,966 tokens；实际账单金额未知。

## 离线使用与边界

解压 ZIP 后，在解压目录执行：

```sh
python3 verify.py
```

验证器只用 Python 标准库读取文件，不 import 或执行候选，不联网，不调用模型或 GPU。它检查文件清单与 SHA256、源码/契约绑定、L2 case 和 L3 资格、首次 L2 时间与调用完成时点、各模型调用的 API usage 与全包汇总。已在仓库外临时目录解压运行；故意修改候选、修改汇总并同步修改清单、以及向汇总注入 URL 均被拒绝。扫描对象为导出的数据文件；验证脚本自身包含检测规则字面量，因此不对自身做敏感词匹配。

安全边界：事件和模型账本采用字段白名单投影，不复制端点、个人路径、凭据、原始 prompt/回复或执行日志；候选源码逐份复制并与原事件的源码哈希核对。投影保留原始事件/调用文件的 SHA256，便于持有私有档案的人回查，但此包没有数字签名，哈希只证明包内一致性，不证明生成者身份或原始记录真实性。它不能复演 GPU 正确性和性能、不能还原完整算子运行环境，也不能代表完整 S6 的可携带原始证据包。

重建本包（不覆盖既有目录）：

```sh
python3 benchmarks/s4_case_bundle.py \
  --source output/plugin_completion/20261027-s4/gpu6-8004 \
  --output output/plugin_completion/20261027-s4/NEW_BUNDLE_DIRECTORY
```

构建时先核对私有证据清单、冻结计划、每个候选源码与事件/契约身份；只允许在新目标目录写入，构建后先运行包内验证器。私人原始证据仍可能包含路径和连接元数据，不能因为此脱敏投影可分享就直接分享原目录。
