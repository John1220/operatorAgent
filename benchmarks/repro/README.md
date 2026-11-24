# 失败回放包（repro）：四个真实故障，一条命令重跑

每个目录都是一份**可独立复跑**的失败样本：修前源码、冻结契约、失败回执、历史补丁、
以及"复现/验收"两个回执。回放器用**生产同一套验证入口**（`operator_agent.app.hitl_tools`），
不是第二套判据。

| 包 | 故障 | 具名失败 | 判据 |
|---|---|---|---|
| [fused_ln_compile_20261118](fused_ln_compile_20261118/README.md) | `tl.arange` 的 bound 不是 2 的幂（`NBLK=6`） | 编译期 `CompilationError` line 30 col 11 | L1 过 / L2 编译失败 |
| [fused_ln_tail_20261118](fused_ln_tail_20261118/README.md) | 补齐 lane 计入方差 + 内层 stride 未读 | `1x33` 误差 9.96e-02 | 32 条 case |
| [matmul_strided_20261107](matmul_strided_20261107/README.md) | 写死连续寻址 | `auto-k4-m32-n32-fp16-random-strided` 误差 9.69140625 | 15 条 case |
| [attention_causal_20261109](attention_causal_20261109/README.md) | 掩码乘在 softmax 之后 | 5 条 causal 误差 1.95e-03（匹配率仍 0.9999） | 8 条 case |

## 一次完整的验收示例（`1×33`）

在仓库根目录运行 `sh benchmarks/repro/run_all_replays.sh`，可复现四个故障并验证对应补丁的完整 L2。
每次批量回放将新回执写入 `logs/replays.*/`。以下示例用 `1×33` 展示如何额外取得独立终评。

```bash
mkdir -p logs
receipts=$(mktemp -d logs/replays.XXXXXX)

# 1) 复现：契约哈希、修前源码哈希、级别、具名用例与误差四项都要对上
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_tail_20261118 \
  --source candidate.py --host h20 --work-dir /root/operator-p2-seed-20261116 \
  --output "$receipts/replay-candidate.json"
#   → passed: true  (L1=true / L2=false, 1x33 max_error=0.099609375)

# 2) 补丁：完整 L2 32/32
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_tail_20261118 \
  --source candidate_fixed.py --expect pass --host h20 \
  --work-dir /root/operator-p2-seed-20261116 \
  --output "$receipts/replay-candidate_fixed.json"
#   → passed: true  (L2=True cases=32/32)

# 3) 独立终评：同源码再取预留的 purpose=final 回执
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_tail_20261118 \
  --source candidate_fixed.py --expect pass --final --host h20 \
  --work-dir /root/operator-p2-seed-20261116 \
  --output "$receipts/replay-candidate_fixed-final.json"
#   → passed: true  (... independent final L2=True)
```

各个步骤的回执在 `$receipts`，详细的验证产物在样本目录的 `run/`。

## 对账（迁移/判据改动之后必须做）

重跑的回执要与**提交进仓的那份**逐字段相同，否则「重放成立」这句话就不再成立：

```bash
python -m benchmarks.repro.compare_replays --fresh logs/replays.XXXXXX
# → {"ok": true, ...} / 非零 + 每条差异
```

不比较的只有三个「运行身份」字段（`evidence` / `run_dir` / `final_evidence`）——
它们描述这一次的产物落在哪；其余字段（契约哈希、源码哈希、级别、判定、编译失败、
final 级别与 purpose）必须相等。

## 回放器判什么

`benchmarks/experiments/v08/failure_replay.py`：

1. **契约哈希**必须等于 `case.json` 的 `contract_hash`；
2. **修前源码哈希**必须等于 `origin.source_hash`（回放 `candidate.py` 时强制；
   回放补丁时按 `--expect pass` 判 L2）；
3. **级别**要与 `expected.levels` 逐项相等；
4. **具名用例**：`passing_cases` 必须通过，`failing_cases` 必须失败且误差在容差内相等。

任一项对不上就判"回放不成立"并返回非零——不允许"跑了个大概"。

## 运行前提

- H20 上有可用的 SSH（`~/.ssh/config` 的 `h20` 别名）；
- 远端工作目录在 **`/data3/operator-work`**（`/root` 所在根分区只剩个位数 GB）。远端会自己
  `mkdir` + `uv init` + 上传脚本，换目录不用预先准备；覆盖它用 `OPERATOR_REMOTE_WORK_DIR`
  或第二个参数（见 `docs/testing/remote-workdir.md`）；
- 回放**不需要模型**：全程零模型调用，只花 GPU 验证时间；
- 回放格目录必须在仓库之外（本仓 `output/` 符号链接即可），否则 DSH 沙箱会挡 `realpath`。

## 与修复对照的关系

这些包只回答"这个故障能不能稳定复现、补丁能不能过独立验收"。
"普通 DSH 与插件谁修得更省"是另一件事，见
[修复对照预登记](../../docs/testing/v08-repair-ac-preregistration.md)：两例都是 `both_qualified`，
目前只验收了闭环，没有测出差别。
