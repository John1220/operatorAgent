#!/bin/sh
# 跑全部四个回放包：修前必须失败（且与冻结预期一致）、补丁必须过完整 L2。
# 零模型调用；只花 GPU 验证时间。用法：sh benchmarks/repro/run_all_replays.sh [host] [work_dir]
set -eu
HOST="${1:-h20}"
WORKDIR="${2:-/root/operator-p2-seed-20261116}"
PY="${PY:-.venv/bin/python}"
fail=0
for case in fused_ln_compile_20261118 fused_ln_tail_20261118 \
            matmul_strided_20261107 attention_causal_20261109; do
  echo "== $case :: 修前"
  "$PY" -m benchmarks.experiments.v08.failure_replay \
      --case "benchmarks/repro/$case" --source candidate.py \
      --host "$HOST" --work-dir "$WORKDIR" || fail=1
  echo "== $case :: 补丁"
  "$PY" -m benchmarks.experiments.v08.failure_replay \
      --case "benchmarks/repro/$case" --source historical_fix.py --expect pass \
      --host "$HOST" --work-dir "$WORKDIR" 2>/dev/null || \
  "$PY" -m benchmarks.experiments.v08.failure_replay \
      --case "benchmarks/repro/$case" --source candidate_fixed.py --expect pass \
      --host "$HOST" --work-dir "$WORKDIR" || fail=1
done
[ "$fail" -eq 0 ] && echo "ALL REPLAYS OK" || echo "SOME REPLAYS FAILED"
exit "$fail"
