#!/bin/sh
# 跑全部四个回放包：修前必须失败（且与冻结预期一致）、补丁必须过完整 L2。
# 零模型调用；只花 GPU 验证时间。用法：sh benchmarks/repro/run_all_replays.sh [host] [work_dir]
#
# 工作目录在 **/data3** 下（`/` 只剩个位数 GB，见 docs/testing/remote-workdir.md）；
# 远端会自己 mkdir + uv init + 上传脚本，所以换目录不需要预先准备什么。
set -eu
HOST="${1:-h20}"
WORKDIR="${2:-${OPERATOR_REMOTE_WORK_DIR:-/data3/operator-work}}"
PY="${PY:-.venv/bin/python}"
mkdir -p logs
receipts=$(mktemp -d logs/replays.XXXXXX)
echo "回放回执写入 $receipts"
fail=0
for case in fused_ln_compile_20261118 fused_ln_tail_20261118 \
            matmul_strided_20261107 attention_causal_20261109; do
  case_dir="benchmarks/repro/$case"
  if [ -f "$case_dir/historical_fix.py" ]; then
    patch=historical_fix.py
  else
    patch=candidate_fixed.py
  fi
  echo "== $case :: 修前"
  "$PY" -m benchmarks.experiments.v08.failure_replay \
      --case "$case_dir" --source candidate.py \
      --output "$receipts/$case/replay-candidate.json" \
      --host "$HOST" --work-dir "$WORKDIR" || fail=1
  echo "== $case :: 补丁"
  "$PY" -m benchmarks.experiments.v08.failure_replay \
      --case "$case_dir" --source "$patch" --expect pass \
      --output "$receipts/$case/replay-${patch%.py}.json" \
      --host "$HOST" --work-dir "$WORKDIR" || fail=1
done
[ "$fail" -eq 0 ] && echo "ALL REPLAYS OK" || echo "SOME REPLAYS FAILED"
exit "$fail"
