"""LeetCUDA pilot 评分器入口。

    python3 -m benchmarks.leetcuda_bench --self-test
    python3 -m benchmarks.leetcuda_bench --check-fixture
    python3 -m benchmarks.leetcuda_bench --from-json results.json
"""
from benchmarks.leetcuda_bench.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
