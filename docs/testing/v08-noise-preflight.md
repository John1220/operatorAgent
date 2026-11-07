# 噪声 preflight：一条命令回答「这个窗口能不能判」

2026-11-07 · v0.8 S3

上一轮八格对照花了 74 分钟、195 次模型调用，最后**没有任何一格**拿到 `valid` 的终评读数；
同日实测主机 load ≈188/192，判决带宽 12.97%，独占卡反而更差（14.39%），CPU 亲和无效。
那次的问题不是代码，是**没有人先问一句「现在测得出差别吗」**。这条命令就是那一句。

## 1. 怎么跑

```bash
# 本地 CPU：只用来验证接线，不能当窗口证据
python -m operator_agent preflight --backend local --reps 5

# 真机窗口：租到卡后、启动任何配对之前
python -m operator_agent preflight --backend ssh --host h20 --reps 5 \
    --planned-effect-pct 5 --json-out /path/to/preflight.json
```

`--planned-effect-pct` 是**你打算判别的效应大小**（百分点）。给了它，回执才能回答
「这次配对看不看得出这么大」；不给就只报散布与档位。

## 2. 回执怎么读

```json
{
  "ok": true, "action": "hold", "predicted_tier": "valid",
  "stats": {"n": 5, "median_ms": 0.2956, "iqr_over_median": 0.0274,
            "max_over_median": 1.0531},
  "empirical_band_pct": 8.22, "nominal_band_pct": 0.1,
  "band_consistent": false, "effective_band_pct": 8.22,
  "planned_effect_pct": 5.0, "min_detectable_pct": 8.22,
  "reasons": ["measured band 8.2206% is wider than the valid table band 0.1000%: use the measured one for this window",
              "planned effect 5.0000% does not clear the 8.2206% band: a pairing would not be able to show it"]
}
```

| 字段 | 含义 |
|---|---|
| `action` | `start` / `hold` / `refuse` —— **hold 是「窗口太吵」，不是「候选不行」** |
| `stats.iqr_over_median` | 同一参照核连跑 `reps` 次的散布；与生产判据同一个统计量 |
| `empirical_band_pct` | `3 ×` 实测散布（与 `NOISE_BAND` 同一取法，不另立系数） |
| `nominal_band_pct` | 该档在标称表里的带宽（valid 0.1% / degraded 0.4% / uninformative 2%） |
| `band_consistent` | 实测带宽是否落在标称表内；`false` 时**以实测为准** |
| `effective_band_pct` | `max(实测, 标称)`，也是 `min_detectable_pct` |

上面这份回执来自**本机 CPU 后端**，实测散布 2.74% ⇒ 有效带宽 8.22% ⇒ 5% 的效应判不出来。
这正是它该说的话：连控制机自己都判不出 5%，就别把 CPU 计时当窗口证据。

## 3. 它证明什么、不证明什么

- **证明**：这个窗口里，同一份代码连跑多次的散布有多大，因此多大的差异**至少**能被看出来。
- **不证明**：任何候选的加速比、任何正确性结论、任何任务的通过与否。
- **必须每窗重跑**：共享主机可以在两次运行之间变吵，而代码一行没改。
- `predicted_tier` 是对**下一次**窗口的预测，不能替代那一次测量。

## 4. 门槛与拒绝

- `reps < 3` 直接拒绝（`bad_reps`，exit 2）：三样本以下的分位数没有意义。
- 任何一次采样抛错或返回非有限/非正值 ⇒ `ok=false`、`action=refuse`：**缺读数不是 0 成本**。
- `mock` / `fake` 后端的回执带 `test_only=true`：那是模拟或控制机 CPU 的计时，不构成真机窗口证据。

## 5. 与上游判据的关系

阈值全部 **import**，不在本模块重写：`DISPERSION_DEGRADED=0.10`、
`DISPERSION_UNINFORMATIVE=0.30`（`decide/measurement.py`）、`NOISE_BAND`（同文件）。
档位只由散布阈值决定（与 `assess()` 的离散度那一条同规则），标称带宽**不许**把档位降级——
`uninformative` 的意思是「完全无法比较」，而「实测带宽很宽」只意味着「小效应看不清」，
后者由 `band_consistent` 与 `effective_band_pct` 承担。