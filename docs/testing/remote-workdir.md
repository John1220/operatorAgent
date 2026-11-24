# 远程工作目录：一律在 `/data3` 下

**2026-11-24 起，H20 上的回放/验证工作目录放在 `/data3`，不再往 `/root` 写。**

| 项 | 值 |
|---|---|
| 默认工作目录 | `/data3/operator-work` |
| 覆盖方式 | `OPERATOR_REMOTE_WORK_DIR` 环境变量，或 `run_all_replays.sh` 第二个参数 / `failure_replay.py --work-dir` |
| 出现在 | `benchmarks/repro/run_all_replays.sh`、`benchmarks/experiments/v08/failure_replay.py` |

## 为什么

H20 的根分区（`/`）只剩个位数 GB，`/data3` 还有 3.3T：

```
/dev/nvme1n1p3  177G  161G  9.2G  95% /        ← /root 在这里
/dev/nvme4n1    3.5T   40G  3.3T   2% /data3
```

一次 `uv` 环境就有 5 G 量级，回放产物再叠上去，`/` 很快就满 —— 满了之后失败形态是
「写一半」「远端任务莫名失败」，比慢更难查。

## 远端会自己准备，不需要预置

`SSHExecutor._setup_uv_env()` 会在工作目录里 `mkdir` + `uv init` + `uv add`，并把
`remote_scripts/` 上传进去。所以**换到一个空路径就能跑**，不需要从旧目录拷任何东西。

## 2026-11-24 清理记录

删除了 `/root` 下已经跑完的任务工作目录（远端无相关进程在跑、证据已回收到仓里；
删前逐个核对路径前缀、拒绝软链，删后 `/` 从 95%/9.2 G 变成 92%/15 G）：

```
5.3G  /root/operator-d2-gpu7-20261117
530M  /root/operator-v08-c5.eJCQQO
940K  /root/operator-r2-fp32.g6XsHy
800K  /root/operator-p2-seed-20261116
780K  /root/operator-o8-a1-161201
700K  /root/operator-p1-20261114
684K  /root/operator-d3-20261114
480K  /root/operator-v08-diagnosis-ab.QQGfoz
480K  /root/operator-v08-fp32-ieee.5qkJCW
380K  /root/operator-v082-spike-20261119
```

`/data3/operator-work` 已建好并验证可写。

**`/root/operator_agent` 保留**：它不是某个任务的产物，而是 `REMOTE_WORK_DIR` 的旧默认值
（`~/operator_agent`）—— 一个 5.3 G 的 uv 环境加 19 个 `profile-profile-*` 目录。
要不要一起清、或者把它挪到 `/data3` 再指过去，等确认后再动。