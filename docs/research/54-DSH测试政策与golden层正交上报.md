# 54 · DSH 的测试政策：golden 层的 `PASS` 盖住了四件事

English | （本文暂无英文版）

> 出处：DeepSeek Harness（DSH）本地 vendored 树 `~/repos/github/deepseek-harness`，
> HEAD `47f943859b`（2026-08-13），`docs/testing.md`（整份 **49 行**）。它是
> [`52-共享机落点与静默拆除.md`](52-共享机落点与静默拆除.md) 那份 `defensive-patterns.md`
> 的姊妹文件 —— 后者 `:5` 就写着「Test-tier counterparts (real entry path,
> world-verification, resource ownership) are in [testing.md](testing.md)」。
> 逐字引用，不是转述。落码任务 #73。

## 一、这一篇是怎么撞出来的：跑了一次自家的量具

上一轮把 `defensive-patterns.md` 七条清完（doc 52 §九）之后，顺着它 `:5` 那条交叉引用
读到这份测试政策。要验证它对本仓有没有意义，最便宜的做法不是评论，而是**按它说的去跑
我们自己的对应仪器**：

```console
$ python3 -B tests/golden_regression.py compare
  matmul_mock_py     PASS       112.60     666.72  +492.1%  功能一致, GFLOPS …
  softmax_mock_py    PASS            -          -        -  功能一致 (strict_l3=False, 不判性能)
  layernorm_mock_py  PASS            -          -        -  baseline 本就是 FAIL, 当前也 FAIL (无功能回归)
  relu_mock_py       PASS            -          -        -  baseline 本就是 FAIL, 当前也 FAIL (无功能回归)
汇总: PASS 4 / FAIL 0 / SKIP 0 (共 4 case)                      ← 0.38 秒, 退出码 0
```

`testing.md:19` 那句 *"they catch the 'green unit tests, broken product' class"* 在这里
以另一种形式出现：**报表全绿，而四个 case 里只有一个是活的仪器**。逐条核下去：

| case | 报表说 | 实际是 |
|---|---|---|
| `matmul_mock_py` | PASS | 期望值 112.60，当前读数 666.72 —— **差 4.9 倍**。`DEFAULT_TOLERANCE` 的注释写着「±20% 以内算 OK」，而 `_compare_one` 的窗口是 `baseline*(1-tol) ~ +inf`：**写着双向、实现只罚单向** |
| `softmax_mock_py` | PASS，理由是「`strict_l3=False`, 不判性能」 | 运行时那一维是**开着**的（`strict_l3 = not args.lenient`，默认 True）。真实原因是 `baseline.gflops == 0` 没有分母，而代码把这条路径统一印成了「lenient 关掉了」——**报表谎称一个没关的开关是关的** |
| `layernorm_mock_py` / `relu_mock_py` | PASS「无功能回归」 | baseline 本身就是 FAIL（`L3 FAILED: 无 baseline 参考, 按绝对目标判定: 0.00 GFLOPS (目标 0.00)`）。一个 baseline 为 FAIL 的 golden case **什么都检测不了**（它唯一能检测的是「变好」），是死仪器 |

四件事都叫 `PASS`。这就是 `defensive-patterns.md:9` 说的形态：
*"A result can be several things at once … never nest one flag's report inside another's
branch, or a caller reads a cut-short run as a clean success."* 差别只在 DSH 的例子是
「超时却退出 0」，我们的是「没有仪器却报 4/4」。

## 二、还有一个更糟的：那套期望根本不在版本控制里

`tests/golden_registry.json` 里每个 case 存的是**绝对家目录路径**
（`/Users/…/output/matmul/mock_gpu/20260809_171737_…`）。于是它被 `.gitignore` 第 30 行
整行排除。核了一下历史：

```console
$ git log --all --oneline -- tests/golden_registry.json     # 空
$ git check-ignore -v tests/golden_registry.json            # .gitignore:30
```

**从未被跟踪过**。所以真实缺陷不是「绝对路径泄露进了公开仓库」（我第一反应是这个，
错了 —— 见 §五），而是**方向相反的那件事**：整套 golden 期望只存在这台机器上，
任何人 clone 下来拿到的是一套空期望，`compare` 只会 `exit 2` 说「请先跑 baseline」。
`testing.md:35` 那句「real entry path means the **published artifact**」在这里的对应是：
一条期望如果不可移植，它就不是团队的期望，只是某台机器的回忆。

## 三、吸纳的形状（四条一手规则 → 四个具体动作）

| DSH 原文 | 本仓的动作 |
|---|---|
| `defensive-patterns.md:9` 正交结果各自上报 | `_compare_one` 不再返回一个状态词，而是返回 `(status, facts, reason, dr)`：`FACTS` 六个词（`functional_regression`/`perf_regression`/`perf_drift_up`/`baseline_was_failing`/`fixed_since_baseline`/`no_perf_reference`），状态由事实**推出**。汇总行除 `PASS/REVIEW/FAIL/SKIP` 之外逐条报独立事实数 |
| 同一条：不同结论要求不同动作 | 新状态 `REVIEW` 与 `FAIL` 分家，退出码也分家（**1=代码坏了要修，2=期望过期了要有人签认**）。合成一个码就是把刚拆开的两件事又叠回去 |
| `testing.md:12` *"review every JSONL and expected-output diff"* | 上漂超阈值算 `REVIEW` 而不是 `PASS`：真变快不是回归，但那个期望值已经不代表任何被测物，必须有人重新签认。（本片就把 112.60→708.86 这个 **真实 diff 摊开签认了一次**，见 §四） |
| `testing.md:29` *"Assert untouched files are byte-identical"* | 用例 `test_compare_leaves_the_registry_byte_identical`：跑完整 `compare` 之后断言 registry **字节不变** —— 被评分的一方不许改写评分标准 |
| `testing.md:34` *"A guard only guards if the regression actually fails it … introduce the regression, watch red, revert"* | 与本仓的变异标定是**独立同源的第三方印证**，且它多要求一步：标定产物要能说出「哪条用例为这条变异红」。9 条变异全都有归属（表在测试文件 docstring） |
| `testing.md:21` *"Prefer the real implementation over a mock"* | `_compare_one` 没有外部边界（不碰 LLM/GPU/时钟），所以**直调它就是真实入口路径**，这里不需要任何假件；反过来端到端那条（字节不变）坚持跑真 `compare`，不用假 harness |

外加两个不属于规则但被 §一逼出来的修正：**双向容忍窗**（上漂同样要报），
以及 reason 文案不再谎称 `strict_l3=False`。

## 四、落地实况与一次真实的重新签认

- `tests/golden_regression.py`：`FACTS`/`STATUSES`/`portable()`/双向窗口/两个退出码。
- registry 的路径字段改成**相对 repo 根**（`portable()`：repo 内相对、repo 外保持绝对，
  因为 `../` 形式会随 cwd 漂移）；解除 ignore 并**入库**。
- 入库前必须重跑 `baseline`，因此这次的期望被刷新了一次。刷新前后的完整 diff：

  | case | passed | gflops |
  |---|---|---|
  | `matmul_mock_py` | True→True | **112.5988 → 708.8565** |
  | `softmax_mock_py` | True→True | 0.0→0.0 |
  | `layernorm_mock_py` | False→False | 0.0→0.0 |
  | `relu_mock_py` | False→False | 0.0→0.0 |

  这就是 §一 那个 +492% 的**根因**：旧期望是 2026-08-09 打的，其间 MockGPU 的性能模型
  与本仓的测量链路都动过，而这个数没人复核过。签认的方式不是说「112 是错的」就算了，
  而是让今天的读数带上当天的时间戳与出处指针。
- `tests/test_golden_tier.py` **18 例**；变异标定 9 条 **0 逃逸**（表在文件 docstring）。
- ⚠️ 刷新之后 `compare` 的退出码是 **2**（`layernorm`/`relu` 仍是死仪器、
  `softmax` 仍没有性能读数）。这是**故意留下的**：那三例现在会持续要求有人处理，
  而不是被 `PASS 4` 掩掉。修它们要碰 `verifier.py`（并发会话在途文件），见 §五。

## 五、这一轮**没有**吸纳的，以及为什么

- **不修 `layernorm`/`relu` 的 L3 判据**。它们的失败原因是
  `无 baseline 参考, 按绝对目标判定: 0.00 GFLOPS (目标 0.00)` —— 判据在 `verifier.py`，
  那是并发会话的在途文件（纪律：不编辑、不 stage）。本片做的恰好是**不需要**碰它的部分：
  停止把「检测不了东西的 case」报成好消息。
- **不动 MockGPU 的性能数**。核过 `mock_gpu.py:391`：`gflops = flops / elapsed / 1e9`，
  而 `elapsed` 来自 `time.perf_counter()` —— **golden 层的性能轴是一个墙钟量**。
  实测同一段代码两次跑出 666.72 与 708.86（**6% 散布**）。这与 doc 53 §7.2 刚吸纳进来的
  「墙钟不作判据」直接冲突，也是本仓 #50 那一族。开任务 #74。
  本片因此**刻意不让任何用例断言 gflops 的具体数值**（那是非确定性写死，doc 49 §7.16）。
- **不照抄 DSH 的分层本身**（unit / coverage gate / real-API e2e / snapshot / web）。
  它有 `pnpm` 与 CI 矩阵，我们没有 CI；本仓对应物只有 `MockGPU 全量单测 + 真机驱动 + golden`
  这三层，多引进一层空转的只会再制造一个「算了没人读」。

## 六、我自己这一轮的三处错（按纪律记下来，不回写冻结源）

1. **第一判断是「绝对路径泄露了」** —— 错了。`git log --all` 证明那份 registry
   **从未被跟踪**，所以没有任何东西进过公开历史。真实的缺陷方向相反：期望**不在**版本控制里。
   与 doc 53 §七 那条同一形状：**低估与高估外部已有物一样带错决定**，而这里是
   高估了一个自己造成的风险。
2. **在 `golden_regression.py` 的注释里写了「见 doc 54」** —— 写的时候 doc 54 还不存在。
   这条被自家守卫 `test_repo_hygiene::test_every_reference_points_at_an_existing_doc`
   当堂拦下（`[] != ['tests/golden_regression.py:27 -> docs/research/54']`）。
   同一个失效我在 doc 52 已经记过一次（那处的两条假交叉引用是我手写的），**第二次犯**。
   差别是这次有门拦着 —— 那正是门存在的意义。
3. **中途加了新守卫（可达性）却没回头重推所有变异的期望** —— 标定连红两次
   （M3、M1），两次都是期望记账错而非守卫错。通则：加一条守卫之后，
   **所有**可能命中它的期望集合要整体重推，不是只补新发现的那条。

## 七、状态

| 项 | 状态 |
|---|---|
| 一手取证（`testing.md` 49 行逐字） | ✅ 本会话自己 `Read` 的 vendored 树，HEAD 记在头部 |
| 报表全绿而 4 例里只有 1 例是仪器 | ✅ 已修（正交 facts + REVIEW 态），9 条变异 0 逃逸 |
| 期望文件不在版本控制里 | ✅ 已修（`portable()` + 解除 ignore + 入库） |
| 被评分一方改写期望的风险 | ✅ 有字节不变的哨兵（真跑 `compare` 之后核） |
| `layernorm`/`relu` 死仪器本身 | 🚫 未修 —— 判据在并发会话的 `verifier.py` |
| golden 性能轴是墙钟量（6% 散布） | 🚫 未修，开任务 #74 |
| 真机 | ⏸ 不适用：本片全部改动在 `tests/` 与无 GPU 路径内 |
