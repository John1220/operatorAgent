# 审查后修掉的三处 verify 路径阻塞（2026-11-07）

来源：对 `b2f3e6c..80a9156` 的审查（5 条）。本文件记录其中 **#1 / #2 / #3** 三条的
复现、修法与核对。#4（FP32 预登记）见 [候选余量](candidate-headroom-20261107.md)，
#5（`docs/status` 布局守卫）通过把进度入口移入 [v0.8 状态目录](README.md) 处理。

## 0. 为什么这三条现在才修

C5 停在**身份门**（见 [C5 卡点](c5-identity-blocker-20261107.md)），候选核从没被执行过，
所以身份门之后的缺陷在真机上看不见。审查指出：一旦选修法 A（把远端 torch pin 回受信版本），
这三处会一次撞上。三条都在本地复现并修好，各自做了**变异核对**——把源文件换回 `HEAD`
跑新测试确认红，再字节级复原（`RESTORED byte-identical: True`）。

| # | 位置 | 修前后果 | HEAD 上新测试 |
|---|---|---|---|
| 1 | `remote_scripts/run_kernel.py` | 正确候选被判失败 | 2 红 |
| 2 | `operator_agent/decide/correctness.py` | 整个 L2 中止 + 跳过审计写入 | 16 红 |
| 3 | `operator_agent/decide/anticheat.py` | 原生实现进入性能对照 | 15 红 |

## 1. 输出设备检查硬取 `X`

`run_verify` 的受信任参考分支用 `candidate_inputs["X"].device` 判输出设备。attention 的输入是
`Q/K/V` + 按值直传的 `causal` 标量，没有 `X` ⇒ `KeyError`，被外层吞成
`Remote verification error: KeyError: 'X'`，**把一个正确候选判成失败**。

复现方式：只 stub 身份门（它要真机库指纹，本地不可能过），其余走真实代码路径。

**不能**改成比 `run_verify` 里那个 `device` 变量：它是 `torch.device("cuda")`（index=None），
张量回来的是 `cuda:0`，两者 `!=` —— 那会在真机上判否每一个候选。本地 CPU 看不出来，
因为 `cpu == cpu`。所以改为从**实际输入张量**收集设备集合，判 `actual.device` 是否在其中。

实测（两个 profile 都覆盖）：

- 全零 Q/K/V 下 SDPA 也是全零 ⇒ attention 候选现在真的判 `success=True`，不只是"没炸"；
- layernorm 的设备门同样放行（它后面另有一道 CUDA-only 参考门，本地到不了终判，
  所以那一条只断言设备门已放行，不断言终判）；
- 负例两个 profile 都仍被拒：输出换设备（`meta`）、输出换精度（fp32）。

## 2. pattern/layout 变换施加到了标量上

`CorrectnessCase.materialize()` 对 `arrays` 里每个值施加 pattern 与 layout 变换。8 条里
`strided` / `zeros` / `large` 三条把 bool `causal` 当数组取 `.ndim` / `.shape` ⇒ `AttributeError`。

**后果比"三条造不出输入"重**：`verifier._verify_function` 只捕
`(BenchCaseError, ValueError, CapabilityMissing)`，`_timed_level` 也没有兜底 ⇒ 异常逃出
整个套件循环、中止 L2，并跳过 `_timed_level` 里那次审计写入。

修法：只对 `np.ndarray` 施加变换，标量原样透传。标量是 case 身份的一部分（按值直传且进 hash），
变换它等于悄悄改掉被验的语义。**没有**去扩大那个捕获元组——把 `AttributeError` 降级成
unknown 会掩盖真缺陷。

为什么此前没被测到：`tests/unit/test_correctness_suite.py` 的 `case()` 助手不带标量，
已有的 pattern/layout 参数化测试从没走到这条路。所以回归钉在**既有读者**上：

- 5 pattern × 3 layout 网格，同时断言张量**真的**被变换（否则"整个循环都跳过"也能绿）；
- 逐条走 8 条 attention case，核标量值与类型、张量形状/dtype、布局与 pattern 是否落地；
- 三条曾坏的 case 再物化一次必须逐位相同（verifier 的重复检查会再物化）。

变异核对里 `contiguous-random` 那一格在 HEAD 上仍是绿的——该组合本来就不施加任何变换，
标量从没被碰到。这是应该的，不是漏钉。

## 3. `torch.ops` 是通用后门，且不止 attention

[attention 设计 §15.2](../../design/v08/attention-family.md) 那张表把 `torch.ops.aten`
记成"已覆盖"，但当时只覆盖 `import torch.ops.aten as A` —— 那个写法在 torch 2.x 上直接
`ModuleNotFoundError`，测试绿只证了"这条分支没人走"。**真能执行**的两种写法全部漏过：

| 写法 | 修前静态 | 修前运行时 |
|---|---|---|
| `torch.ops.aten.scaled_dot_product_attention.default(...)` | 放行 | 执行成功 |
| `from torch.ops import aten` | 放行 | 执行成功 |

而且不是 attention 专属：`aten.mm.default` / `aten.matmul.default` / `aten._softmax.default`
同样绕过 ⇒ 这是**任何算子**都能用的后门。

四处一起改：

1. `_BLOCKED_MODULE_PATHS` 从 `torch.ops.aten` 放宽到命名空间根 `torch.ops`；
2. `_path_violation` 拦下任何解析到 `torch.ops.*` 的调用路径；
3. `_resolve` 补 getattr 折叠 —— `getattr(torch,'ops').aten.mm.default(...)` 里 getattr
   结果是属性链的**底座**而不是被调函数，原来整条链解析不出就静默放行（这一条是修的过程中
   自己测出来的，不在审查原文里）；
4. 运行时 `shield_torch` 把整个 `torch.ops` 换成抛错桩，沿用既有复原列表进出各一次 setattr。
   标定过 15 种合法 wrapper 用法（empty / zeros / empty_like / arange / tensor / cat / stack /
   contiguous / view / reshape / to / copy_ / clone / transpose / cuda.is_available）零误杀。

错误文案仍走 `_runtime_cheat_message`：远端只回一句文本，`lib_fallback_from_error` 只认那一个
格式，生产者与解析器必须认出同一个符号（测试里断言了这一点）。

**分工边界（如实说明）**：生产里 `_load_module` 在进盾之前就跑完了，所以
`from torch.ops import aten` 这种**导入期捕获**盾追不回来，那一半由静态侧拦。运行时侧覆盖的是
**调用期**属性访问。这不是本次引入的限制，既有的 `from torch import matmul` 同理。

测试落点：新写法进 `CAUGHT_BOTH_DSL`，由**既有**的两个 DSL 读者覆盖；运行时另加一个类，
断言拦得住、会复原、不误杀三件事。attention 那条原来测死写法的用例换成真能执行的两种绕道。

复核还发现 `op = torch.ops.aten.mm.default` 可以在模块加载时先保存可调用对象。静态检查现在在
读取 `torch.ops` 时就报违规，覆盖直接访问、常量 `getattr` 和 `from torch import ops`，因此
候选在进入加载阶段前就会被拒绝。

## 4. 这三条不改变 C5 的决策

身份门本身没动，`PINNED_RUNTIME` 没动，A/B 两条修法仍待你选（[C5 卡点 §4](c5-identity-blocker-20261107.md)）。
差别是：**选 A 之后不再连撞三处**。真机复测仍需要一次 H20 机时。
