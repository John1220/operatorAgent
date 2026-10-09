# 本地语料持久化提交与幂等恢复

日期：2026-10-09。接续 [版本化 CUDA 量具与真实入库](versioned-cuda-ruler-and-admission.md)。
上一轮量具与 H20 验收已本地提交为 `9fbc099`；本文描述其后的入库协议改动。

## 1. 解决的问题与范围

原先 `_save_to_db` 把资格门的 `keep=True` 当成落盘成功，`KernelDatabase.promote`
先改内存再追加文件，写盘异常又被吞掉。因此磁盘满、权限错误、中断时，报告仍可能
说“已入库”；重试还缺少绑定一次实际测量的幂等身份。

本轮区分两个事实：

- **eligible / promotable**：资格门与独立确认通过，可以申请入库。
- **persisted**：实际语料事件完成持久化确认，可以声明已经入库。

资格判据仍只有 `oracle.admit_to_corpus` 一处；精度、参考、计时地板、性能阈值、
独立确认及覆盖上限均未放宽。落盘失败不把 valid 读数改成坏读数，也不当成候选 bug。

支持范围是 **POSIX 本地、可信控制端、一个语料所有者/写者**。不是分布式 exactly-once，
不是预算、bandit、失败语料或 GPU 执行的事务系统，也不自动恢复一次算子开发任务。

## 2. 单一提交权威

`corpus_journal.py` 维护现有 JSONL。新增 kernel event 的 admission seal 包含：

| 字段 | 作用 |
|---|---|
| schema / id / key | 协议版本与这一次入库的稳定身份 |
| record_hash | 除首次入库时间外，完整记录的规范 JSON 摘要 |
| event_hash | 含实际首次时间的完整事件摘要，不含 seal 本身 |

**只有完整、换行终止、seal 校验通过的实际 kernel event 才是提交记录。**
`admission.json`、result/benchmark 中的状态及进程内对象都是意图或投影，不是第二权威。

正常提交顺序：

1. 快照待写记录，校验稳定身份与源码/首测 binding 一致。
2. 对实际文件取得非阻塞排他锁，扫描现有事件；发现完整坏行则拒绝新写入。
3. 若同 admission 已存在，核对完整 record_hash，重新完成 durability fence，
   返回 `already_committed`，**不再追加**。
4. 否则保留既有同 kernel ID 的分数替换规则，追加一次带 seal 的事件。
5. flush、文件 fsync、父目录 fsync；新建目录时也同步其父目录项。
6. 从实际文件精确读回同一事件，才能返回 `committed` 并安装内存记录。

意外第二个写者/重放者持锁时直接失败关闭，不把锁当并发任务调度器。
文件或直属目录为 symlink 时拒绝；这不等于对所有祖先目录、恶意同 UID 的完整防护。

### 2.1 状态语义

| 回执 | 可以声明 persisted | 含义 |
|---|---|---|
| committed | 是 | 新事件已 fsync，且精确读回 |
| already_committed | 是 | 同一事件存在，重新完成 fence；没有第二次追加 |
| persistence_disabled | 否 | harness 显式关闭语料，不能靠内存声称落盘 |
| refused | 否 | 日志损坏、同身份不同 payload、旧 durable 分数未被改善 |
| failed | 否 | 在尝试追加/fence 前失败，如路径权限或身份不合法 |
| outcome_unknown | 否 | 已开始写入或确认，但没有取得完整成功回执 |
| recovery_hold | 否 | 显式恢复证据不匹配，或恢复/发布未能完整完成 |

`outcome_unknown` **不等于事件一定没写进去**：完整行可能已经存在，只是 fsync/readback
失败或回执丢了。必须使用同一 admission 身份去核对并重新 fence，而不是重新测量、
换一个 ID 再 append。失败时不安装新记录，不覆盖已有 durable best。

资格通过但未获提交回执时，`corpus_admission.eligible=true`、`kept/persisted=false`，
声明等级为 `promotable`，原因中明确写“符合资格，但未确认持久化”。
等级仍由 `eligibility.claim_level` 单点推导；调用方不能凭资格传 `persisted=True`。

## 3. 幂等身份不等于候选代码去重

生产 confirmed-observation key 绑定：

```text
run_id
manifest_hash / contract_hash / workload_hash / protocol_hash / candidate_hash
first_verification_id / second_verification_id
```

两次 binding 的公共身份必须完全相同，verification ID 必须不同；实际确认文件里的
首测与确认性能窗口也必须不同。profile 身份已被 frozen manifest 的 controls 间接覆盖。

相同一次主测＋确认重复提交、进程中断后恢复：同一个 admission ID。
新运行或新测量/确认身份：不同 admission ID，即使候选代码完全相同也不是全局重复。
这不取消同 kernel ID 的既有分数替换规则；正常新产物有自己的 kernel ID。

创建时间由第一次实际追加确定，不参与重试的 record_hash；完整 event_hash 仍校验
该实际时间。其余 payload 改变不能冒充同一次提交。

旧无 seal 的 kernel/retrieved 日志仍能按既有记录 schema 读取，不替它们补造测量证明。
直接 `promote(record)` 兼容 API 使用 payload 派生的 legacy-record 身份，返回值现在
代表真实持久化成功；**它不是 CUDA 资格认证接口**。显式无效 key 不得回退 legacy。
生产一旦携带 run binding，就必须有完整准备证据，不能借此兼容路径绕过。

## 4. 准备意图与显式恢复

`corpus_admission.py` 在语料副作用前，通过同目录临时文件、fsync、atomic replace、
目录 fsync 写 `admission.json`：

- 完整待写记录、record_hash、稳定 key、admission ID；
- 首测 metrics、实际 ConfirmRecord、冻结阈值及原资格理由；
- manifest/contract/verification/confirmation/源码的实际文件摘要；
- 指定语料的绝对路径；
- 整个 request 的规范摘要；初始状态 `prepared`。

恢复不信任之前的 `status` 或成功回执投影。必须重新读取实际文件，核对 manifest 自身、
task/workload/contract 摘要、完整**当前 evaluator 源码身份**、首测不可变 metrics、
候选文件、确认 binding/读数及独立窗口，再调用同一个 `admit_to_corpus`。
只修改 intent 后重算 SHA，也不能替代未改动的实际 benchmark/confirmation 证据。

源码、契约、阈值、候选、确认、窗口、metrics、目标 corpus 等任一漂移，或证据缺失，
均 HOLD，不进入新语料提交。哈希是完整性检查，不是签名；不抵御可信 owner 同时伪造
全部文件。对需要换版本/换输入的任务，应执行新的验证，不把旧意图“迁移”为新资格。

### 4.1 操作入口

先确认没有同 corpus 的活跃写者，使用原 artifact 与原 corpus 的绝对路径：

```sh
python -m operator_agent.corpus_admission \
  --artifact /absolute/path/to/original/artifact \
  --corpus /absolute/path/to/original/kernel_db.jsonl
```

stdout 为结构化 CommitResult；只有 committed/already_committed 返回码为 0。
同一命令重复执行不新增 kernel event。不同 corpus 路径 HOLD。

恢复只完成该次语料副作用与登记产物字段，**不调用后端/LLM，不消耗确认预算，
不恢复 best/recipe/bandit，不让旧 L2/L3 成为新任务的 live credential**。
正常 Conductor resume 仍走原有身份校验、重新验证与新测量；不自动调用此恢复入口。

## 5. 半条日志、坏行与重放

未换行结尾不是提交，哪怕字节已经是合法完整 JSON，也不装入内存。
后续追加前保留其全部字节，追加 `NUL + newline` 将其隔离，再写新事件。
不能只补 newline，否则会把一次未确认的旧 JSON 意外变成有效事件。
重复中断隔离操作仍能再次隔离，不截断原始文件。

完整坏行跳过并记数，保留此前可读历史，但阻止新提交，避免在损坏 authority 上续写。
不完整旧更新必须先通过 KernelRecord schema，不能覆盖此前同 ID 的有效记录。
seal 相同的物理重复事件只重放一次；同 admission 的不同事件被标记为冲突。

重放在安装读回记录前也重新完成 fence，解决完整事件存在但原回执丢失的情况。
读取/fence 异常写入 `persistence_error`，summary 中可见，不能静默称作成功载入。
retrieved 事件按原顺序作用到记录的副本，重复 replay 不重复加计数。
检索日志使用同一追加/fence 和尾部隔离机制，但**不提供检索副作用 exactly-once**。
FailureDatabase 本轮未改造。

## 6. 产物发布不是跨文件事务

实际 journal 提交后，将回执写回 `admission.json`，再原子更新 result.json、benchmark.json
已登记的 corpus_admission/claim_level 及 evaluator 派生字段；不重建或替换原测量数据。
ArtifactStore 保存/读回这两项登记 metadata。

这些文件分别原子写，**不是全体文件同时提交**。中途死亡可能留下旧 benchmark、
新 result 或旧 prepared 状态；以 journal 为准，再执行显式恢复修复投影。
发布失败不会回滚已经 durable 的事件，也不会因重试制造第二条。

正常 run 的 report.md 在提交尝试之后生成，能真实显示提交失败及原因。
显式恢复不重建整份历史报告/运行上下文；恢复后的当前状态应读取 JSON 回执及语料，
原 report.md 仍是其生成时的历史快照，不是恢复状态的实时权威。

原始源码/verification/benchmark 的初次保存仍沿用 ArtifactStore 的既有方式，
本轮不承诺整棵产物目录的断电持久化事务；恢复证据缺失时必须 HOLD，不能从 journal
反造测量证明。新的持久化成功声明针对实际 corpus 提交记录，不针对全体产物文件。

## 7. 验证与诚实边界

验证入口：

```sh
python -m pytest tests/test_corpus_commit.py tests/test_spine_wiring.py \
  tests/test_kernel_db_persistence.py tests/test_execute_inner_behavior.py -q
python -m pytest tests -q
python tests/golden_regression.py compare
```

专项 **123 passed**。新的 corpus 测试实际走生产 Conductor 资格/提交接线、实际文件、
真实 fresh-instance replay 和 CLI；L3 数据明确标为离线 fixture，不是 CUDA 性能证据。
另验证 `_produce_remeasure` 实际生成确认 sidecar 后可以准备、提交、恢复；恢复没有
第二次调用 scripted L3，也没有第二笔 confirm 预算。

最终全量 **3376 passed，1 skipped，2 warnings**（两条既有 nonfinite-reference warning）；
Golden **4/4 PASS**，未刷新 registry。完整结果见该验证目录的 `summary.json`。

7 个实际子进程 `os._exit(73)` 中断点：

1. 追加前；2. fsync 的半条行；3. 完整追加后、fence 前；4. fence 后；
5. 内存安装后；6. 原子提交回执后；7. result 发布后、benchmark 发布前。

所有点恢复后只有一条有效 kernel event；再次 fresh-instance 恢复字节不变。
另覆盖权限/短写/两种 fsync/readback 失败、禁用持久化、锁冲突、身份/payload 漂移、
seal 篡改、重算 intent 摘要、日志尾部隔离、重复事件与旧完整坏更新。

隔离副本执行 **15/15 守卫 mutation 杀死**：分别移除 fence、文件/目录 fsync、
精确 readback、重复提交核对、payload 冲突、完整 event_hash、尾部隔离、未换行过滤、
显式 key 校验、资格/持久化分离、mandatory prepare、request hash、当前源码检查、
实际首测 metrics 检查。先跑 clean 副本，失败均为目标测试断言/边界守卫，不是导入错误。
日志与结果保留在 `output/corpus_commit_validation`，该目录不进 Git。

这验证的是进程死亡及局部 I/O 故障模型，不是物理断电实验。durable 声明依赖本地
文件系统/设备诚实实现 fsync，不承诺 NFS、失真的设备缓存或恶意 owner 下的事务保证。

**本轮未执行新 H20 性能验收。** 入库源码及 ArtifactStore 已加入 frozen protocol 身份，
旧 run 的完整源码身份已与当前版本不匹配。需区分：量具 profile 只绑定计时通道的
部署源码、设备和 runtime；仅改变控制端入库源码，不一定使同一仪器的旧 profile 校验
失败，但一定改变 frozen evaluator 身份。上一轮 H20 结果继续属于 `9fbc099`，不能
据此声明当前版本已验收。下一次验收主动重新独立预检、生成新的 profile，并完成
当前 frozen evaluator 的新主测与确认，不复用旧读数。

后续在 `c92ed01` 上已完成新的独立预检、H20 主测/确认与真实 seal 入库，以及六次
新进程重复恢复；见 [持久化入库协议的 H20 验收](durable-corpus-h20-acceptance.md)。
