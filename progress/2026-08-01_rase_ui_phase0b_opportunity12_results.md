# RASE Unified Intervention Phase 0B：48-state 三算子结果与下一步

日期：2026-08-01（Asia/Shanghai）  
状态：**实验完成；success-only opportunity gate = NOT_READY；禁止立即训练 selector/world model**

## 一句话结论

这轮实验证明了 `SWITCH_OFT` 能恢复大量 SmolVLA 无法恢复的状态，但没有证明“按状态选择算子”能提高成功率：OFT 的成功集合严格覆盖了 CONTINUE 与 REPLAN，固定选择 OFT 已达到 same-state oracle 的 62.5%。真正值得继续的方向是**预先冻结资源成本后的 utility-aware intervention benchmark**；零成本 success-only selector 分支当前应暂停。

## 实验设计

- 12 个 source episodes / 12 个不同 task / 48 个 strict decision-context states。
- 四套件均衡：Spatial、Object、Goal、Long 各 3 task、12 states。
- 扰动均衡：clean、camera、robot 各 16 states；camera/robot 覆盖 L1/L2。
- 每个 episode 固定 snapshot step `0,2,4,6`，每个 state 完整评测同状态三臂：
  `CONTINUE_SMOL_ACTIVE_CHUNK`、`REPLAN_SMOL`、`SWITCH_OFT`。
- 每臂 1 个 continuation seed；本轮属于 opportunity screen，不是论文 test split。
- 所有 144 outcomes 都是 observed、non-proxy，48/48 states 完整配对。

环境与冻结标识：

- Git HEAD：`454f76384e5195a750584dd9753c29b0701bb6af`
- `env.lock.md` SHA-256：`0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2`
- SmolVLA policy SHA-256：`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`
- frozen keys SHA-256：`c60ee5956e6e3b246cc72254dd830bccf322b8dca38385ff2db8cded8ed7e1ac`
- Python：SmolVLA env 3.12.13；OFT env 3.10.20。
- OFT checkpoint bundle SHA-256：Spatial `664ee2eefd68b46945ea24b281059df049a400e2b500ef2c044e9eebe1b6c5b0`；Object `94659ba31d32147d07a87dedd78f627a4f8e0925b0eea10e39aef5487e56d5a6`；Goal `c8b86f6d797328447e2cba59efbbf6638b63f19c5426df92b3c9e000890e68ff`；Long `a572ce03ab9800eaf530ae1f139536811fa9314609ee60b0cfb1e298187ccdad`。

## 主结果

| 算子 | 成功数 | 成功率 |
|---|---:|---:|
| CONTINUE | 9/48 | 18.75% |
| REPLAN | 13/48 | 27.08% |
| SWITCH_OFT | 30/48 | 62.50% |
| same-state oracle | 30/48 | 62.50% |

- success pattern（顺序 C/R/O）：`000=18`、`001=17`、`011=4`、`111=9`。
- 没有出现 `100/010/101/110`：OFT 覆盖了 Smol 的全部成功。
- success-only unique winner：只有 OFT，17 states / 9 tasks。
- oracle − best fixed = `0.0`；12-episode cluster bootstrap 95% CI = `[0.0, 0.0]`，gap 为正的 bootstrap 比例为 0。
- 配对诊断：REPLAN 比 CONTINUE 多救 4 states、少救 0（exact McNemar `p=0.125`）；OFT 多救 21、少救 0（`p=9.54e-7`）。后者是 state-level 诊断值，不能替代 episode-cluster inference。
- 正式 success-only gate 失败原因：gap `<0.05`；有 task 支撑的 unique-winning operator 仅 1 个，低于 3 个。

## 分层结果揭示了什么

### 按扰动

| 扰动 | CONTINUE | REPLAN | OFT | 无算子成功 |
|---|---:|---:|---:|---:|
| clean | 56.25% | 81.25% | 93.75% | 1/16 |
| camera | 0% | 0% | 37.50% | 10/16 |
| robot | 0% | 0% | 56.25% | 7/16 |

camera/robot L1–L2 对 Smol 过难，采样分布被推入“必须升级到 OFT”的区域。下一批不能继续简单扩大 L1/L2 failure states。

### 按时间

| snapshot step | CONTINUE | REPLAN | OFT | 无算子成功 |
|---:|---:|---:|---:|---:|
| 0 | 33.33% | 33.33% | 75.00% | 3/12 |
| 2 | 25.00% | 33.33% | 66.67% | 4/12 |
| 4 | 16.67% | 33.33% | 58.33% | 5/12 |
| 6 | 0% | 8.33% | 50.00% | 6/12 |

可恢复性随干预推迟单调下降，因此最终方法必须同时建模“是否干预、何时干预、采用哪种干预”，不能退化成 task-level operator classifier。

### 按套件

- Spatial：C/R/O = 8.3% / 25.0% / 75.0%。
- Object：25.0% / 33.3% / 75.0%。
- Goal：25.0% / 25.0% / 50.0%。
- Long：16.7% / 25.0% / 50.0%。

## 成本敏感性：为什么仍可能有 selector 机会

本轮正式 outcome 的 `utility_cost=0`，所以 success-only 结论不能被事后改写。新增工具只做下一轮预注册前的敏感性分析：成功奖励固定为 1、CONTINUE cost=0、REPLAN cost=0.01，扫描 OFT 固定调用代价。

| OFT cost | best fixed | oracle utility gap |
|---:|---|---:|
| 0.00 | OFT | 0.0000 |
| 0.02 | OFT | 0.0121 |
| 0.05 | OFT | 0.0315 |
| 0.10 | OFT | 0.0638 |
| 0.20 | OFT | 0.1283 |
| 0.30 | OFT | 0.1929 |
| 0.40 | REPLAN | 0.2217 |
| 0.50 | REPLAN | 0.1863 |

当 OFT cost > REPLAN cost 时，三类 utility winner 自然出现：CONTINUE 27 states/11 tasks（成功相同时免升级，或全失败时避免徒劳开销）、REPLAN 4/3、OFT 17/9。OFT cost=0.10 时 gap 超过 0.05，但 `0.10` 是探索值，不是已校准事实。必须先根据推理能耗、峰值显存、部署价格或系统预算定义其物理含义，再冻结后确认。

## 已完成的代码完善

1. 修复 opportunity gate 的 tie bug：全失败状态不再被错误计为每个算子的 winning region；gate 使用 unique winners，并报告 state/task 支撑数。
2. 新增 12-task / 48-state Phase 0B 配置和一键流水线。
3. 新增矩阵分层分析器：suite、扰动、step、success pattern、paired McNemar、episode-cluster bootstrap。
4. 新增固定干预成本敏感性扫描器，明确标记为 diagnostic、要求 confirmation 前冻结成本。
5. 修复 OFT runner 对 `adapter` import string + `adapter_config` 的兼容性。
6. 修复 NumPy scalar 导致分析 JSON 无法写出的错误。
7. 流水线改为 stage-level resume：已有 pool/keys/Smol summary 会跳过，避免恢复运行重写 collection summary；OFT 已按 suite 跳过完整结果。
8. 为以上修复和工具补充回归测试。

## 接下来怎么做（按优先级）

### P0：冻结问题定义与成本协议（先做，暂不训练）

1. 定义论文主对象为 resource-aware unified intervention：动作至少包含 CONTINUE、REPLAN、SWITCH_POLICY；决策包含 intervention timing。
2. 单独测量 inference-only latency、GPU energy、peak memory 和实际部署价格，不能使用“整段 rollout 用时”冒充模型调用成本。本轮整段 rollout 中 OFT 因更早成功反而平均更短，这恰好说明两者必须分离。
3. 预注册三种资源预算（low/medium/high）或一个物理成本公式；冻结 success reward、cost normalization、harm/futility penalty 后再看确认集。
4. success-only 指标继续作为能力诊断；主 gate 改为 preregistered utility gap，但同时报告 raw success，防止用成本掩盖能力退化。

### P1：Phase 0C boundary screen（建议下一次实际运行）

1. 不再只收 L1/L2 failure states；先做小规模 difficulty calibration，加入 clean、弱扰动（目标 source success 20%–80%）与 L1，减少 Smol 全零 strata。
2. 目标至少 24 source episodes / 24 unique tasks / 96 states，四 suite 均衡；每 episode 仍取 `0,2,4,6`，screen 每臂 1 seed。
3. cohort 预设覆盖：source-success/shared-success、replan rescue、OFT rescue、all-fail；采样依据只能来自独立 calibration，不能按确认集 outcome 挑状态。
4. Screen gate：96/96 完整；每个 operator 至少 4 个 unique-winning tasks；预注册 utility oracle gap ≥0.05；episode-cluster bootstrap 下界 >0；harm/futility 非零且可审计。
5. 若仍由固定 OFT 支配，停止 learned selector，论文保留 benchmark/diagnosis + resource frontier；不要继续堆模型。

### P2：独立 Phase 0D confirmation

1. 使用 task/episode-disjoint frozen cohort；至少 3 continuation seeds/arm，报告 seed sensitivity。
2. 所有阈值、成本、状态采样和排除规则在运行前写入 protocol 与 hash manifest。
3. 统计单位以 source episode/task cluster 为主；报告 cluster bootstrap CI、paired permutation，控制 suite/扰动/step 多重比较。
4. 必须加入固定算子、always-OFT、always-CONTINUE、random cost-matched、置信度路由和 oracle 上界。

### P3：只有 confirmation 通过后才训练 selector/world model

1. 先训练简单且可校准的 utility/regret predictor；按 task+episode group split，禁止同 episode snapshots 泄漏。
2. 目标不是 action accuracy，而是 held-out utility、oracle regret、success–cost Pareto、risk–coverage 与 calibration error。
3. world model 先做 operator-conditional potential-outcome prediction和不确定性校准；与无 dynamics、无 history、无 uncertainty 的消融比较。
4. 只有简单模型稳定超过 cost-matched/random/always-OFT 后，才开放 sequence model、RL 或更复杂 world model。

### P4：顶会完整证据链

- 多 suite × 多扰动强度 × 多时间点 × 多 seed；报告 task-level 与 episode-level CI。
- 明确 operator semantics、snapshot/restore determinism、paired noise、proxy=0、checkpoint/hash。
- 核心消融：无 cost、无 timing、无 uncertainty、无 world model、两算子 vs 三算子。
- 外部有效性：新 task 或真实机器人只作为最终 confirmation，不用于反复调 gate。
- 当前可主张：OFT 对此 cohort 显著扩展 recoverability，干预越晚可恢复性越低。
- 当前不可主张：learned selector 有效、world model 有效、三算子 success oracle 优于固定 OFT、真实机器人泛化。

## 复现实验与分析命令

完整新运行：

```bash
cd /root/autodl-tmp/RASE
tmux attach -t 0
FRESH_RUN=1 TAG=screen_v2 ./scripts/run_rase_ui_phase0b_opportunity12.sh
```

断点恢复：

```bash
cd /root/autodl-tmp/RASE
FRESH_RUN=0 TAG=screen_v1 ./scripts/run_rase_ui_phase0b_opportunity12.sh
```

本轮关键产物：

- `runs/rase_ui_phase0b_opportunity12_matrix_screen_v1/summary.json`
- `runs/rase_ui_phase0b_opportunity12_matrix_screen_v1/opportunity_audit_success_only.json`
- `runs/rase_ui_phase0b_opportunity12_matrix_screen_v1/analysis.json`
- `runs/rase_ui_phase0b_opportunity12_matrix_screen_v1/cost_sensitivity.json`
- `runs/rase_ui_phase0b_opportunity12_screen_v1.log`
- `runs/rase_ui_phase0b_opportunity12_screen_v1_smol_complete.log`

## 已知偏差

- 首次 OFT 启动因 runner 错把 `adapter` import string 当 mapping 而退出；无正式 OFT arm 被消耗。修复、测试后从 OFT 阶段恢复。
- 恢复版旧流水线重跑 collection summary，导致该 summary 显示“12 episodes skipped、0 new states”；真实 pool manifest、frozen keys 和 48-state matrix 未变。新版 stage-level resume 已修复此问题。
- 本轮单 continuation seed，不能用于最终方差估计。
- 成本扫描是 post-hoc sensitivity，不是确认性结果。
