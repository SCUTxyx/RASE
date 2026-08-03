# RASE Phase 1A：48-task Replacement Audit 结果（2026-08-02）

## 状态

**COMPLETE（development-only pilot）**

本轮回答旗舰计划中最危险的审稿问题：**为什么不从 reset 开始一直使用 OFT？**

该 48-task cohort 复用 Phase 0G 的开发任务与冻结 metadata-only design，只用于路线决策；
它明确排除于未来 flagship hidden test，不能作为最终论文测试集。

## 冻结协议

- design：`runs/rase_ui_phase0g_independent48_design.json`
- 48 个 task / 48 个 episode，task 与 episode 均唯一；
- 4 suites × 3 cells（clean L0 / camera L1 / robot L1）× 4 tasks；
- SOURCE-ONLY：SmolVLA full horizon，不再使用旧 80-step cap；
- OFT-ONLY：LIBERO `reset()` 完成后、第一次 source action 之前恢复 OFT；
- source→OFT：复用 Phase 0G 在 policy env-step 25 的 immediate OFT；
- 统计单位：task/episode；exact join；McNemar exact test；10,000 次 task bootstrap；
- bootstrap seed：`2026080201`。

### Reset 语义勘误

LIBERO 的 `reset()` 内部执行 10 个机器人初始化 simulator steps。因此真正的
“from reset / zero source action”快照满足：

- snapshot policy step = 0；
- source actions before snapshot = 0；
- post-reset simulator timestep = 10。

48/48 首快照均满足上述条件。不能把 simulator counter=10 误解为执行过 10 个
SmolVLA 动作，也不能再把该字段写成必须为 0。

冻结 state-key 集：

- n = 48；
- state-key SHA-256：
  `df345cbd35e22ff1b736893ccd9d67cfb0e6b28b5290fc5e3072517819365065`；
- selection uses outcomes = false。

## 主结果

| 模式 | 成功 | 成功率 |
|---|---:|---:|
| SOURCE-ONLY full horizon | 10/48 | 20.83% |
| OFT-ONLY from reset | 42/48 | 87.50% |
| source→OFT at policy env-step 25 | 37/48 | 77.08% |

### 配对统计

- OFT-ONLY − SOURCE-ONLY：
  **+32/48 = +66.67pp**；
  task bootstrap 95% CI **[+50.00pp, +81.25pp]**；
  McNemar exact **p=1.94e-8**。
- source→OFT − OFT-ONLY：
  **−5/48 = −10.42pp**；
  95% CI **[−22.92pp, +2.08pp]**；
  McNemar exact **p=0.2266**。

SOURCE vs OFT-only 四象限：

| 象限 | 数量 |
|---|---:|
| OFT rescue（source fail, OFT success） | 34 |
| source unique / harm（source success, OFT fail） | 2 |
| redundant（both success） | 8 |
| unsupported（both fail） | 4 |

### 分 suite

| Suite | SOURCE-ONLY | OFT-ONLY | source→OFT |
|---|---:|---:|---:|
| Spatial | 2/12 | 12/12 | 10/12 |
| Object | 3/12 | 9/12 | 8/12 |
| Goal | 3/12 | 9/12 | 9/12 |
| Long | 2/12 | 12/12 | 10/12 |

### 分条件

| Cell | SOURCE-ONLY | OFT-ONLY | source→OFT |
|---|---:|---:|---:|
| clean L0 | 9/16 | 15/16 | 16/16 |
| camera L1 | 0/16 | 15/16 | 11/16 |
| robot L1 | 1/16 | 12/16 | 10/16 |

OFT-only 不仅在 perturbed 条件占优，在 clean 条件也由 9/16 提升至 15/16。
因此最新版计划的 replacement kill condition 第一项已经触发；真实成本与部署优势
两项仍未完成，当前还不能正式宣告 kill，但该 policy pair 已处于**高替代风险**。

## 稀疏互补例外

冻结 pilot gate 按预注册规则返回 `recovery_framing_signal`，原因是 SOURCE-ONLY 有
2 个 OFT-only unique wins 且分布在两个 suite：

1. `libero_goal_000616`，robot L1：
   source=true，OFT-only=false，source→OFT=false；
2. `libero_object_000008`，clean L0：
   source=true，OFT-only=false，source→OFT=true。

这个 gate 只能解释为“存在待复现的能力例外”，不能解释为整体 recovery 叙事已经成立：

- 例外仅 2/48；
- 其中只有 1 个同时支持“source prefix 使 OFT 可成功”的互补解释；
- source→OFT 相对 OFT-only 有 3 个 unique wins，但 OFT-only 相对 handoff 有 8 个 unique wins；
- handoff 总体没有超过 OFT-only。

因此下一步不是训练 selector/world model，而是独立复现这些 exception，并补真实
cost/deployment audit。

## Timing 与成本边界

观测值：

- OFT-only：约 16.16 policy ms/env-step；
- SOURCE-only：约 28.17 policy ms/env-step；
- source→OFT 历史 artifact：约 16.14 ms/env-step，但只覆盖 env-step-25 之后的 OFT
  continuation，不含 source prefix。

限制：

- SOURCE-only 采集末段曾检测到一个短暂外部 GPU process；terminal success 不受影响，
  但本轮 SOURCE wall-clock/policy timing 不作为正式独占计时证据；
- 三模式 episode horizon 与终止时刻不同；
- source→OFT 缺少前 25 步 source 计算；
- 尚无功耗、峰值显存、网络、端侧、本地部署和隐私数据。

故当前只能做 success replacement 结论，不能做最终 resource complementarity claim。

## 实施与代码

新增或修改：

- `configs/collect_rase_ui_phase1a_replacement48_source_v2.json`
- `rase/collect/pipeline.py`
- `rase/collect/lerobot_libero_plus_adapter.py`
- `scripts/export_initial_replacement_keys.py`
- `scripts/analyze_replacement_audit.py`
- `scripts/run_rase_ui_phase1a_replacement48.sh`
- `tests/test_replacement_audit.py`

新增能力：

- full-horizon source episode metrics；
- exact task/episode/design join；
- post-reset、zero-source-action snapshot audit；
- suite-serial OFT-only；
- success quadrants、McNemar、task bootstrap、suite/cell 分层；
- replacement pilot gate；
- append-only fresh/resume runner。

验证：

- Ruff：PASS；
- bash syntax：PASS；
- targeted regression：**21 passed**；
- 实验：4/4 suite summary，48/48 exact join，`PHASE1A_DONE`。

## Artifact

- source summary：
  `runs/rase_ui_phase1a_replacement48_source_summary_v2.json`
- reset keys：
  `runs/rase_ui_phase1a_replacement48_initial_keys_v2.json`
- OFT-only：
  `runs/rase_ui_phase1a_replacement48_oft_only_{spatial,object,goal,10}_v2/summary.json`
- analysis：
  `runs/rase_ui_phase1a_replacement48_analysis_v2.json`
- report：
  `runs/rase_ui_phase1a_replacement48_analysis_v2.md`
- log：
  `runs/rase_ui_phase1a_replacement48_v2.log`

Artifact file hashes：

- keys JSON：
  `abf6a156856c4121ffb10b9b90864375df0e837b905869e9040008208072ffb5`
- source summary：
  `ad0d8dc6c49dd96f7d278c196a5c99005ff7f4dca601ac45b3bda5b8d726b858`
- analysis JSON：
  `7b2a984ef7054553704b66721ce78bb63b4af88b05c67ac2f451963c614f7909`
- analysis Markdown：
  `42df10bc453c4dedb6731fc8aa2b8906d5413d355f44ba67f4e1be628efc2928`

## 决策与下一阶段

### 立即做：Phase 1B complementarity confirmation

1. 冻结上述 2 个 source-unique、3 个 handoff-unique、8 个 OFT-over-handoff
   任务身份，禁止 outcome-driven 换任务；
2. 对每个任务扩展独立 init states/seeds，至少每任务 10–20 个 episode；
3. 同时跑 SOURCE-ONLY、OFT-ONLY、source→OFT，全部 full horizon；
4. 加入独占 GPU provenance、峰值显存、GPU-seconds、wall time、RPC、网络字节；
5. 预注册保留 recovery framing 的最低 replication 与实际资源优势阈值。

### 并行基础设施

- 冻结 flagship 100-task 候选池与 hidden split，但不打开 hidden test；
- 选择另外 2 个 source/OFT policy pairs；
- 跑 LIBERO + 第二平台 full-arm smoke test；
- 建 failure taxonomy codebook 和 rollout budget dashboard。

### Gate 后再做

只有当稀疏互补在独立 init/seed 上复现，或 source 有真实端侧/成本/隐私优势时，
才进入 500–1,000 state failure taxonomy pilot 和 fixed visual intervention。
否则更换 policy pair，或将主叙事改为 policy replacement audit。

继续暂停 timing selector、通用 operator selector和无明确 opportunity gate 的 world model。

