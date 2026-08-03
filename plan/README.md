# RASE research-plan index

更新：2026-07-31

## 当前唯一执行主线

[`RASE_top_conference_execution_v4.md`](RASE_top_conference_execution_v4.md)
是当前 canonical research plan。发生冲突时，v4 优先于本目录其他文档。

### 2026-07-31 outcome addendum（当前状态）

W9C 已修复 clean task identity（Plus 0–9 并非 official clean-10），随后：probe **PASS**
（mean SR 0.6125）、clean32 coverage **PASS**、episode/task readiness **PASS**。但 ridge
在 episode-held-out 与 task-held-out 上相对 action-matched random 的 Δutility 都为 **0**，
置信区间跨 0，触发预注册 `kill_method_branch`。因此：

- 不升级 ridge，不训练 MLP/RL；原 preregistration 与 kill criteria 继续有效；
- 论文主线冻结为 **benchmark / diagnosis**，selector 仅作负结果、appendix 或未来工作；
- task-held-out test 仅 **8 个 clean-control states、0 failure-challenge states、0 learned
  escalation actions**，只能说明 gate 未通过，不能声称跨任务 failure routing 泛化；
- W10 Object/Spatial：collect **80/80 failure**，direct Smol **0/16**，direct OFT **1/16**
  （Spatial 0/8），split **`NOT_READY`**；正例仍以 Goal/Long 为主，不得把 Object/Spatial
  写成普遍可恢复；
- 下一阶段只允许做预注册 diagnosis / 新覆盖扩展登记，不得重开 selector，不得改 W10 seed
  救 split。

详见 [`2026-07-31 W9C selector gate`](../progress/2026-07-31_w9c_selector_gate_result.md)、
[`2026-07-31 W10 result`](../progress/2026-07-31_w10_object_spatial_benchmark.md)
与 [`idea evolution / next questions`](../progress/2026-07-31_idea_evolution_and_next_questions.md)。

当前论文问题已从“SmolVLA 上的五 fallback RL selector”调整为：

> 视觉扰动下的 policy-relative recoverability benchmark，以及在候选执行、
> cross-policy escalation 和 abstention 之间进行成本敏感决策的轻量 selector。

### 2026-07-29 最新冻结决策

W6 配对矩阵已完成（Smol 0/8 states，OFT 2/8，`p=0.5`）；随后 direct/zero/candidate
prefix attribution 得到 candidate-specific rescue `0/2`。因此首版 selector 冻结为
`CONTINUE_SMOL / ESCALATE_OFT / ABSTAIN` 三动作，candidate 0..K-1 只作诊断/上界，
`any_of_K` portfolio 不得作为训练标签。W7 的 episode-disjoint 24-state replication
已得到 Smol `0/192、0/24 states`，OFT 最终矩阵等待远端 artifact 核验。W8 下一实验是
仅 24 次 direct-OFT-from-snapshot rollout，建立真实 deployable escalation 标签。

W5 proposal-temperature sweep 已完整覆盖 `0.3/0.7/1.0`：同一 24-state
failure-frontier cohort 上三档合计 `0/576` one-shot candidate outcomes。候选温度
元数据、文件独立性与多样性均已核查；mean pairwise endpoint L2 随温度从
`0.726 → 2.133 → 3.066` 增加。因此：

- proposal-temperature 诊断线永久关闭；
- W5 formal confirm 因 screen-hit union 为空而不运行；
- 当前活动实验切换到 `docs/runbooks/w6_l1_l2_policy_matrix.md`；
- W6 已冻结 8-state、episode-distinct 的 L1–L2 failure-challenge cohort；
  Smol→Smol one-shot screen 已完成为 `0/64 candidates、0/8 states`。首次 OFT
  启动因 GPU OOM、未产生 outcome；当前唯一阻塞项是完成同候选 OFT arm。
  clean-success control 作为独立 cohort 后续构建，不混合分母。

转向由以下冻结证据触发：

- W3/W4 SmolVLA continuation 在 ADEQUATE 状态上持续零恢复；
- W4 同状态、同候选下，OFT portfolio 恢复 17/32 状态、命中 65/256 候选；
- W5 t=0.7 在 OFT-recovered 与 failure-frontier cohort 上仍为零命中；
- W4 全部 `t0=0` 且现有 pool 为 3666/3666 failure，不能承担时机泛化或
  clean-regret 结论。

## 文档状态

| 文档 | 状态 | 可继续使用的部分 | 不再直接执行的部分 |
|---|---|---|---|
| `RASE_top_conference_execution_v4.md` | **Current** | 全文 | — |
| `RASE-Lite_design_report_v3.1.md` | Historical design | 文献图谱、FEB 定义、统计纪律 | 原 C3 novelty、五 fallback+DQN 时间线 |
| `guide1_ngc_plus_data_collection.md` | Reference, needs v4 gates | fork、候选、Wilson、QC | 固定 4,000 状态目标、state-level random split |
| `guide2_vla_reproduction_finetuning.md` | Reference | baseline/OFT 环境与复现纪律 | 当前阶段的 LoRA/RL 微调优先级 |
| `guide3_rl_selector_training_framework.md` | Deferred | 预算记账、日志、action masking 思路 | 立即实现 BC+CQL→DQN、七路重特征 |
| `codebase_strategy.md` | Reference | frozen dependency + thin wrapper 原则 | 立即自研五 fallback 和完整 DQN |

## 当前 gates（2026-07-31 outcome 后）

1. **W9C identity/probe/coverage/readiness：PASS。** 该结果修复数据有效性并允许执行
   ridge gate，但不等于 selector 有效。
2. **Ridge method gate：KILL。** episode/task held-out 均未优于 matched-random；按原
   kill criterion 停止 selector 方法分支，明确不上 MLP/RL。
3. **W10 Object/Spatial coverage：COMPLETED / split NOT_READY。** direct Smol 0/16、
   OFT 1/16；escalate oracle 不足。详见
   [`2026-07-31 W10 result`](../progress/2026-07-31_w10_object_spatial_benchmark.md)。
4. **当前活动 gate：benchmark / diagnosis。** 正例仍以 W7/W8 Goal/Long 为主；
   Object/Spatial 当前 regime 记为 mostly both-fail。新分析必须先登记；不得重开
   selector，也不得为让 split READY 而改 seed/补采。

以下为原执行 gate 记录，保留其预注册语义：

1. **Release gate**：episode-grouped split、cross-oracle 统计与 W4 数据导出。
2. **Coverage gate（已通过）**：L1–L2 failure-challenge 固定为
   `dim × level`，每 cell 两个不同 episode，共 8 states；该 cohort 不能用于
   无条件 NGC rate 或 clean-regret。
3. **Policy-matrix gate（W6 完成、W7 复现待最终核验）**：W7 输出后先冻结 paired
   state result，再只对 discordant state 做归因，不看结果调参。
4. **Control/selector gate（当前代码与 W8）**：先补 24-state direct OFT action，另建
   clean-success control，再用 `CONTINUE / ESCALATE / ABSTAIN` 做 linear/MLP；
   task-held-out 上优于 matched random-trigger 后，才扩展 fallback 或进入 RL。

任何新实验先写明它服务哪个 gate、kill criterion 和预注册主比较。没有对应 gate 的
大规模 rollout 不应启动。
