# R6-C 无世界模型多臂基线的 5-seed OOF 结果：诚实负结果（2026-08-10）

## 摘要

R6-C 无 WM 基线（CandidateArmStudent + 两边界 dwell + 1.64-sigma LCB）的 5-seed task-held-out OOF **未通过 R6-C 阶段门**：两个合格 VLA（pi05_libero、pi0fast_libero）在全部 6 个模式（per-VLA / zero-shot / LOVO）的 5 个训练 seed 上均为 0/5 通过（要求每 VLA ≥4/5 seed 同时满足 success gap ≥ -5pp、false continue ≤ 5%、savings ≥ 20%）。

结果经过两种独立聚合方式的交叉验证（官方 avg-threshold 汇总 + fold-correct 逐折聚合），结论一致。**这是可信的科学负结果，不是实现 bug。**

## 门定义（协议 `configs/r6b1_dynamic_boundary_protocol_v1.json`）

- 评估：5 task-held-out 外折，3 个 task-bootstrap ensemble 成员，1.64-sigma source-success LCB，两边界 dwell。
- 基线：**ENTER_PERSISTENT_OFT at t=0**（无条件立即切入 OFT 的对照）。
- per-policy 门：`success_gap >= -0.05` 且 `false_continue_rate <= 0.05` 且 `teacher_step_savings >= 0.20`；每 VLA ≥4/5 seed。
- `false_continue`：控制器选择"继续 source"（未进入 OFT）但 source 最终失败且 t=0 切入本可成功（persistent[0]=true）的次数；分母为 baseline_successes。
- `savings`：`1 - controller_teacher_steps / baseline_teacher_steps`，对比"t=0 全切"基线。

## 官方结果（`runs/pre_c0_r6/r6c_candidate_arm_oof_v1/stability.json`，avg-threshold 汇总）

| 模式 | success_gap | false_continue | savings | 通过 |
|---|---|---|---|---|
| per_vla_pi05_libero | +28.8pp | 8.6% | 96.5% | 0/5 |
| per_vla_pi0fast_libero | -7.9pp | 2.5% | 8.8% | 0/5 |
| zero_pi0fast→pi05 | +8.2pp | 1.7% | 14.3% | 0/5 |
| zero_pi05→pi0fast | -17.7pp | 65.6% | 96.0% | 0/5 |
| loo_pi0fast | -17.4pp | 63.7% | 92.4% | 0/5 |
| loo_pi05 | +11.4pp | 1.0% | 14.4% | 0/5 |

## Fold-correct 交叉验证（逐折 validation_metrics 聚合，方法学修正）

官方汇总用所有 fold 预测的平均阈值重放全部数据，会混合不同 fold 的阈值（如 pi05 的 {-0.01,1.01,0.996,-0.01,-0.01} 平均为 0.395）。逐折修正后的更严谨汇总：

| 模式 | success_gap | false_continue | savings | 通过 |
|---|---|---|---|---|
| per_vla_pi05_libero | +18.7pp | 7.8% | 73.3% | 0/5 |
| per_vla_pi0fast_libero | -21.5pp | 5.9% | -0.8% | 0/5 |
| zero_pi0fast→pi05 | +4.4pp | 1.6% | 12.4% | 0/5 |
| zero_pi05→pi0fast | -39.2pp | 48.3% | 72.0% | 0/5 |
| loo_pi0fast | -38.1pp | 47.8% | 74.3% | 0/5 |
| loo_pi05 | +7.2pp | 1.3% | 14.3% | 0/5 |

**两种汇总均 0/5 FAIL。** 方法学注记：avg-threshold 汇总低估了 pi05 的 savings/success-gap 并高估了 per-VLA 的夸张表现，但 fold-correct 后所有模式仍失败，因此负结果稳健。后续评估报告应优先采用 fold-correct 聚合。

## 负结果的结构性解读

风险模型本身**学到了有效的 source 失败风险排序**（验证：pi0fast 上 `source_mean` 高 bin 失败率 34% vs 低 bin 70%；LCB 高 bin 失败率 30% vs 低 bin 65%）。但控制器在门指标上失败，原因分 VLA：

1. **pi05（source 92.6% 成功）**：模型正确地"少切"（fold-correct entered 仅 24/95），带来极高 savings 与正 success gap，但**漏掉了 5 个本可被 t=0 切入救回的失败轨迹**（fc=7.8% > 5%）。在"永远全切"的基线对照下，保守策略的 false_continue 被定义为相对 baseline_success 的比例，少量漏网即超限。

2. **pi0fast（source 39.6% 成功）**：模型正确识别风险并频繁切换（entered 45/53），但 **persistent OFT 的救回率随进入时刻急剧下降**（t=0 时 87.5% → elapsed 64 后 ~35%）。控制器在非零边界进入，救回率低于 t=0 全切基线 → savings 相对全切为负、success 反而下降。

3. **跨 VLA 泛化失败**：用 pi05 训练的模型（学到"少切"）在 pi0fast 上几乎不进入（fc≈48-66%）；用 pi0fast 训练的模型在 pi05 上过度切入（entered≈77-83/95，savings 仅 14%）。**"源策略可靠性"是 VLA 特有的，无法通过共享核心零样本迁移。**

## 对后续阶段的约束（按协议 lock）

- **R6-D WM 消融保持密封**（`world_model_lock`：仅当无 WM 基线完成且可做 Pareto 比较时解锁；本结果未过门）。
- **独立验证 / test / 100+ 闭环保持锁定**（`validation_test_lock`：需要两 VLA ≥4/5 seed 通过开发 OOF）。
- 第 3 个 source VLA（计划阶段 6）不启动。

## 与既有证据的一致性

- 与 R6-B0 一致：B0 的 4/5-seed 门也已失败（Pi0Fast per-VLA 0.478 AUROC，接近随机）；本次确认**即使有动态边界 counterfactual 数据，学习到的选择器仍无法同时满足三项门指标**。
- R6-A 的机会上限（Pi0Fast 42.8%、Pi0.5 94.7%）证明**上限存在**，但那是特权（oracle）视角；本负结果证明**无 WM、学习到的 LCB+dwell 无法兑现该上限**，且失败模式（false_continue、晚切救回衰减、跨 VLA 迁移）指向具体机制。

## 产物

- `runs/pre_c0_r6/r6c_candidate_arm_oof_v1/` — 25 个 seed×模式评估（json + log）
- `runs/pre_c0_r6/r6c_candidate_arm_oof_v1/stability.json` — 官方 per-VLA 汇总（avg-threshold）
- `runs/pre_c0_r6/r6b1_b1p2_v1/r6c_candidate_arm_dataset.npz` — 排除 1 个非确定组后的候选臂数据集（143 groups / 767 rows）
- `runs/pre_c0_r6/r6b1_b1p2_v1/parity_audit.json` — PASS 的 parity 门（143 seen / 1 excluded / 0 failures）

## 建议的下一步（供决策）

1. **方法学修正重跑**：改用 fold-correct 逐折聚合作为正式指标（不改变负结论，但修正 pi05/zero-shot 的数字）——低成本。
2. **false_continue 定义复议**：当前分母是 baseline_successes（t=0 全切可救数），对"少切"策略惩罚重；若协议允许，可审议分母为"全部 source-failure episode"或加入"仅当模型在最后一边界前未提示风险"的判定。
3. **失败机制是洞见而非噪音**：晚切救回衰减（pi0fast）与跨 VLA 可靠性迁移失败，应作为下一轮假设（如"仅 t=0/16 决策窗口允许切入"、"source-adaptation 层跨 VLA"）写入 R6-C.1。
