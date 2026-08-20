# R6-C.0 正式冻结：Fold-Correct 聚合的最终报告（2026-08-10）

## 摘要

R6-C 无 WM 基线（CandidateArmStudent + 两边界 dwell + 1.64-sigma LCB）的 5-seed task-held-out OOF 在**方法学修正后的 fold-correct 聚合**下**仍然是 FAIL 0/5**（每 VLA 0/5 通过，要求 ≥4/5）。本报告正式冻结该负结果，**不追溯修改门结论**。

本阶段同时修复了一个实现级 bug（`group_boundaries` 用局部 argsort 索引误当全局行号，导致分组错乱），并基于修复后的逐折评估重新聚合。修复后负结论保持，进一步确认失败是科学性的（模型/控制器/数据问题），而非评估管线 bug。

## 本轮发现并修复的实现 bug

`scripts/train_r6c_candidate_arm_student.py` 中 `group_boundaries`：

```python
order = np.argsort(idx, kind="stable")
for position in order:
    group = str(data["group_id"][position])   # 错：position 是 idx 局部索引
```

当 `idx`（fit/cal/val 行索引）不是从 0 连续开始时（per-VLA/zero-shot/LOO 下几乎总是如此），`position` 会错误地索引 dataset 的前 N 行，导致 trajectory 分组错乱。修复为 `data["group_id"][idx[position]]`。此 bug 只影响评估聚合（`controller_metrics` → 阈值选择与 fold 指标），**不影响模型训练**。修复后已完整重跑 30 个 seed×模式评估。

旧 avg-threshold 结果备份于 `runs/pre_c0_r6/r6c_candidate_arm_oof_v1_avgthr_backup/`。

## 官方 fold-correct 结果（修复后，`r6c_fold_correct_final_report.json`）

逐折聚合口径：每折用自己的训练数据选定阈值，在该折的 held-out 上评估，episode-level counts 跨折、跨 5 seed 求和。

| 模式 | success_gap | false_continue | savings | 通过 |
|---|---|---|---|---|
| per_vla_pi05_libero | +0.1pp | 7.2% | 76.4% | 0/5 |
| per_vla_pi0fast_libero | -16.7pp | 5.7% | 3.1% | 0/5 |
| zero_pi0fast→pi05 | -0.7pp | 0.7% | 12.7% | 0/5 |
| zero_pi05→pi0fast | -42.9pp | 53.3% | 83.2% | 0/5 |
| loo_pi0fast | -37.5pp | 50.2% | 71.5% | 0/5 |
| loo_pi05 | -5.3pp | 0.5% | 4.2% | 0/5 |

> 注：表中 per-VLA 行的数字是"以该 VLA 为目标训练"的模型评估该 VLA 本身；zero/loo 行分别评估跨 VLA 泛化。跨 5 seed 的 seed 级明细见 JSON 报告。

## 新增指标（point estimate + task-cluster bootstrap 95% 区间）

以 per_vla 模式、跨 5 seed pooled 的 decision-metrics 为准（`decision_metrics` 与 `task_cluster_bootstrap_95`）：

- **conditional missed-rescue rate**（`未进入且source失败且t0 OFT成功 / source失败且t0 OFT成功`）：
  - pi05 目标：1.000（95% CI 全为 1.000）——Pi0.5 的全部 6 个失败样本中 t0 可救的正例几乎全部漏切；
  - pi0fast 目标：~0.115-0.154（CI 约 0.02-0.25）——Pi0Fast 漏切相对少，但救回正例基数大。
- **absolute paired harm**（`未进入且source失败且t0 OFT成功 / 全部groups`）：
  - pi05 目标：6.3%（CI 约 2-13%）——绝对占比低但不可忽略；
  - pi0fast 目标：4.2-8.3%（CI 约 0-13%）。
- **rescue rate / intervention burden**：
  - pi05 目标：rescue ~8-30%，burden ~10-39%；
  - pi0fast 目标：rescue ~48-69%，burden ~77-90%。
- **teacher-step savings**（真实 persistent teacher steps）与 avg-threshold 敏感性分析对照见 JSON。

红线遵守：conditional missed-rescue **不做 5% 主门**（Pi0.5 分母仅 6 个正例，统计不稳定），仅报告点估计与 task-cluster 区间。

## 机制解读（与初版负结果一致，分组修复后更精确）

1. **Pi0.5（source 92.6% 成功）**：模型保守少切（burden ~10-39%），savings 高（63-88%），但 false_continue 7.2% 略超 5% 门，且 conditional missed-rescue=100%（漏切全在 t0 可救样本上）。少切策略在"永远全切"基线对照下被 false_continue 惩罚。
2. **Pi0Fast（source 39.6% 成功）**：模型高频切换（burden 77-90%），但 persistent OFT 救回率随进入时刻衰减（t0 87.5% → 64 后 ~35%），进入越晚救回越少 → success_gap 大幅为负（-8%~-25%）、savings 部分 seed 为负。**晚切是失败主因**。
3. **跨 VLA zero-shot/LOO**：以 pi0fast 训练的模型在 pi05 上过度切入（harm ~40-50%），以 pi05 训练的模型在 pi0fast 上漏切（fc 48-66%）。**共享核心纯 zero-shot 不可行**，需要 policy-specific 校准（R6-C.2）。

## 数据与产物 hash 固化

| 产物 | SHA256 |
|---|---|
| collector `collect_r6b1_dynamic_boundaries.py` | `0d512d2c5e37fb0af7ba5a5f3b696c87aa1472053c12605b9f6975a6554d61a3` |
| protocol `configs/r6b1_dynamic_boundary_protocol_v1.json` | `2fe72433239985b2e92ceffdd201748e09900304e5a61f56d419eee97855509c` |
| initial-keys `rase_ui_phase1a_replacement48_initial_keys_v2.json`（state_keys_sha256） | `df345cbd35e22ff1b736893ccd9d67cfb0e6b28b5290fc5e3072517819365065` |
| dataset `r6c_candidate_arm_dataset.npz` | `98f4f3cbf8a1050dd1b23d737794bf3fd278012e177c9eff488d53fcfac92f10` |
| dataset report | `a22ef584b31d6b418d2eb9dd7ff93ea723f31bffc7a2a65925936516f101a7e6` |
| exclusion manifest `r6b1_b12_exclusions_v1.json` | `dc480b38e51a9fd8265efb9cc18f2cf461de7babd8792423d976638c4a134372` |

## 产物

- `runs/pre_c0_r6/r6c_candidate_arm_oof_v1/r6c_fold_correct_final_report.json` — fold-correct 官方报告（含 seed 级指标、bootstrap、avg-threshold 敏感性附录）
- `runs/pre_c0_r6/r6c_candidate_arm_oof_v1/stability.json` — 门判定（FAIL 0/5，修复后重跑）
- `runs/pre_c0_r6/r6c_candidate_arm_oof_v1_avgthr_backup/` — 旧 avg-threshold 结果备份
- `scripts/aggregate_r6c_fold_correct.py` — 聚合工具

## 对后续阶段的锁定

- R6-C 记录为 **FAIL 0/5**，不追溯改门。
- R6-D WM 消融保持密封（需双 VLA 无 WM R6-C.1 gate 通过后解锁）。
- 独立验证/test/100+ 闭环保持锁定。
- 第 3 个 source VLA 不启动。

## 下一步（R6-C.1）

机会审计（R6-C.1A）与数据补采（R6-C.1B）路径已由本报告确认：Pi0.5 source-failure 正例仅 6 个（<10 门槛），必须补采；Pi0Fast 有充足正例但暴露晚切救回衰减，需要早切窗口（t0/t8/t16）控制。
