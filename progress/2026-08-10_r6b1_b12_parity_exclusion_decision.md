# R6-B1.2 Parity 硬门：单状态非确定性排除决策（2026-08-10）

## 摘要

B1.2 全量采集的 source-parity 硬门在 1 条轨迹上失败（6 个 boundary 全部触发）：`pi05_libero` seed1，Goal suite `libero_goal_000010`，state `sp1_b0b5e524da0d318935146d898a89ef8c`。参考（R6-A）记录 154 步，B1.2 采到 138 步。其余 143 条轨迹全部 PASS。

通过系统复现实验确认：**该状态自身的 source rollout 长度本质非确定**，不是采集器回归。按最严格处理原则，将该轨迹组从 R6-C 训练数据集中**排除**，并冻结排除清单。

## 证据链

同 checkpoint、同 rollout_seed（3295964530）、同 snapshot 下的完整重跑分布：

| 实验 | 方法 | 步数 | 成功 |
|---|---|---|---|
| R6-A 历史记录 | R6-A 脚本（旧 run） | 154 | true |
| R6-A 重跑 #a | R6-A 脚本（当前代码） | 153 | true |
| R6-A 重跑 #b | R6-A 脚本（当前代码） | 138 | true |
| R6-A 重跑 #c | R6-A 脚本（当前代码） | 153 | true |
| R6-A 重跑 #d | R6-A 脚本（当前代码） | 138 | true |
| B1.2 全量采集 | B1.2 collector | 138 | true |
| B1.2 单独重跑 | B1.2 collector | 138 | true |

**结论**：当前冻结代码下 R6-A 参考值 154 无法复现（4 次重跑 = {153, 138, 153, 138}）。B1.2 采集的 138 可复现（B1.2 2 次 + R6-A 脚本重跑 2 次均 138）。`rollout_seed` 与 `source_final_success` 与参考完全一致，唯一差异是步数，且该差异落在状态自身非确定分布内。

## 为什么判定为"状态特性"而非"采集器回归"

1. **采集器稳定**：B1.2 collector 两次独立运行（全量 + 单轨迹重跑）均产出 138 步。
2. **R6-A 脚本自身不稳定**：同一 checkpoint/seed/snapshot 下，R6-A 脚本重跑 4 次得到两个不同的值（153 出现 2 次，138 出现 2 次），说明参考值本身不是确定性真值。
3. **143/144 全部通过**：若为采集器系统性回归，应出现系统性偏差，而非单状态。
4. **边界状态特征**：同状态兄弟 runs 均接近成功终止——pi0fast seed0/1 = 155/155，pi05 seed0 = 142。该状态处于成功边界，微小扰动即改变 rollout 长度。

## 处理决定

- **排除** `(pi05_libero, seed1, sp1_b0b5e524da0d318935146d898a89ef8c)` 整组（1 组、6 个 boundary rows）于所有下游产物。
- 冻结排除清单：`runs/pre_c0_r6/r6b1_b12_exclusions_v1.json`。
- parity 硬门在排除后 **PASS**（143 seen / 1 excluded / 0 failures），audit 输出含 `n_excluded_trajectories` 与排除明细，保证决策可审计。
- 下游数据管道（candidate-arm build、analysis、R6-D WM cache）均支持 `--exclusions`，保持一致。

## 对数据可信度的影响

- 训练数据集：143 groups / 767 rows（原 144/773）。四个 suite、两个合格 policy 保持完整。
- 该状态在数据集中仍以其他 policy/seed 形式存在（pi0fast seed0/1、pi05 seed0），即 `n_states=48` 不变，但该特定高方差轨迹组不再作为训练/评估样本。
- within-horizon 标签（within8/16/32）依赖 source 步数，在非确定状态下本身不稳定；排除消除了这部分标签噪声。

## 产物清单

- `runs/pre_c0_r6/r6b1_b12_exclusions_v1.json` — 冻结排除清单（含证据）
- `runs/pre_c0_r6/r6b1_b1p2_v1/parity_audit.json` — 重新生成的 PASS 审计
- `runs/pre_c0_r6/r6b1_b1p2_v1/r6c_candidate_arm_dataset.npz` — 排除后正式数据集
- `runs/pre_c0_r6/r6b1_b1p2_v1/candidate_arm_analysis.json` — 排除后分析
- `progress/2026-08-10_r6b1_b12_candidate_arm_analysis.md` — 分析报告

## 涉及脚本

- `scripts/audit_r6b1_source_parity.py` — 新增 `--exclude`
- `scripts/build_r6c_dynamic_dataset.py` / `scripts/build_candidate_arm_dataset.py` — 新增 `--exclusions`
- `scripts/analyze_r6c_candidate_arms.py` — 新增 `--exclusions`
- `scripts/cache_r6d_wm_features.py` — 新增 `--exclusions`（R6-D 预留）
- `scripts/run_r6c_pipeline_after_b12.sh` — 传递 `$EXCLUSIONS`
- `scripts/run_r6c_candidate_arm_oof.sh` — 构建时传递 `--exclusions`
