# R6-C.1A 早切窗口 Model-Free 机会审计（2026-08-10）

## 摘要

用现有 R6-B1.2 采集（143 groups，零新采集）构造五类基于标签的（model-free）策略，审计**早切窗口机会**。本审计**只测量 opportunity，不声称特征可分离性**（可分离性必须由 R6-C.1C 的 task-held-out probe 回答）。

五类策略：
1. `CONTINUE_SOURCE` — 不切换；
2. `ENTER_OFT@t0` — 首边界立即切入 OFT；
3. `CONTINUE_TO_t16_THEN_OFT` — 跑到 t16 判断点，OFT 可救则切换；
4. privileged success oracle — 在成功选项中任选；
5. privileged cost-aware early oracle — 在成功选项中选 OFT teacher steps 最少者。

现有数据边界只有 `t={0,16,32,64,96,128}`（t=8 将在 R6-C.1B 补采），因此本审计的"早切窗口"体现为 **t0 vs t16** 的机会对比。

## 总体结果（143 groups，pooled）

| 策略 | 成功率 | vs ENTER_OFT@t0 gap | OFT savings | paired harm | rescue |
|---|---|---|---|---|---|
| CONTINUE_SOURCE | 76.2% | −11.2pp | 100% | 22.4% | 0% |
| ENTER_OFT@t0 | 87.4% | 0 | 0 | 0% | 22.4% |
| CONTINUE_TO_t16_THEN_OFT | 95.1% | +7.7pp | 26.5% | 4.2% | 18.9% |
| privileged_success_oracle | 99.3% | +11.9pp | 79.9% | 0% | 23.1% |
| privileged_cost_aware_early_oracle | 99.3% | +11.9pp | 80.6% | 0% | 23.1% |

**机会确实存在**：cost-aware oracle 达到 99.3% 成功率且比"t0 全切"节省 80.6% teacher steps。t16 才切换比 t0 立即切换省下的 steps 少（26.5%），印证"早切比晚切便宜"。

## Per-VLA gate（每 VLA 同时满足）

| 条件 | Pi0.5 | Pi0Fast |
|---|---|---|
| cost-aware oracle success gap ≥ −5pp | ✓ (+12.6pp) | ✓ (+10.4pp) |
| OFT savings ≥ 30% | ✓ (95.5%) | ✓ (51.0%) |
| ≥20 decision-divergence groups | ✓ (95) | ✓ (48) |
| ≥10 source-fail & early-rescuable groups | ✗ (**6**) | ✓ (26) |
| 覆盖 4 suites & ≥12 tasks | ✓ (4 / 48) | ✓ (4 / 48) |
| **gate 判定** | **FAIL** | **PASS** |

- **Pi0.5**：唯一失败项是"source-failure & early-rescuable groups ≥ 10"——只有 **6 个**（正是已知的 Pi0.5 source 失败样本不足问题）。这是正例不足，不是方法失败；按计划决策树**进入 R6-C.1B 定向补采**。
- **Pi0Fast**：全部条件满足，**gate PASS**。

## Per-suite 明细（per VLA）

| VLA | suite | groups | tasks | source-fail/rescuable | divergence |
|---|---|---|---|---|---|
| Pi0.5 | Goal | 24 | 12 | 1 | 24 |
| Pi0.5 | Long | 24 | 12 | 2 | 24 |
| Pi0.5 | Object | 24 | 12 | 1 | 24 |
| Pi0.5 | Spatial | 23 | 12 | 2 | 23 |
| Pi0Fast | Goal | 11 | 12 | 7 | 11 |
| Pi0Fast | Long | 12 | 12 | 7 | 12 |
| Pi0Fast | Object | 12 | 12 | 6 | 12 |
| Pi0Fast | Spatial | 13 | 12 | 6 | 13 |

Pi0Fast 的 rescuable 正例均匀分布在 4 suites（每 suite 6-7 个），训练支持充分。

## 关键判读

1. **早切窗口机会真实存在**：cost-aware oracle 在保存 80% teacher steps 的同时将成功率从 76%（continue）提到 99%。
2. **Pi0.5 的机会上限高但正例稀疏**：95 组里只有 6 个 source-failure 可救组，任何学习方法都无法从这里学到通用风险排序——补采是唯一路径。
3. **t16 窗口仍可救 18.9%**，但"只有 t0 可救、到 t16 已不可救"的组占 rescuable 的 16.7%（Pi0.5）/19.2%（Pi0Fast）——**确认早切窗口（t0）的价值**，与 R6-C.0 负结果中"晚切救回衰减"机制一致。
4. 本审计不回答可分离性；R6-C.1C 的 OOF probe 才能回答"学习到的特征能否区分"。

## 产物

- `runs/pre_c0_r6/r6c1_early_window_audit.json` — 完整审计（五策略、per-VLA、per-suite、task-cluster bootstrap、gate 判定）
- `scripts/audit_r6c1_early_window.py` — 审计脚本（零采集，只读 metadata）

## 对后续阶段的约束

- **Pi0.5 → R6-C.1B 补采**（source-failure 正例 6 → 目标 ≥30 组）；不判定方法失败。
- **Pi0Fast → 进入 R6-C.1B/1C**，有机会继续。
- 若 1B 后仍无足够正例，则停止该 policy pair（决策树）。
