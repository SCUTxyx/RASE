# RASE Deployment-v2 判定(2026-08-18,A 线收尾)

## 一句话结论

> **Deployment-v2 FAIL(全部预注册场景)**。learned selector 在 S0 现行成本、
> S1 延迟预算、S2 fallback 配额三个预注册场景下均无 break-even;oracle 上界
> 显示真实改进空间仅 0.7%(fallback 非最优 1/144 units)。按计划停机规则:
> **关闭 learned-selector 部署主线**;动作信号可学习性结论保持(Signal PASS 冻结)。

## A0 成本账本

- `runs/rase_vnext/frozen/cost_ledger_v1.json`(sha256 `28ee4ad4…`)
- 144 units / 432 ledger rows;sunk prefix 与 incremental cost 分离;
- 缺失率:`source_prefix_wall_s` 100% 缺失(未落盘,如实标记);
  `incremental_wall_s_vs_continue` 存在负值 → 计时边界不可靠(已在文档标注,
  S1 场景因此无法基于 wall 生效,退化为 S0)。

## A1 场景协议

- `protocols/selector_deployment_scenarios_v1.json`(冻结,sha256 见报告):
  S0 current-cheap-fallback(复现 FAIL)/ S1 latency-budgeted / S2 fallback-constrained;
  λ 预算映射 + oracle 停机规则预注册。

## A2 Oracle 可行性

| 指标 | 值 |
|---|---|
| fallback 非最优 units | **1/144(0.7%)**,1 root、1 task |
| recoverable success(continue 败→fallback 胜) | 48.6% |
| fallback success rate | 95.1%(CI [0.910, 0.986]) |
| S0 oracle gap(λ=0.1) | 0.041(主要为成本规避,非成功改进) |
| S1 oracle gap(全部预算) | **0.007**(= 1/144 的改进,无实际空间) |
| S2 oracle gap(quota 0.3-0.7) | 0.028-0.076(反事实场景) |

**停止规则判定**:S1 的 oracle 空间 ≈ 单 unit 改进,任何非 oracle selector 无法
净赚 → S1 关闭;S0 差距本质是成本规避;仅 S2(配额反事实)存在名义空间。

## A3 margin_λ OOF(outer 一次评估)

| 场景 | selector | continue | fallback | quota-fb | oracle | 判定 |
|---|---:|---:|---:|---:|---:|---|
| S0(λ=0.1,margin=0) | 0.889 | 0.472 | 0.951 | — | 0.958 | **FAIL** |
| S1(退化 S0,wall 不可靠) | 0.889 | 0.472 | 0.951 | — | 0.958 | FAIL(数据局限) |
| S2(quota 0.3) | 0.556 | 0.472 | 0.951 | 0.604 | 0.667 | **FAIL**(sel < quota-fb) |

- margin_λ 在 calib 上冻结(全部选 0.0,即不 abstain——abstain 在 calib U 口径下
  无增益);
- S2 的 selector(0.556)低于 quota-aware fallback(0.604)——即使 fallback 受限,
  简单配额策略仍优于 learned selector。

## 根因(与 Signal 结果一致)

1. fallback 非最优仅 0.7%:改进空间不存在,不是模型不够好;
2. fallback 成功率 95.1% 且成本低:无 break-even 区间;
3. action 信号真实(Signal PASS)但**该信号在当前任务/成本结构下无部署价值**。

## 最终判定与后续

- **关闭 learned-selector 部署主线**(停止 selector 调参/closed-loop);
- Signal PASS 冻结:动作信号可学习性(48-task OOF + K3 确认)仍是有效科学结果;
- 科学结论收窄:动作反事实信号可预测,但 fallback 主导场景下选择器无实用空间;
- B 线(π0.5 cross-policy challenge)独立进行中,不受 A 线影响。
