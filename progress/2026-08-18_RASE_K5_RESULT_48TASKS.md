# RASE K5 扩大验证结果(48 tasks)——2026-08-18

## 1. 一句话结论

> K5(48 tasks × 1 root × 6 ops × K3 = 864 slots)完整收集并分析:
> **same-root 动作信号在 48 tasks 上不稀释且更强**(raw-action pairwise acc 0.787,
> K3 为 0.750;4/4 folds 同向;task bootstrap 95% CI [0.359, 0.625] 下界 > 0;
> 28/48 tasks 方向为正,远超 Phase-C 的 13 informative tasks)。
> **source-source(同策略 continue vs requery)acc 0.548 > 0.5**——同策略动作语义
> 在 48 tasks 上可学习,确认信号不是纯 operator 分布先验。

## 2. 冻结产物

- manifest:`runs/rase_vnext/frozen/k5_cohort_manifest_v1.json`
  (sha256 `b55fc0ddd88bdf34ad6b0d63adc6f4a4642a40f7e6733e8bc4df7d37be81372d`)
- 收集:`runs/rase_vnext/k5_collect_v1/`(144 groups、864 rows COMPLETE、
  432 executable、271 success、captures 全量 v2)
- 分析:`runs/rase_vnext/k5_collect_v1/k3_analysis.json`(复用分析器)

## 3. 特征 ladder(48 tasks,same-root pairwise,task-held-out ridge)

| 特征 | OOF pairwise acc | gain vs state-only |
|---|---:|---:|
| state-only | 0.000 | — |
| **raw-action** | **0.787** | **+0.787** |
| trace-only | 0.737 | +0.737 |
| trace+semantic | 0.703 | +0.703 |

- 4/4 folds 方向一致;task bootstrap mean +0.491,95% CI [0.359, 0.625];
- **28/48 tasks 正向**(Phase-C 同口径仅 13/48);
- 536 informative pairs(二元 success 非 tie 对)。

## 4. Operator 剥离(48 tasks)

| 子集 | raw-action acc | 解读 |
|---|---:|---|
| **source-source**(continue vs requery) | **0.548** | 同策略、同 root、仅采样不同 → 动作内容可预测结局(语义信号) |
| continue vs fallback | ~0.55-0.63 | fallback 分布差异 |
| requery vs fallback | 0.787 | 最大贡献(operator 先验) |

结论:信号 = operator 分布先验 + 同策略动作语义,两者都在 48 tasks 上稳健;
source-source 0.548 虽然幅度中等,但 144+ pairs 统计上方向明确。

## 5. 实用性与校准

- selector utility 0.679 vs continue 0.458(+0.221,regret ↓46.6%);
  vs always-fallback 0.901 —— selector 仍低于 fallback(成本权重下的权衡,不变);
- 校准:ΔBrier -0.027、ΔECE -0.012(相对 state-only 改善);
- swap:0.703→0.702(不退化),预测敏感 90%。

## 6. K3/K5 对比(信号稳健性)

| 指标 | K3(8 tasks) | K5(48 tasks) |
|---|---:|---:|
| raw-action pairwise acc | 0.750 | **0.787** |
| source-source acc | 0.625 | 0.548 |
| folds 同向 | 4/4 | 4/4 |
| tasks 正向 | 8/8 | 28/48 |
| task bootstrap CI | [0.396, 0.896] | [0.359, 0.625] |
| informative pairs | 324 | 536 |

信号在规模扩大后保持甚至增强;informative task 占比 28/48(58%),显著高于
Phase-C 的 13/48(27%)——K5 的动作级数据质量优于历史 source-prefix 数据。

## 7. 限制(不变的口径边界)

- 单策略(pi0-fast)、单 decision point;multi-VLA / closed-loop / OPD 仍 locked;
- selector < always-fallback(成本场景才有价值);
- source-source 0.548 幅度中等(统计显著但效应中等)。

## 8. 下一步(建议顺序)

1. **semantic selector 训练协议冻结**(E2/E3 特征定稿:raw-action stats 为主 +
   trace 重设计;目标 ΔU(success)同 root pairwise,tie margin abstain);
2. **π0.5 独立 challenge**(8-task × K3,复用链路,验证 policy 泛化性);
3. K5 数据 + selector 的 closed-loop simulator 验证(离线选择 → 真实 rollout,
   与 continue/always-fallback 对比);
4. 上述 PASS 后才考虑 OPD/RL/world-model 或第三 VLA。
