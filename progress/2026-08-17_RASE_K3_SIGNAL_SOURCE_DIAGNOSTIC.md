# RASE K3 信号来源诊断(E2/E3 ladder + operator 剥离)——2026-08-17 深夜

## 分析修正(重要)

在深挖时发现两个分析缺陷,已修复并重跑:

1. **continue 候选 chunk 只有 1 步**:K3-E0 的 capture 语义里,continue = `env_chunk[cursor:]`(cursor=9 时只剩 1 步)。与 requery/fallback(10 步)对比时统计特征不可比(std=0 等)。**修复**:continue 改用 capture 中保存的 `full_env_chunk`(该次推理的完整动作块,与 requery/fallback 同构)。
2. **primary target 用了 utility**:utility 混合了 operator 级成本(latency/query),使"动作→结局"信号被成本污染(source-source 曾出现 accuracy<0.5 的反向现象)。**修复**:same-root pairwise ranking 的 primary target 改用 **success**(协议 §6.1),utility 仅保留在 offline_utility。

## 修正后结果(success target,8 tasks × 3 roots × K3)

### E2/E3 特征 ladder(全 cohort,same-root pairwise,task-held-out ridge)

| 特征 | OOF pairwise acc | gain vs state-only |
|---|---:|---:|
| state-only(proprio) | 0.000 | — |
| **raw-action stats** | **0.750** | **+0.750** |
| trace-only(MotionTrace) | 0.556 | +0.556 |
| trace+semantic | 0.556 | +0.556 |

- task bootstrap mean gain +0.656,95% CI [0.396, 0.896];7/8 tasks 正向;
- 324 informative pairs(二元 success 下非 tie 对)。

### Operator 剥离诊断(信号来源分解)

| 子集 | state-only | raw-action | trace-only | informative pairs |
|---|---:|---:|---:|---:|
| **source-source**(continue vs requery,同策略不同采样) | 0.000 | **0.625** | 0.500 | 72 |
| cont-vs-fb | 0.000 | 0.714 | 0.643 | 126 |
| req-vs-fb | 0.000 | **0.857** | 0.643 | 126 |

### 结论

1. **信号不是纯 operator 先验**:source-source(两个候选来自同一 policy、同一 root、仅采样 seed 不同)的 raw-action pairwise accuracy = 0.625 > 0.5——**同策略内动作内容可预测结局**,这是"动作语义信号"的直接证据(72 pairs,方向明确)。
2. **fallback 分布差异贡献最大**:req-vs-fb 0.857 / cont-vs-fb 0.714 高于 source-source——operator 先验存在且重要,但不是全部。
3. **raw-action stats 强于 MotionTrace**(0.750 vs 0.556):当前 MotionTrace 特征(速度/加速度/路径等)不如动作统计特征。E3 特征需要重新设计或与 raw stats 组合(trace+semantic 0.556 未超过 raw——组合方式需调整)。
4. selector utility 修正后 0.810(gain +0.352 vs continue,regret ↓69.5%),已接近 always-fallback(0.858)——成本感知 selector 现在有竞争力。
5. 校准:ΔBrier -0.047、ΔECE -0.015(相对 state-only 改善)。

## 待办

- [x] 信号来源诊断(本记录)
- [ ] K5 扩大 cohort 至 48 tasks(确认信号不稀释,Phase-C 教训)
- [ ] semantic selector 特征设计(基于诊断:raw stats 为主 + 需重设计的 trace)
