# RASE G1 多尺度同策略重生成执行报告

日期：2026-08-20

## 结论

G1 已完成，预注册结论为 `fail_do_not_scale`。因此不采集独立确认集、不训练 verifier、不运行闭环。

这次实验没有改变 RASE idea：仍然使用冻结 source VLA、same-root 多候选、真实终局标签、common continuation RNG 和最后手段的跨策略 fallback。改变的只是 `resample.source` 的生成覆盖：从单一 temperature 扩展为多尺度原生 flow-noise sampling。

多尺度候选在 8 个开发 roots 上把首候选成功率 62.5% 提升到 oracle@8 75%，但全部 +12.5pp 增益来自同一个 task/root。预注册 Gate 要求至少 2 个 `candidate 0 失败 → 后续候选成功` roots，实际只有 1 个，且 mixed roots 也只有 1/8。

因此高温候选提供了一个真实但孤立的 rescue，尚不足以证明可迁移的风险引导重生成机会。

## 代码完善

### Candidate artifact v2

- `CandidateMetadata` 新增逐候选 `temperatures`；
- 非均匀 schedule 时，旧的单值 `temperature` 明确为 `None`，不再错误表示整组候选；
- `.npz` 同时保存 `temperatures[K]`、seeds、shape、policy hash 和动作；
- loader 向后兼容 artifact v1，旧单温度文件加载后自动得到重复的 per-candidate schedule；
- 已验证旧 K=4、temperature=0.7 artifact 可无修改读取。

### 多尺度生成与实验管线

- `generate_candidates` 支持显式 per-candidate temperature schedule；
- `generate_pool_candidates.py` 从配置解析、验证并记录完整 schedule；
- `pool_candidates.py` 的 resume/skip 判定比较完整 schedule，避免错误复用另一组候选；
- 新增 G1 配置、可恢复运行脚本和 development-only 分析器；
- probe 分析器按 candidate index 与 temperature 汇总真实 success，并输出是否允许独立确认。

测试结果：25 passed。额外完成 artifact v2 round-trip、旧 v1 读取、2-root/16-rollout GPU smoke。smoke 中每个 root 的 8 个分支共享一个 continuation seed，不同 root 使用不同 seed。

## 实验协议

- Cohort：4 个 clean Goal tasks × step 2/4，共 8 roots；只按 metadata 选择。
- 用途：development-only generator selection；不得当作 held-out 证据。
- K=8 schedule：`[0.5, 0.5, 0.3, 0.3, 0.7, 0.7, 0.9, 0.9]`。
- Candidate 0：temperature 0.5，作为预注册基线。
- 候选：10-step 原生 SmolVLA flow-noise chunks；每个候选有独立 generation seed。
- 后果：候选 chunk 后继续运行冻结 SmolVLA 至真实 success/horizon。
- 方差控制：同一 root 的 8 个候选共享 continuation RNG。
- 计算量：64 条真实终局 rollouts；生成 41.2s，rollout 783.2s；单条中位数 7.24s。
- Gate：至少 2 个 `first fail → later success` roots，才允许收集新的独立确认集。

## 结果

| 指标 | 结果 |
|---|---:|
| Candidate 0 success | 5/8 = 62.5% |
| Oracle@8 success | 6/8 = 75.0% |
| Oracle@8 − candidate 0 | +12.5pp |
| First-fail/later-success roots | 1/8 |
| Rescue 覆盖 tasks | 1/4 |
| Mixed-outcome roots | 1/8 = 12.5% |
| Gate | FAIL（1 < 2） |

按 temperature 聚合：

| Temperature | Success |
|---:|---:|
| 0.3 | 10/16 = 62.5% |
| 0.5 | 10/16 = 62.5% |
| 0.7 | 10/16 = 62.5% |
| 0.9 | 12/16 = 75.0% |

唯一 mixed root 的结果为：

```text
T=0.5: fail, fail
T=0.3: fail, fail
T=0.7: fail, fail
T=0.9: success, success
```

这是有价值的机制证据：该 root 上 rescue 对两个 0.9 seeds 都成立，不是一个偶然样本。但另外两个困难 roots 在所有 8 个候选下仍然全失败；其余五个可解 roots 在所有温度下全成功。主要结构仍是 root/task difficulty，而非普遍的候选级互补。

## Gate 后决策

由于 G1 FAIL：

- 不扩大 K；
- 不在相同 roots 上继续调 temperature；
- 不收集所谓 held-out confirmation；
- 不用这个单一 mixed root 训练 verifier；
- 不运行 Risk-Guided Regeneration 闭环。

下一条最高信息增益路线转到既定的 G2 Cross-Policy Goldilocks Screen：

1. 测量 π0-fast 在 clean LIBERO Long 的直接成功率；
2. 只有成功率落在 30%–70% 窗口内才进入 same-root mini screen；
3. mini screen 必须同时出现 continue-only 与 fallback-only，并满足 `H_within ≥5%`、oracle gain ≥5pp；
4. 若仍发生单方支配，则停止该 policy pair，不训练风险模型。

这保留 RASE 的跨策略、same-root 反事实、零样本 policy-invariance 和保守仲裁主线，同时避免继续在 temperature 噪声轴上消耗算力。
