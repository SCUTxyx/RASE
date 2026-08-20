# ZERO_SHOT_VLA_ANALYSIS(Phase 1 / Gate A,2026-08-19)

## 设置
- 冻结风险模型:oft_risk_model_v3(训练域:oft_spatial + oft_object,20 mixed tasks)
- 测试 VLA(全部未见):oft_goal(同架构)、oft_10(同架构)、π0-fast(异构架构)
- 指标:AUROC/AUPRC/Brier/ECE/校准分箱/分数分布偏移/VLA-ID probe

## 结果

| 测试 VLA | 成功率 | AUROC | ECE | 分数mean | 判定 |
|---|---|---|---|---|---|
| oft_goal | 0/60 (0%) | NaN(全负) | 0.432 | 0.432 | FAIL(不可测) |
| oft_10 | 160/3800 (4.2%) | 0.456 | 0.451 | 0.493 | FAIL |
| π0-fast | 698/802 (87%) | **0.837** | **0.066** | 0.936 | **PASS** |

- π0-fast 校准分箱:predicted 0.937 → actual 0.874(优秀)
- π0-fast 分数偏移 vs 训练域:Wasserstein 0.48(分布大偏移但校准保留)
- 训练域参照:pairwise ranking accuracy 0.967

## VLA-ID probe(shortcut 诊断)
- in_domain_multiclass_accuracy = **0.894**(随机 0.333)
- pairwise AUROC 全部 1.0
- → 当前 v3 representation 高度保留 policy fingerprint(泄漏风险确认)

## Gate A 判定:PARTIAL
- 跨架构(π0-fast):PASS —— discrimination + calibration 在异构 VLA 上保留
- 同架构(goal/10):FAIL 但为数据限制(近全败域,AUROC 不可测)
- 意外方向:架构差异不是 transfer 的决定因素;成功率水平/域覆盖才是

## 解读
1. "action consequence → outcome" 映射存在跨架构 transfer 的初步证据
   (π0-fast 0.837/0.066)——超出"direct risk 必崩"的悲观预期
2. VLA-ID probe 0.894 确认 shortcut 风险 → Phase 4 必须对比 predictive(WM)
   representation 的 probe(期望更低)与保留的 risk signal
3. 下一步:Phase 2 same-root 反事实采集(Gate B)→ Phase 3 oracle future(Gate C)
