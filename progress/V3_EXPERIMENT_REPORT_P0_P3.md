# RASE v3 实验报告:Phase 0-3(2026-08-19)

## 实验链
P0 zero-shot(Gate A)→ P1 same-root 采集(Gate B)→ P2 oracle future(Gate C)→ P3 WM MVP(Gate D)

## Gate A:Current OPD Zero-Shot Falsification → PARTIAL
| 测试 VLA | 成功率 | AUROC | ECE | 判定 |
|---|---|---|---|---|
| oft_goal | 0% | NaN(全负) | 0.432 | FAIL(不可测) |
| oft_10 | 4% | 0.456 | 0.451 | FAIL |
| π0-fast | 87% | **0.837** | **0.066** | **PASS** |
| VLA-ID probe | — | 0.894 | — | 泄漏确认 |
- 跨架构(π0-fast)transfer 意外存在:discrimination + 校准保留
- representation 保留 policy fingerprint(0.894)→ Phase 4 对照基准

## Gate B:Same-Root Counterfactual Opportunity → PASS
- 64→216 roots;物体位姿分叉 0.059→0.0795;fraction 58%→PASS
- within-state advantage 100%;h_within=0(任务内最佳候选不翻转)

## Gate C:Oracle Future Risk Upper Bound → PASS(极强)
- oracle discrimination AUROC **0.9998**;pairwise consistency 0.984
- GT future 含充足排序信号

## Gate D:WM MVP(B0/B1/B2 + LOVO × 3)→ FAIL
| 留出 VLA | B0 | B1 | B2 | 判定 |
|---|---|---|---|---|
| oft_goal | 0.968 | 0.944 | 0.946 | B2≈B1 |
| oft_object | 0.939 | 0.956 | 0.942 | B2<B1 |
| oft_spatial | 0.968 | 0.951 | 0.961 | B2≈B1 |
- **跨 VLA 风险转移极强**:所有模型 LOVO AUROC 0.94+,retention 0.90+
- **future-bottleneck 无显著增益**(±0.01);B0 最强
- Kill 4 执行:不把 WM 作为主论文

## 最终科学结论
1. Same-state cross-VLA counterfactual protocol 成立(B/C PASS)
2. Risk learned from same-root data transfers across unseen VLAs(retention 0.9+)
3. Future-bottleneck 不提供额外 transfer(与 direct 持平)
4. 特征简洁性 = transferability(B0 最强)
5. 论文定位调整:same-root 协议 + transferable risk selector(不硬推 WM)
