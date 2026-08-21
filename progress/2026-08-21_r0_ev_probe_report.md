# RASE R0 执行报告：执行期验证前提探针（2026-08-21）

## 结论（Gate：FAIL）

执行期验证（CheckVLA 式"实际 vs 预期偏差检测"）在 clean LIBERO-Long + π0-fast 上
**无信号**：proprio 演化可预测性 99.3%（predictor 相对恒等基线），预测偏差与终局
结局无关（AUROC 0.43–0.51）。

**机制**：干净 LIBERO 是确定性、零执行噪声的仿真——动作块被完美执行，预测误差
恒≈0，"偏差检测"没有用武之地。失败不是"偏离预期"，而是"预期本身指向失败"。

## 协议

- 数据：E4-0 的 24 决策状态 × 8 候选（同 seed 重采样，chunk 确定性一致）；
  每候选执行 native 8 步，记录 branch-end proprio（8 维：eef_pos + axisangle + gripper_qpos）；
- 终局标签复用 E4-0（192 条真实 rollout）；
- P1：ridge (s_t, chunk24) → s_{t+8}，对比恒等基线；
- P2：预测偏差 vs 终局 success 的 AUROC（全局 + per-state z 标准化）；
- Gate：P1 改善 ≥10% 且 P2 AUROC ≥0.65。

## 结果

| 测量 | 值 |
|---|---|
| 恒等基线 MSE | 0.00134 |
| predictor MSE | 9.18e-06 |
| P1 相对改善 | **99.3%** |
| P2 AUROC（全局） | 0.426 |
| P2 AUROC（per-state z） | 0.509 |
| 失败率按偏差三分位 | 64.1% / 62.5% / 78.1% |
| Gate | FAIL |

## 对证据链的贡献（第八环）

| 环 | 形态 | 失败机制 |
|---|---|---|
| G0-G2b | 策略间切换 | 嵌套失败模式（定理级） |
| G1/E4-0 | 策略内采样候选 | 候选结局无分叉（oracle=0） |
| E3/E3-B | 监督纠正（BC/DAgger） | 教师天花板 |
| **R0** | **执行期验证（偏差检测）** | **确定性仿真：偏差恒≈0，无信号** |

**结论**：验证类方法的自然域是带执行噪声/随机性的环境（真实世界、随机化仿真）。
干净 LIBERO 从机制上排除验证类方法。

## 下一步（R0b，待用户确认）

带噪声执行环境下的"重规划频率"实验：动作加 N(0,σ²)（σ∈{0,0.03,0.06}），
对比每 4 步 vs 每 8 步重规划的终局成功率——若 σ>0 时高频重规划 > 低频，
验证-提前干预有价值（R1 值得做）；若相同，LIBERO 域关闭，转论文。

## 产物

- `runs/e5_ev_probe_v1/rows.jsonl`（192 行：s_t/chunk/s_t8/success）+ summary.json
- 脚本：`scripts/e5_ev_probe.py`、`scripts/run_e5_probe.sh`
