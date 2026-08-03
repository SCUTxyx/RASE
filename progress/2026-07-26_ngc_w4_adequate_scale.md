# NGC W4 ADEQUATE scale：SmolVLA Wilson triage + 四 suite OFT portfolio

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-27 11:51 CST**（`SUMMARY_ONLY=1` 正式 v2 汇总；rollout 已于此前完成） |
| 状态 | **完成（实验）+ 汇总语义已修正（工程）** |
| 问题 | ADEQUATE 扩样到 32 态后，SmolVLA 是否仍全 Set C？suite-matched OFT 的候选组合覆盖率如何？ |
| 配置 | [`configs/ngc_w4_adequate_scale.yaml`](../configs/ngc_w4_adequate_scale.yaml) |
| Keys | [`runs/ngc_w4_adequate_state_keys.json`](../runs/ngc_w4_adequate_state_keys.json) |
| 候选 | [`runs/ngc_w4_adequate_candidates/`](../runs/ngc_w4_adequate_candidates/)（K=8） |
| SmolVLA 产物 | [`runs/ngc_w4_pilot_adequate/`](../runs/ngc_w4_pilot_adequate/) |
| OFT 产物 | `runs/ngc_w4_oft_{spatial,object,goal,10}_adequate/` |
| 双 oracle 汇总 v2 | [`runs/ngc_w4_adequate_dual_oracle_summary.json`](../runs/ngc_w4_adequate_dual_oracle_summary.json) |
| Causal 表 | [`runs/ngc_w4_adequate_causal_yield.json`](../runs/ngc_w4_adequate_causal_yield.json) |
| 母记录 | [W3 ADEQUATE pilot](2026-07-19_ngc_w3_pilot_adequate.md) |
| 后续 | [W5 screen 温度网格](2026-07-27_ngc_w5_smol_screen_t07.md) · [W5 L1–L2 小池](2026-07-27_ngc_w5_l1_l2_pool.md) |
| Git（汇总时 HEAD） | `ea7ad403c002302234cf7aa81476bb869e86b586` |

---

## 1. 一句话结论

W4 ADEQUATE 32 态上，SmolVLA 仍 **32×Set C / 0/1536**；suite-matched OFT 为 **65/256 候选命中**、**17/32 状态至少一候选成功**（portfolio coverage **53.1%**，状态级 95% Wilson CI **[36.4%, 69.1%]**）。交叉标签为 `oft_only`（17）或 `both_fail`（15）。OFT 支持 fallback / portfolio 叙事，**不是** Wilson Set A/B 认证。

---

## 2. 协议与样本审计

| 项 | 约定 |
|---|---|
| SmolVLA | Wilson `n1=6→20`，`τ=0.5`，K=8，cont_temp=0.5 |
| OFT | `deterministic_one_shot`；portfolio = any(success) |
| 采样偏置 | 全部 **t0=0**；level 过滤 L3/L4/L5 = 12/13/7（未按 level 分层） |

---

## 3. 结果摘要

| 量 | 值 |
|---|---|
| SmolVLA | **0/1536**，**32×C** |
| OFT 候选命中 | **65/256** |
| OFT portfolio | **17/32**（CI [0.3645, 0.6913]） |
| camera / robot portfolio | 12/16 · 5/16 |

`SUMMARY_ONLY` 自动 stub 曾覆盖本文件；本版为恢复后的正式记录，并链到最新 W5 进度。
