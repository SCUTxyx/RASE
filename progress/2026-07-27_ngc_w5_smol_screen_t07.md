# NGC W5 SmolVLA screen：proposal-temperature 网格 + OFT-recovered smoke

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-27 18:43 CST** |
| 状态 | **Screen 完成（t=0.3/0.7/1.0）；Confirm 未跑（全程 0 命中）** |
| 问题 | 在现 L3–L5 池上，换分层 failure / OFT 可救态 / proposal 温度后，SmolVLA 1-shot screen 能否出现非零命中？ |
| 池 | `pool/ngc_step1_scale200`（manifest **0 success / 3666 failure**） |
| Failure keys | [`runs/ngc_w5_failure_frontier_state_keys.json`](../runs/ngc_w5_failure_frontier_state_keys.json)（suite×dim×level，per_cell=1，`max_t0=40`，n=24） |
| Smoke keys | [`runs/ngc_w5_oft_recovered_state_keys.json`](../runs/ngc_w5_oft_recovered_state_keys.json)（W4 `splits.oft_only`，n=17） |
| 配置 | [`configs/ngc_w5_failure_frontier_screen.yaml`](../configs/ngc_w5_failure_frontier_screen.yaml) · [`configs/ngc_w5_oft_recovered_smoke.yaml`](../configs/ngc_w5_oft_recovered_smoke.yaml) |
| 产物 | `runs/ngc_w5_failure_frontier_screen_{t03,t07,t10}/` · `runs/ngc_w5_oft_recovered_screen_t07/` |
| 母记录 | [W4 ADEQUATE](2026-07-26_ngc_w4_adequate_scale.md) · [L1–L2 小池](2026-07-27_ngc_w5_l1_l2_pool.md) |
| Git HEAD | `ea7ad403c002302234cf7aa81476bb869e86b586`（工作树另有未提交加固） |

---

## 1. 一句话结论

W5 在现池上的 SmolVLA screen **全部零命中**：failure frontier 三温度合计 **0/576**，OFT-recovered smoke **0/136**。**Confirm 未跑**（无冻结命中 keys）。与 W4「换候选 + SmolVLA 续完不可救」同向；proposal temperature ∈ {0.3, 0.7, 1.0} **不能**在 L3–L5 ADEQUATE failure 上打开可救边界。

---

## 2. 协议（screen only）

| 项 | 值 |
|---|---|
| mode | `smolvla-screen` |
| K | 8 |
| 每候选 trials | **1** |
| continuation temperature | 0.5 |
| proposal temperatures | **0.3 / 0.7 / 1.0**（failure）；smoke 仅 **0.7** |
| 正式 Set A/B/C | **关闭** |

Diagnostic 标签为 `uncertain`（1-shot 预期），**不得**写成 Wilson Set C。

---

## 3. 结果表

| Cohort | temp | n_states | rollouts | 候选命中 | 状态命中 | wall |
|---|---:|---:|---:|---:|---:|---:|
| OFT-recovered smoke | 0.7 | 17 | 136 | **0** | **0** | 2562 s |
| Failure frontier | 0.3 | 24 | 192 | **0** | **0** | 2326 s |
| Failure frontier | 0.7 | 24 | 192 | **0** | **0** | 2175 s |
| Failure frontier | 1.0 | 24 | 192 | **0** | **0** | 2228 s |
| **Failure 合计** | — | 24×3 | **576** | **0** | **0** | — |

Failure 采样：24 个 `suite×dim×level` cell 各 1 态；切片（suite / dim / level）在各温度下均为 0 命中。

### 可以说

- 在现 L3–L5 池、K=8、1-shot screen 下，**proposal 温度网格未产生任何命中**。
- W4 OFT portfolio 可救的 17 态上 SmolVLA 仍全灭 → 强化 **续完能力缺口**，而非「只抽到特别难的 failure」。
- retained-success 正控制在本池不可行（0 success 快照）；已用 `oft_only` smoke 做流水线自检。

### 不可以说

- 「任意扰动档都不可救」（L1–L2 小池已采，**screen 尚未跑**；见下一条 progress）。
- 「这些态是 Set C」或「可直接进 confirm」。

---

## 4. Confirm

**未执行。** 零命中时无科学意义。

---

## 5. 下一步

1. 对 [`pool/ngc_w5_l1_l2_camera_robot`](../pool/ngc_w5_l1_l2_camera_robot) 做 inventory → sample → screen（建议先 t=0.7）。  
2. 若 L1–L2 仍≈0：主叙事转为 **SmolVLA 提案 + OFT fallback**，停止在同一 frozen 续完协议上堆预算。  
3. 若有命中：冻结 keys 后跑 `configs/ngc_w5_frontier_confirm.yaml`。
