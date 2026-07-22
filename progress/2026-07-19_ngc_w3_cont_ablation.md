# NGC W3 续完温度消融：结果说明与进度

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-19 19:05 CST** |
| 状态 | **完成** |
| 问题 | Pilot 零救回是「温度选错 / 判据假阴性 / 没时间」，还是 SmolVLA 续完能力不足？ |
| 配置 | `configs/ngc_w3_cont_ablation.yaml` |
| CLI | `scripts/rollout_pool_candidates.py --continuation-temperature` |
| 状态 | `sp1_c3da4b196dca67cb4f845a7ffc4fcfe3`（Spatial camera L3；t0=10，rem=270） |
| 候选 | 复用 `runs/ngc_w3_pilot_candidates/candidates`（K=8，同 artifact） |
| 对照 | OFT L1 verify **8/8**（`runs/ngc_w3_oft_spatial`） |
| 汇总 | [`runs/ngc_w3_cont_ablation_summary.json`](../runs/ngc_w3_cont_ablation_summary.json) |
| 母记录 | [W3 流水线](2026-07-19_ngc_w3_pipeline.md) |

---

## 1. 一句话结论

在 **OFT 已证明可救** 的 early ADEQUATE 硬失败态上，SmolVLA 用同一组候选、扫续完温度 **0.0–1.0**，仍 **0/192 成功**。  
说明 W3 pilot 的零救回主要是 **SmolVLA 续完相对 OFT 的能力缺口**，不是「碰巧温度不对」，也不是「成功判据永远不亮灯」。

---

## 2. 这个结果说明了什么

| 假设 | 是否成立 | 依据 |
|---|---|---|
| 统计协议坏了（早停误杀） | **否** | 0/6→Set C 是校准行为；消融同样全 Set C |
| 成功判据假阴性 | **否** | 同态同候选上 OFT **8/8** 成功，判据能亮灯 |
| 「没时间救」 | **否（对该态）** | t0=10，remaining=270；续完步数≈260 |
| 只是 temp=0.5 选错 | **否** | 0.0 / 0.2 / 0.5 / 1.0 均为 0/48 |
| 候选本身「有毒」、任何 oracle 都救不回 | **否** | OFT 用同一批候选 8/8 救回 |
| SmolVLA 续完在该可救态上失效 | **是（本实验支持）** | 0/192 vs OFT 8/8，同 env / 同判据 |

### 对 NGC 叙事的含义

1. **主 oracle（SmolVLA）在 Plus 硬失败池上救回极难**：16-state pilot 0/768 与消融 0/192 同向。
2. **双 oracle 分歧是真实科学信号**：换更强的 suite-matched OFT 续完，至少有一态可全救回 → NGC「换候选」对 **续完策略** 敏感，不能只报 SmolVLA 产率。
3. **下一轮协议应排除 NARROW 态**（`min_remaining_steps≥100`），避免把「没时间」混进 Set C 主结论；模板见 `configs/ngc_w3_pilot_adequate.yaml`。
4. **仍不盲扩全池**：机制已钉住，产率估计要在 ADEQUATE 队列 + 明确续完 oracle 下重做。

### 明确不可以说

- 「换候选永远没用」（未改候选生成策略，且 OFT 证明候选可续完成功）。
- 「全局 NGC 产率 = 0」（n 小、失败偏置池、仅测到可用 OFT suite）。
- 「温度网格穷尽了所有续完失败原因」（未测更长候选、不同采样步、clean LIBERO 阳性对照等）。

---

## 3. 实验数字

| cont_temp | run | successes / trials | Set |
|---|---|---|---|
| 0.0 | `runs/ngc_w3_cont_ablation_t00` | 0 / 48 | C |
| 0.2 | `runs/ngc_w3_cont_ablation_t02` | 0 / 48 | C |
| 0.5 | `runs/ngc_w3_cont_ablation_t05` | 0 / 48 | C |
| 1.0 | `runs/ngc_w3_cont_ablation_t10` | 0 / 48 | C |
| **合计** | — | **0 / 192** | — |
| OFT 对照 | `runs/ngc_w3_oft_spatial`（该态） | **8 / 8** | — |

协议冻结：K=8，n1=6→n2=20，τ=0.5，α-spend 0.01+0.04；每温独立 `output_dir/scheduler`。  
每条 rollout 记录含 `oracle=smolvla`、`continuation_temperature`、`continuation_steps≈260`、`stop_reason=horizon`。

---

## 4. 工程与协议落地

- `--continuation-temperature` 覆盖 YAML，并写入 `run_manifest.json`。
- 分层采样支持 `sample.min_remaining_steps`（suite horizon：Spatial/Object 280，Goal 300，Long 520）。
- 消融中 t=0.2 曾 SIGFPE（exit 136），同目录 resume 后补齐；不影响结论。

---

## 5. 下一步

1. ~~补 `oft_object` / `oft_goal` ckpt，补齐四 suite 交叉验证。~~ → 见 [ADEQUATE pilot](2026-07-19_ngc_w3_pilot_adequate.md)。
2. ~~用 `configs/ngc_w3_pilot_adequate.yaml` 做 ADEQUATE-only 标注。~~ → 完成：SmolVLA 仍 16×C / 0/768。
3. 再谈扩池或改候选多样性 / 更早采样步；主文报告应并列 SmolVLA 与 OFT 续完（**仍不盲扩全池**）。
