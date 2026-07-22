# NGC W3 科研级续完与双 Oracle 流水线

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-19 21:34 CST** |
| 状态 | **ADEQUATE pilot 完成：SmolVLA 仍 16×C / 0/768；四 suite OFT 齐（spatial 7/32，其余 0）** |
| 目的 | candidate→fork rollout→SmolVLA 随机续完→sequential triage；suite-matched OFT 确定性交叉验证 |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（采集/续完）；`oft`（OFT server / parity） |
| 配置 | `configs/ngc_w3_smoke.yaml`、`configs/ngc_w3_pilot.yaml`、`configs/ngc_w3_pilot_adequate.yaml` |
| 主 CLI | `scripts/rollout_pool_candidates.py` |
| Pilot 产物 | `runs/ngc_w3_pilot_v2/`（`summary.json` + `horizon_diagnosis.json`） |
| ADEQUATE 产物 | `runs/ngc_w3_pilot_adequate/` + `runs/ngc_w3_oft_{spatial,object,goal,10}_adequate/` |
| OFT 产物（旧 16） | `runs/ngc_w3_oft_spatial/`、`runs/ngc_w3_oft_10/`（各自独立 `scheduler/`） |
| 消融产物 | `runs/ngc_w3_cont_ablation_t{00,02,05,10}/` + `runs/ngc_w3_cont_ablation_summary.json` |
| Runbook | `docs/runbooks/ngc_pilot.md` |

前置：

- [W2 candidates pilot](2026-07-18_ngc_w2_candidates_pilot.md)
- Pool：`pool/ngc_step1_scale200`
- Pilot 候选：`runs/ngc_w3_pilot_candidates/candidates`（16×`[8,10,7]`）
- [续完温度消融进度（结果说明）](2026-07-19_ngc_w3_cont_ablation.md)
- [ADEQUATE-only pilot（结果说明）](2026-07-19_ngc_w3_pilot_adequate.md)

### 阶段结论（给报告用）

W3 在冻结协议下对 16 个 Plus 硬失败态得到 **16×Set C / 0/768**。OFT 在同一批候选上救回 1 个 early ADEQUATE spatial 态（8/8）；对该态扫 SmolVLA 续完温度 0.0–1.0 仍 **0/192**。随后在 **ADEQUATE-only** 新 16 态上 SmolVLA 仍 **16×C / 0/768**（0 NARROW），四 suite OFT 为 spatial **7/32**、object/goal/10 **0/32**。主结论是 **SmolVLA 续完能力缺口 + 局部双 oracle 分歧**，不是 NARROW 污染、统计协议或成功判据假阴性；产率估计须限定 ADEQUATE 队列并并列报告 OFT，**仍不盲扩全池**。

---

## 1. 冻结协议

| 参数 | 值 |
|---|---|
| SmolVLA 续完 temp | 0.5（Bernoulli；scaled flow-matching noise） |
| OFT | L1 确定性交叉验证 |
| 候选 | K=8, T=10, temp=0.7，env-space |
| τ | 0.5 |
| n1→n2 | **6→20** |
| α spending | 0.01 + 0.04（one-sided） |
| set_a_min_good | 3 |
| 协议版本 | `wilson-onesided-alpha-spend-v1` |

## 2. 门禁结果

| 门禁 | 结果 |
|---|---|
| `pytest -q` | **通过** |
| 统计校准 Set C FP ≤5% | **通过**（`runs/ngc_w3_calibration_v1`，FP=0） |
| OFT adapter parity | **通过**（`oft_spatial`，`max_abs_diff=0.0`） |
| 20-rollout smoke | **通过**（`runs/ngc_w3_smoke_v2`） |
| 16-state SmolVLA pilot | **完成**：`label_counts={"C":16}`，**0/768 success** |
| OFT suite verify（旧 16） | spatial **8/32**；libero_10 **0/32**；当时 `object`/`goal` 无 ckpt |
| ADEQUATE SmolVLA pilot | **完成**：`label_counts={"C":16}`，**0/768**；`horizon_diagnosis` **0 NARROW** |
| ADEQUATE OFT 四 suite | spatial **7/32**；object/goal/10 **0/32**（ckpt 已齐） |

## 3. Pilot 主结果（科研解读）

| 量 | 值 |
|---|---|
| 状态 | 16（4 suites × {camera,robot} × 2，L3–L5） |
| Set 标签 | **16/16 Set C** |
| 每候选 trials | 恒 **6**（全部 `stopped_early`；0/6 → U≈0.474&lt;0.5） |
| 总 rollout | **768 = 16×8×6** |
| 原始成功 | **0 true / 768 false** |
| stop_reason | **768× `horizon`**（无中途 success） |
| 单次耗时 | median≈2.0 s，p90≈4.8 s |
| 续完步数 | median≈70（多数状态确有 SmolVLA 续完） |

### 可以说

- 在本冻结协议下，对该 **16 状态分层样本**，「换候选 + SmolVLA temp=0.5 续完」**未观察到任何任务成功**；全部 Set C。
- fork / scheduler / α-spending 早停 / triage 汇总与协议一致。
- 与 LIBERO-Plus 上 SmolVLA 近乎塌缩的先验相容，但 **不能**外推为全局 NGC 产率=0。

### 不可以说

- 「换候选永远没用」或「NGC 产率为 0」（n=16、失败偏置池、剩余 horizon 不均）。
- 「统计协议坏了」（0/6→Set C 正是校准行为）。

## 4. 剩余 horizon 诊断

产物：[`runs/ngc_w3_pilot_v2/horizon_diagnosis.json`](../runs/ngc_w3_pilot_v2/horizon_diagnosis.json)

| 桶 | 定义 | 数量 | 含义 |
|---|---|---|---|
| NARROW | remaining ≤ 20 | **4** | 几乎只够跑完候选（`cont_steps=0`）；救回空间极窄 |
| MID | 20 &lt; rem &lt; 100 | 5 | 有限续完 |
| ADEQUATE | remaining ≥ 100 | **7** | 时间够仍 **全为 Set C / 零成功** |

结论：零救回 **不能**仅归因于「没时间」——至少 7 个状态在 ≥100 步剩余下仍全灭。成功判据与 W1 一致读取 `final_info.is_success`。后续 OFT 交叉验证 + 续完温度消融已排除「判据永不亮灯」与「仅 temp=0.5 不对」——见 [消融进度](2026-07-19_ngc_w3_cont_ablation.md)。

## 5. OFT 交叉验证（可用 ckpt）

工程修正：`oft-verify` 强制 `scheduler_root = output_dir/scheduler`，避免与 SmolVLA pilot scheduler 碰撞（此前错误复用 `rollout_index=0`）。本轮结果均为真实 OFT 调用（`oracle=oft`，`rollouts_this_process=32`）。

| Suite | 产物 | 成功 | 解读 |
|---|---|---|---|
| `libero_spatial` | `runs/ngc_w3_oft_spatial` | **8/32** | **1/4 状态 8/8 全救回**（`sp1_c3da4b…`）；其余 3 状态 0/8 |
| `libero_10` | `runs/ngc_w3_oft_10` | **0/32** | 4 状态均全灭（与 SmolVLA 同向） |
| `libero_object` / `libero_goal` | — | **未跑** | 本地无 `ckpts/oft_{object,goal}`（Hub 超时） |

### 双 oracle 对照（相对 SmolVLA 0/768）

- **Spatial 分歧**：同一硬失败态上，OFT 确定性续完可救回整组候选，而 SmolVLA temp=0.5 续完 0 成功 → **不是**「成功判据永远不亮灯」；更像续完策略/模型能力分歧。
- **libero_10 同向全灭**：强化「部分状态本身极难 / 剩余 horizon 过窄」叙事（若干 verify `steps=10` 仅跑完候选）。
- triage 标签在 n=1 确定性 verify 下均为 `uncertain`（预期：单次成功不足以过 Wilson τ 置信门槛）。

## 6. 续完温度消融（E1）

在 OFT 已 8/8 救回的 early ADEQUATE 态 `sp1_c3da4b…`（t0=10，rem=270）上，复用同一组 K=8 候选，扫 SmolVLA 续完温度 **0.0 / 0.2 / 0.5 / 1.0**（其余协议冻结：n1=6，τ=0.5）。

CLI：`--continuation-temperature`（写入 `run_manifest.json`）。配置：`configs/ngc_w3_cont_ablation.yaml`。

| cont_temp | 产物 | 成功 | Set |
|---|---|---|---|
| 0.0 | `runs/ngc_w3_cont_ablation_t00` | **0/48** | C |
| 0.2 | `runs/ngc_w3_cont_ablation_t02` | **0/48** | C |
| 0.5 | `runs/ngc_w3_cont_ablation_t05` | **0/48** | C |
| 1.0 | `runs/ngc_w3_cont_ablation_t10` | **0/48** | C |

汇总：[`runs/ngc_w3_cont_ablation_summary.json`](../runs/ngc_w3_cont_ablation_summary.json) — **0/192** SmolVLA vs OFT **8/8**。

### 可以说

- 在该可救态上，SmolVLA 续完失败 **不是**「碰巧选了 temp=0.5」；greedy→高噪声网格均无成功。
- 相对 OFT 的分歧更像 **续完模型能力缺口**（同候选、同 env、同成功判据）。
- 工程假阴性（判据永不亮灯）已被 OFT 排除。

### 不可以说

- 「换候选永远没用」（未改候选生成 / 未扩池）。
- 「全局 NGC 产率=0」。

## 7. 协议补丁：剩余 horizon 过滤（E2）

- [`rase/collect/stratified_sample.py`](../rase/collect/stratified_sample.py)：`min_remaining_steps` + `DEFAULT_SUITE_HORIZONS`（Spatial/Object=280，Goal=300，Long=520）。
- YAML 透传：`sample.min_remaining_steps` / `sample.suite_horizons`。
- ADEQUATE 模板与执行：[`configs/ngc_w3_pilot_adequate.yaml`](../configs/ngc_w3_pilot_adequate.yaml)；结果见 [ADEQUATE pilot](2026-07-19_ngc_w3_pilot_adequate.md)。

## 8. ADEQUATE-only 复验（已完成）

新 16 keys（seed=1，与旧 pilot 仅重叠 1 key），候选落在 `runs/ngc_w3_adequate_candidates/`（**无**嵌套 `candidates/`）。

| Oracle | 成功 | Set / 备注 |
|---|---|---|
| SmolVLA temp=0.5 | **0/768** | 16×Set C；0 NARROW |
| OFT spatial | **7/32** | `sp1_c947b80…` 7/8 |
| OFT object / goal / 10 | **0/32** 各 | ckpt 已齐并跑完 |

汇总：[`runs/ngc_w3_adequate_dual_oracle_summary.json`](../runs/ngc_w3_adequate_dual_oracle_summary.json)。

## 9. 下一步

1. **仍不盲扩全池**；若估产率，扩大 ADEQUATE 样本并并列 SmolVLA/OFT。
2. 不优先再扫续完温度或改候选生成温度（多样性已过；消融已钉温度）。
3. 报告边界：ADEQUATE-only；局部 spatial 双 oracle 分歧，非全局 OFT 可救。

## 10. 运维备注（原 Blocker 已解除项）

- `oft_object` / `oft_goal`：本地 ckpt 已齐（~15G / suite）；四 suite ADEQUATE verify 完成。
- Pilot / 消融偶发 SIGFPE（exit 136）；resume 同 run root 可续。
- 换 OFT suite 前须先杀掉旧 server（否则 GPU0 OOM）。
