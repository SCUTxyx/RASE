# NGC W3 ADEQUATE-only pilot：SmolVLA + 四 suite OFT

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-19 21:34 CST** |
| 状态 | **完成** |
| 问题 | 排除 NARROW 后，SmolVLA 在 ADEQUATE 队列上是否仍全 Set C？suite-matched OFT 能否并列救回？ |
| 配置 | [`configs/ngc_w3_pilot_adequate.yaml`](../configs/ngc_w3_pilot_adequate.yaml) |
| Keys | [`runs/ngc_w3_adequate_state_keys.json`](../runs/ngc_w3_adequate_state_keys.json)（sample_seed=1，`min_remaining≥100`） |
| 候选 | [`runs/ngc_w3_adequate_candidates/`](../runs/ngc_w3_adequate_candidates/)（**根目录** 16×`.npz`，无嵌套 `candidates/`） |
| SmolVLA 产物 | [`runs/ngc_w3_pilot_adequate/`](../runs/ngc_w3_pilot_adequate/) |
| OFT 产物 | `runs/ngc_w3_oft_{spatial,object,goal,10}_adequate/` |
| 双 oracle 汇总 | [`runs/ngc_w3_adequate_dual_oracle_summary.json`](../runs/ngc_w3_adequate_dual_oracle_summary.json) |
| 母记录 | [W3 流水线](2026-07-19_ngc_w3_pipeline.md) · [续完温度消融](2026-07-19_ngc_w3_cont_ablation.md) |
| Git | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 前缀 | `0609adae34282dfb` |

---

## 1. 一句话结论

ADEQUATE-only（16 态，0 NARROW）上 SmolVLA 仍 **16×Set C / 0/768**。四 suite OFT 中仅 **spatial 7/32**（1 态 7/8），object / goal / libero_10 均为 **0/32**。说明旧 pilot 零救回 **不能**归因于 NARROW 污染；双 oracle 分歧在 ADEQUATE 队列上复现（spatial），但并非各 suite 普遍可救。

---

## 2. 工程对齐

| 项 | 结果 |
|---|---|
| `candidates_dir` | `runs/ngc_w3_adequate_candidates`（匹配实际 npz 落点） |
| `sample.state_keys` | 冻结 16 keys（显式列表优先于重新 stratified） |
| 候选验收 | 16/16 `.npz` 存在 |
| 多样性 | mean endpoint L2 ≈ **1.39**（过 W2 门禁量级） |
| 与旧 pilot 重叠 | **1** key：`sp1_804aeb3b74e569456bb465dd3fa8993e` |
| 误放的 `runs/summary.json` | 已挪至 `runs/ngc_w3_adequate_candidates/summary.json` |

协议冻结（与 W3 一致）：K=8，n1=6→20，τ=0.5，cont_temp=0.5，α-spend；OFT 为 L1 确定性 verify（每候选 1 trial）。

---

## 3. SmolVLA primary

| 量 | 值 |
|---|---|
| Set 标签 | **`{"C": 16}`** |
| 原始成功 | **0 / 768** |
| stop_reason | 全部 `horizon`（见 `horizon_diagnosis.json`） |
| NARROW (rem≤20) | **0** |
| ADEQUATE (rem≥100) | **16 / 16** |
| ADEQUATE 仍 Set C | **16 / 16** |
| wall | ≈1774 s |

产物：[`runs/ngc_w3_pilot_adequate/horizon_diagnosis.json`](../runs/ngc_w3_pilot_adequate/horizon_diagnosis.json)。

### 可以说

- 在冻结协议下，对该 **ADEQUATE-only** 16 态分层样本，SmolVLA 换候选续完仍无任务成功。
- 旧 16×Set C **不是**「NARROW 混进样本」的假象——过滤后结论不变。
- 与 [续完温度消融](2026-07-19_ngc_w3_cont_ablation.md) 同向：主 oracle 续完能力缺口。

### 不可以说

- 「全局 NGC 产率 = 0」（仍是 n=16、失败偏置池）。
- 「换候选永远没用」（未改候选生成；且 OFT spatial 证明至少一态可救）。
- 「OFT 普遍优于 SmolVLA」（仅 spatial 有正例；其余三 suite 0/32）。

---

## 4. OFT 四 suite 交叉验证

每 suite：独立 `--output-dir` + `scheduler/`；`mode=oft-verify`；验收 `rollouts_this_process=32`；结果含 `model_info.name=openvla-oft` / suite-matched ckpt。

| Suite | ckpt | 产物 | 成功 | 解读 |
|---|---|---|---|---|
| `libero_spatial` | `ckpts/oft_spatial` | `runs/ngc_w3_oft_spatial_adequate` | **7/32** | **1/4 态 7/8**（`sp1_c947b80caa10d5ff…`，Spatial camera L4，t0=8，rem=272）；其余 3 态 0/8 |
| `libero_object` | `ckpts/oft_object` | `runs/ngc_w3_oft_object_adequate` | **0/32** | 4 态全灭 |
| `libero_goal` | `ckpts/oft_goal` | `runs/ngc_w3_oft_goal_adequate` | **0/32** | 4 态全灭 |
| `libero_10` | `ckpts/oft_10` | `runs/ngc_w3_oft_10_adequate` | **0/32** | 4 态全灭 |
| **合计** | — | — | **7 / 128** | — |

triage：各 suite 4 态均为 `uncertain`（n=1 确定性 verify 预期行为）。

### 与旧 pilot / 消融的衔接

| 对照 | 旧 W3 pilot | 本 ADEQUATE pilot |
|---|---|---|
| SmolVLA | 0/768，16×C（含 4 NARROW） | 0/768，16×C（**0 NARROW**） |
| OFT spatial | 8/32（`sp1_c3da4b…` 8/8） | 7/32（`sp1_c947b80…` 7/8） |
| OFT object/goal | blocker（无 ckpt） | **已跑，0/32** |
| OFT libero_10 | 0/32 | 0/32 |

旧 OFT 阳性态 `sp1_c3da4b…` **不在**本 ADEQUATE 16 keys 内；本队列另有一 spatial ADEQUATE 态被 OFT 近全救回，而 SmolVLA 同态仍 0 成功 → 双 oracle 分歧在 ADEQUATE 队列上独立复现。

---

## 5. 对报告的含义

1. **产率主表应并列 SmolVLA vs OFT**，且限定 ADEQUATE 队列；勿把 NARROW 混进「没时间」叙事后仍外推。
2. **能力缺口叙事更强**：过滤后 SmolVLA 仍全灭；OFT 仅在 spatial 有局部救回。
3. **仍不盲扩全池**；下一步若估产率，需更大 ADEQUATE 样本 + 明确续完 oracle，而非再扫温度或改候选生成（多样性已过）。

---

## 6. 运行备注

- 必须 `cd /data/data2/yuxuan/RASE`（错误 cwd 会打到空 `pool/`）。
- SmolVLA 偶发 SIGFPE/segfault：同 `output_dir` 去掉 `--force-new-run` resume。
- 换 OFT suite 前须杀掉旧 `rase.oracle.server`（否则 GPU0 OOM）；勿在启动 server 的同一 shell 里 `pkill -f` 自杀。
- ADEQUATE 候选目录 **无** 嵌套 `candidates/` 子目录。
