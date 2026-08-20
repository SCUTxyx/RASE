# RASE 下一阶段落地执行计划：Same-State Counterfactual Residual Risk (CRR)

> 日期：2026-08-20
> 服务器：AutoDL `connect.bjb2.seetacloud.com:28921`（RTX 5090 32G，当前 GPU 空闲）
> 上游规划：用户《RASE 下一阶段研究与实验执行规划》（不修改 RASE idea，先排除结构性失败，再投入算力）
> 本文件将 idea/进度/问题三文档（`progress/RASE_IDEA.md`、`RASE_EXPERIMENT_PROGRESS.md`、`RASE_PROBLEMS.md`）与服务器实际资产核对后，转成可执行的阶段计划。

---

## 0. 现状核实（上服务器验证过的事实）

### 0.1 资产清单（全部在 `/root/autodl-tmp/RASE/`）

| 资产 | 位置 | 用途 |
|---|---|---|
| same-root 数据集 **648 行 = 216 roots × 3 candidates** | `runs/oft_opportunity/same_root_w1.jsonl` | P1/P2 主数据 |
| 决策点数据（分支轨迹、无 future） | `runs/oft_opportunity/dp_collect_{spatial,object,goal,10,pi0fast}.jsonl` | 分布 gap 对照（P0） |
| B1 risk 模型 + LOVO 变体 | `runs/oft_opportunity/same_root_risk.npz`（+`_lovo_*`） | 闭环 v2 失败模型、P1 对照 |
| 闭环 v2 结果 | `runs/oft_opportunity/closed_loop_v2.json` | 负结果证据 |
| W1 未来信息诊断 | `runs/oft_opportunity/w1_diagnostic.json` | 未来增量 +3.4~3.8pp（2/3 折） |
| 模型 ckpts | `ckpts/{oft_spatial,oft_object,oft_goal,oft_10,pi0fast_libero,pi05_libero,smolvla_libero,SmolVLM2-500M-Instruct}` | P3/P4/P5 全部就绪 |
| 采集脚本 | `scripts/collect_same_root.py` | P3 扩展基座 |
| 训练脚本 | `scripts/train_same_root_risk.py` | P0 审计的 offline 路径 |
| 特征管线 | `scripts/rase_common.py`（canonical 24 维 + bigram + ridge，纯 numpy） | P0/P1 复用 |
| 闭环部署 | `scripts/rase_selector_loop.py` + `scripts/selector_risk.py` | P0 审计的 deploy 路径；P7/P8 复用 |
| 环境 | conda `oft`（无 sklearn，纯 numpy 约定）、`smolvla`（LeRobot） | 按任务选用 |

### 0.2 已实测的数据事实（P0 预检）

- **root 一致性 PASS**：216 roots 内 `s_t_proprio` 最大差 = 0.0（216/216 完全一致）。
- `chunk_raw` 均为 8×7，root 内 3 个 candidate 的 chunk 互异（216/216）。
- `future_steps` 全部 = 64；`future_objects` 每 8 步记录，root 内物体位姿最大分叉中位数 **0.238**（Gate B 的 0.0795 为另一口径）。
- `consequence_label` ∈ [0.0138, 0.6684]，mean 0.315（proprio 位移代理，训练时 median 二值化）。
- suites：libero_spatial 540 行 / libero_object 108 行；**无 goal/10 任务**。
- 特征维度：B1 = 135 = 8 proprio + 24 canonical + 103 bigram；vocab 103 项已落盘 `same_root_risk.vocab.json`。
- 闭环 v2（`run_closed_loop_v2.sh` 确认）：`--selector same_root_risk.npz`、双候选 oft_spatial/oft_object、20 任务 × 6 trials。结果：**59/120 = 49.2%，switch=0，abstain=1455**，决策日志中双臂 mu≈0.9999/1.0000、sigma≈0.02 → LCB/UCB 差恒 < delta(0.05) → 永不切换。
- 环境：`oft` 环境 **无 sklearn**（`fit_ridge` 等纯 numpy 实现，P1 保持一致）。

### 0.3 科学状态一句话

反事实协议成立（Gate B/C PASS）、风险预测 OFT 架构内转移强（LOVO 0.94+）、**但风险驱动同状态选择在 LIBERO 结构性失败（闭环 v2）**、跨架构未确证、h_within=0（比较优势在任务级）。用户的 CRR 规划正是对"同状态相对信号"这一核心悬案的最省算力裁决实验。

---

## 1. 总体决策树（预算纪律）

```text
P0 audit（纯 CPU, ~1-2h）
   ↓ PASS（任一 FAIL 则修管线，禁止训练）
P1 relative CRR 三模型（纯 CPU, ~1h）←←← 核心决策关口
   ├─ PASS（pairwise acc≥0.65 & AUROC≥0.70 & 85%精度下 coverage≥5%）
   │    └─ P2 opportunity mining（纯 CPU）→ PASS(≥5%) → P3 补采 → P4/P5 → P6 → P7 → P8 → P9
   └─ FAIL
        └─ 不补采、不调 threshold → 三选一：
           (a) 换域（Eligibility Screen 重跑）
           (b) 任务级决策粒度（episode 级，对照查询表，须证非记忆）
           (c) probe/bandit 在线适配退路
```

**核心逻辑**：P1 的 within-root pairwise 精度就是"同状态双候选 chunk 特征是否携带相对信号"的直接检验（root 内 s_t/bigram 恒定 → 排序只由 chunk 决定）。这同时是对闭环 v2 失败机制的归因实验。它只需要现有 648 行，几乎零算力。

---

## 2. Phase P0：Feature/Provenance Audit（纯 CPU，~1-2h）

新脚本：`scripts/audit_same_root_provenance.py`，输出 JSON 报告 + `progress/2026-08-20_p0_audit_report.md`。

### 2.1 必查项（对照用户规划 §3.2）

| 项目 | 做法 | 状态 |
|---|---|---|
| Root 一致性 | 216 roots 内 s_t_proprio 逐维 max diff（应 = 0）；future 长度一致 | ✅ 已实测 PASS，脚本固化 |
| Candidate 冻结 | `chunk_stats` 8 个字段 == 从 `chunk_raw` 重算（存储即部署所见）；记录 queue cursor/seed 语义（代码审查 collect_same_root.py 无重生成路径） | 待脚本化 |
| **特征等价** | 同一存储行分别走 offline 路径（train_same_root_risk.py 内联构造：s_t_proprio+canonical+bigram）与 deploy 路径（rase_selector_loop.py `feats_of` 同构复刻）→ 要求 ‖x_offline−x_deploy‖∞ < 1e-6；再抽查 100 行 risk score 一致性 | **核心测试** |
| Vocab 一致性 | 从 648 行 task 重建 vocab == 落盘 `same_root_risk.vocab.json`（含顺序，103 项） | 待脚本化 |
| Normalizer 一致性 | npz 内 mean/scale == 全量 X 重拟合结果（fit_ridge 确定性）；确认训练/部署共用同一 npz（闭环 v2 已共用） | 待脚本化 |
| Label 语义 | `consequence_label` 公式（\|s_{t+H}−s_t\|?）与 median 二值化边界落文档 | 待脚本化 |
| 分布 gap 量化 | same-root `s_t` 状态分布 vs `dp_collect_*`（分支轨迹）状态分布：逐维 mean/std + Wasserstein 距离 → 给"闭环为何失败"一个数字 | 待脚本化（新内容） |
| API 一致性 | 注意 `rase_common.build_row_features` 读 `row["proprio"]`，same-root 行是 `s_t_proprio` → 记录该不一致，P1 复用 train_same_root_risk 的显式路径（而非 build_row_features），并加单测 | 待文档化 |

### 2.2 Gate P0

- **PASS**：上述全部一致（root 一致 + 特征数值一致 + normalizer 一致 + provenance 无污染）。
- **FAIL**：禁止训练新模型，先修 pipeline；若"分布 gap"是唯一问题（特征构造本身一致），则按 P1 归因实验继续——gap 是数据语义问题不是管线 bug，记录后放行 P1（P1 的结论会裁决它是否致命）。

---

## 3. Phase P1：Relative-Risk Baselines（纯 CPU，~0.5–1h，核心决策关口）

### 3.1 数据构造：`scripts/build_pairwise_dataset.py`

- 216 roots × C(3,2)=3 pairs = **648 pairs**。每 pair 字段：
  - root 常量：`s_t`(8)、bigram(103)
  - candidate：`a_i`、`a_j`（canonical 24 维，或原始 8×7=56 维——先 canonical）
  - 差分：`a_i−a_j`、`|a_i−a_j|`（24+24）
  - 标签：`y_ij = 1[q_i > q_j]`、margin `|Δq_ij|`、权重 `w_ij = min(1, |Δq|/m)`（m 取数据分位，如 0.10）
- **q 定义 v1（现有数据可用）**：
  - `Progress_i`：consequence_label（64 步 proprio 位移，已有，可信）
  - `Drift_i`：物体位姿漂移（`future_objects` 首末差，root 内中位 0.238）
  - `Recoverability_i`：**现有数据没有**（需 reference 从 s_{t+H} 回滚）→ v1 置 0，v2 等 P3 补采后加入（collect_same_root 有 `--label-mode reference` 先例）
  - `q_i = w1·Progress − w2·Drift + w3·Recoverability`（v1: w1=1, w2=0.5, w3=0；标准化后）

### 3.2 三个模型（纯 numpy；ridge 用 `rase_common.fit_ridge`，MLP 2×64 手写 autograd 或 numpy 梯度）

- **A 共享 scorer**：`r_i = f_θ(s_t, a_i)`，P(i≻j)=σ(r_i−r_j)，pairwise logistic loss。注意：f 线性时 ≡ B 的差分形式，所以 A 的线性版与 B 互为验证，A 的 MLP 版才是独立结构。
- **B 显式 Δ 模型**：`x_ij = [s_t, bigram, a_i, a_j, a_i−a_j, |a_i−a_j|] → P(i≻j)`，ridge logistic。
- **C 反对称模型**：`g_ij = h(s,a_i,a_j) − h(s,a_j,a_i)`（MLP），结构保证反对称。
- **关键性质（写进报告）**：root 内 s_t 与 bigram 恒定 → **within-root 排序只由 chunk 特征决定**。因此 pairwise 精度直接回答："同状态双候选的早期 chunk 是否含相对后果信号？"——这正是闭环 v2 失败的核心归因。

### 3.3 划分与对照（防止"任务记忆"冒充风险驱动——项目纪律）

- **主划分：LOVO-by-VLA**（训练 spatial+object 候选，测试 oft_goal 候选行）→ 对齐 zero-shot 主张。
- 辅助划分：suite-split（训 spatial roots，测 object roots）。
- **对照基线（必须同时报）**：
  1. 随机（0.5）
  2. **task-prior / suite-favorite**（spatial 任务恒选 oft_spatial，object 任务恒选 oft_object）——"非记忆"对照的上界参照，也是用户规划的 task-router 诊断位
  3. 现成 B1 `same_root_risk.npz` 绝对分比较（预期 ≈0.5，复现闭环失败，作为负对照）
  4. chunk-only 变体（去掉 s_t/bigram，纯 chunk 差分）——检验信号来源
- 若 LOVO pairwise 精度 ≤ task-prior 且 ≤ B1 绝对分 → 结论：同状态 chunk 无相对信号（结构性 FAIL，按 §1 决策树走）。

### 3.4 指标与四张图（用户规划 §7/§36）

1. pairwise ROC（within-root 配对，按 root 聚合再平均）
2. pairwise margin calibration（预测概率 vs 经验频率，marginal 区间）
3. switch precision–coverage 曲线（margin m_ij=|P(i≻j)−0.5|，阈值 τ；只允许 m>τ 才 switch）
4. estimated selector gain–coverage 曲线（best-fixed / always-fallback / task-router-upper / oracle 参照）

外加：pairwise accuracy、pairwise AUROC/AUPRC、margin correlation（spearman）、ECE。

### 3.5 Gate P1（用户规划 §8）

- **PASS-A 可学习**：held-out（LOVO-by-VLA）pairwise accuracy ≥ 0.65 且 AUROC ≥ 0.70。
- **PASS-B 部署潜力**：存在 τ 使 SwitchPrecision ≥ 0.85 且 Coverage ≥ 0.05。
- **Strong PASS**：Precision ≥ 0.90 且 Coverage ≥ 0.10。
- 同时要求 **显著高于 task-prior 与 B1 绝对分对照**，否则按 FAIL 处理（记忆/水平信号不算 CRR）。
- 产出：`runs/oft_opportunity/crr_p1_*.json` + 四图 + `progress/2026-08-20_crr_p1_report.md`。

### 3.6 FAIL 预案（不动 threshold，不进 P3）

按用户规划 §11/§33：写结论"当前 OFT/LIBERO 机会是 task-level routing，不适合 RASE runtime arbitration 主 benchmark"，然后三选一：
(a) 换域（libero_90 中间难度子集或 π0-fast 微调 libero_90 的 30–70% 区间——IDEA §7.3 重开条件②）；
(b) 任务级决策粒度实验（episode 级选择 + 未见任务 LOO，与查询表 99.2% 对照，须证非记忆）；
(c) probe/bandit 退路（`scripts/bandit_adapter.py` 已就绪）。

---

## 4. Phase P2：Within-Task Opportunity Mining（纯 CPU，~0.5h）

### 4.1 重要事实

**Gate B 已测得 h_within = 0**（任务内最佳候选不随 state 翻转；V3 报告 §Gate B）。用户规划的 H_within≥5% 在现有数据上大概率 FAIL——这正是要诚实复测并落文档的。

### 4.2 新脚本：`scripts/measure_within_task_heterogeneity.py`

- 用 §3.1 的更细 q（含 object-pose progress）重算：每 (task, suite) 内跨 roots 的 argmax_i q_i 翻转率 H_within；state-level oracle gain vs task-level oracle gain；按 root 数做 bootstrap 下界。
- 数据限制如实报告：648 行只覆盖 spatial（10 任务，180 roots）+ object（36 roots），无 goal/10。

### 4.3 Gate P2（用户规划 §11）

- **PASS**：H_within ≥ 5% 且 OracleGain_within ≥ 5pp → 进 P3。
- **FAIL**：H_within < 5% → "当前 domain 是 task-level routing，非 runtime arbitration 主 benchmark"——**换域，不堆模型**（与 P1 FAIL 汇合到同一决策树）。

---

## 5. Phase P3：一次性补采 RGB + OFT + π0 same-root（仅当 P1 或 P2 积极；GPU ~3–6h）

### 5.1 扩展 `scripts/collect_same_root.py`（一次采集三用）

1. **存 RGB**：决策点 `full_image` + `wrist_image`（256×256，原始像素落盘；编码器到 P4 再选，保证灵活性）。数据量小（216 roots × 2 图 ≈ 几百 MB）。
2. **加 π0-fast candidate**：走 LeRobot 路径——`rase/collect/lerobot_libero_plus_adapter.py` 与 `diag_pi0fast.py` 已有先例；ckpt `pi0fast_libero` 在位。候选集 = {oft_spatial, oft_object, oft_goal, π0-fast}（π0-fast 是异构架构，正是 P5 需要的）。
3. **补 Recoverability 标签**：从 s_{t+H} 起用 reference 模型（oft_spatial）回滚 40 步记录 recovery_success（`collect_same_root.py` 已有 `--label-mode reference` 先例）→ 补齐 q 的第三项，P1 v2 重跑。
4. **匹配执行纪律**：同一 snapshot restore + matched seed；candidate chunk 冻结后执行；queue cursor 审计（闭环 v2 教训）。
5. 评估 roots 预冻结、均匀采样，不按 disagreement/outcome 筛选（用户规划 §10.3）。

### 5.2 机器约束（RTX 5090 32G）

- OFT 7B bf16 ≈ 15.5G×2 resident 刚好；**π0-fast 需与 OFT 分时交换或单 OFT resident**。建议两趟：趟 1 全 OFT（含 RGB+recovery），趟 2 π0-fast（复用 snapshot 恢复流程；需确认 π0 在 LIBERO 的采样协议与 action 空间归一化，P10/P12 工程问题先例在 `rase/collect/`）。
- 规模建议：沿用 216 root 结构（spatial 10 任务 + object），每 root 4 候选 × 64 步。若 P1/P2 有积极信号再按需扩 goal/10 任务。

### 5.3 一次数据回答三问（用户规划 §13）

Q1 视觉×action 交互能否区分同状态候选（P4）；Q2 within-task 比较优势是否真实存在（P2 复核，用更完整数据）；Q3 OFT 上学到的 risk 是否 zero-shot 到 π0-fast（P5）。

---

## 6. Phase P4：视觉 × Action-Conditioned CRR（P3 后；Tier 1，低 GPU）

- **冻结编码器**：优先用现成 `SmolVLM2-500M-Instruct`（ckpt 在位）或 OFT 自带 SigLIP/DINO 塔；RGB 每 root 只编码一次 `h_v=E_v(I_t)`。
- **低秩乘性交互**：`z_v=(P_v h_v)⊙(P_a h_a)`，`z_l=(P_l h_l)⊙(Q_a h_a)`，输入 `[h_a, z_v, z_l, proprio]` → MLP 2×64/128，dropout 0–0.1，可训练参数 < 3M（用户规划 §15）。
- **Gate P4**：同 precision≥85% 下，Δpairwise accuracy ≥ 3pp 或 Δcoverage ≥ 5pp，且同时超过 B1 直接风险 / action-only CRR / task-only 文本基线（用户规划 §17）。失败只允许一次 single-layer action-query cross-attention 升级；再失败停止视觉加深。
- 诚实预期：bigram 已编码任务文本，视觉增量可能有限——gate 说了算。

---

## 7. Phase P5：OFT→π0 same-root 零样本（用 P3 数据；Tier 1）

- 训练：OFT 三候选（spatial/object/goal）CRR；测试：π0-fast 候选行。禁止：π0 outcome 进训练、π0-specific scaler/calibration、policy identity、task lookup（用户规划 §18）。
- 指标：within-root pairwise AUROC ≥ 0.65（Minimum PASS，且高于 action-only/随机）；≥0.70 且 85% 精度下 coverage≥5%（Strong）。
- 相对排序是主指标（同 root 内 q_better vs q_worse），**不看绝对成功率**（π0-fast 87% 高分域，绝对 AUROC 可能来自成功率水平——用户规划 §19）。
- FAIL 处理：写死 claim"representation 跨 checkpoint 转移、不跨架构"，不再硬调（用户规划 §20）。

---

## 8. Phase P6–P8：离线模拟 → smoke → 正式闭环

### P6 离线 selector 模拟（纯 CPU，~1h；`scripts/selector_simulation.py`）

- 用 same-root GT + CRR 预测 + abstain 规则（margin τ）回放：EstimatedGain(τ) vs best-fixed / always-fallback / random-switch / task-router-upper / oracle；按 task 或 root 做 paired bootstrap CI。
- **Gate P6**：CI_lower > 0（最好 ≥3pp）才进真实闭环；CI_lower ≤ 0 禁止大规模 rollout（用户规划 §22）。

### P7 Closed-Loop Smoke（Tier 2，GPU ~1h）

- 4–6 任务 × 3 seeds × 2–3 trials；只回答 5 问（用户规划 §23）：真切换？弱势域不再 0 success？abstain 不接近 100%？switch precision 与离线一致？无 oscillation？
- Kill 条件（用户规划 §24）：switch=0 / coverage<3% / precision<70% / switch 后 success 下降 / oscillation → 暂停正式闭环。只允许 1 个预注册 threshold + 1 个敏感性对照。
- 复用 `rase_selector_loop.py` 框架，新增 CRR 模型接口（pairwise 分 → 每候选聚合分，或直接 pairwise 决策）。

### P8 正式 Closed-Loop（Tier 2，GPU ~2–4h）

- 主 endpoint：`Gain = Success(RASE) − Success(best-fixed)`，≥3pp 且 paired bootstrap CI_lower > 0。
- 记录：total/per-task success、switch count、switch precision、abstain/fallback rate、query cost、latency、failure-after-switch、recovery-after-switch（用户规划 §25）。
- 不确定性：5 bootstrap heads + LCB 保守仲裁（用户规划 §26）；Runtime 优化（adaptive candidate invocation）放到方法成功之后（§27）。

---

## 9. 算力预算与时间线

| Tier | 阶段 | 算力 | 预计耗时 |
|---|---|---|---|
| Tier 0 | P0 audit、P1 CRR 三模型、P2 机会挖掘、P6 模拟 | 纯 CPU | 第 1 天（~2–4h） |
| Tier 2 | P3 补采（仅 Gate 通过后） | 5090 GPU | 第 2–3 天（3–6h） |
| Tier 1 | P4 视觉 CRR、P5 跨架构 | 低 GPU（冻结编码 + 小 MLP） | 第 4–5 天 |
| Tier 0 | P6 模拟复核 | CPU | 第 5 天 |
| Tier 2 | P7 smoke、P8 正式闭环 | 5090 GPU | 第 6–7 天（每次 1–4h） |
| Tier 2 | P9 第二平台 | 另行评估 | 论文后期 |
| 禁止 | 视频 WM / 大多模态 verifier / full RL / 大规模 sweep | — | 当前域永久搁置（用户规划 §28/§32） |

---

## 10. 止损决策表（用户规划 §33 落地）

| 阶段 | PASS | FAIL → 动作 |
|---|---|---|
| P0 审计 | 特征完全一致 | 修 pipeline，不训练 |
| P1 relative CRR | pairwise≥0.65 & AUROC≥0.70 & 85%精度下 coverage≥5%，且超 task-prior/绝对分对照 | 不补采不调参 → 换域 / 任务级粒度 / bandit 退路 |
| P2 机会挖掘 | H_within≥5%（预期 FAIL，Gate B 已测=0） | 判定"task-level routing 域"，换域或收缩 claim |
| P4 视觉交互 | Δpairwise≥3pp 或 Δcoverage≥5pp | 只允许一次 cross-attention；再败停止 |
| P5 跨架构 | pairwise AUROC≥0.65 | 写死"架构内转移"claim |
| P6 离线增益 | bootstrap CI_lower>0（最好≥3pp） | 禁止正式闭环 |
| P7 smoke | 有 switch、有 recovery、无 oscillation | 回模型层 |
| P8 闭环 | Gain≥3pp 且 CI_lower>0 | 不上第二平台 |

---

## 11. 纪律与工程卫生

1. **先提交当前未提交改动**（git status 有大量 M，含 rase/collect、scripts）；每个新 run 目录快照 git SHA + `env.lock.md` sha256。
2. 所有新脚本进 `scripts/` 并随 git 提交；报告写 `progress/2026-08-20_*.md`（项目惯例）。
3. 任何"提升"结论必须注明驱动机制（风险驱动 vs 记忆 vs 水平）；特征处理必须拟合/预测一致（防 P13 回归）。
4. 纯 numpy 约定（oft 环境无 sklearn；如需可装，但先保持 rase_common 风格）。
5. 诚实纪律：闭环 v2 的负结果、h_within=0、VLA-ID 0.894 都保留为论文对照/限制，不隐藏。

---

## 12. 立即执行清单（第一批）

```bash
# 0) 提交当前改动快照
cd /root/autodl-tmp/RASE && git add -A && git commit -m "snapshot before CRR phase"

# 1) P0 审计
python scripts/audit_same_root_provenance.py \
  --data runs/oft_opportunity/same_root_w1.jsonl \
  --risk runs/oft_opportunity/same_root_risk.npz \
  --vocab runs/oft_opportunity/same_root_risk.vocab.json \
  --output runs/oft_opportunity/p0_audit.json

# 2) P1 数据构造 + 三模型 + 对照
python scripts/build_pairwise_dataset.py \
  --data runs/oft_opportunity/same_root_w1.jsonl \
  --output runs/oft_opportunity/crr_pairs.jsonl
python scripts/train_crr_baselines.py \
  --pairs runs/oft_opportunity/crr_pairs.jsonl \
  --output runs/oft_opportunity/crr_p1_results.json

# 3) P2 机会挖掘
python scripts/measure_within_task_heterogeneity.py \
  --data runs/oft_opportunity/same_root_w1.jsonl \
  --output runs/oft_opportunity/crr_p2_heterogeneity.json
```

第一批产出：p0 审计报告、P1 四图（pairwise ROC / margin calibration / precision-coverage / gain-coverage）、P2 H_within 与 oracle gain 的 bootstrap 区间、一份 `progress/2026-08-20_crr_p1_report.md`。然后按 §1 决策树决定是否进入 P3。
