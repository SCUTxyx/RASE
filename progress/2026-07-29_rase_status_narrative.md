# RASE 项目进度叙事与路线图（截至 2026-07-29）

> **Superseded current-status note（2026-07-31）：** 本文是截至 7/29 的历史快照，
> 以下数字、当时判断和 next steps 不作回写或篡改。当前状态以
> [`2026-07-31_w9c_clean_task_identity_fix.md`](2026-07-31_w9c_clean_task_identity_fix.md)、
> [`2026-07-31_w9c_selector_gate_result.md`](2026-07-31_w9c_selector_gate_result.md) 和
> [`2026-07-31_idea_evolution_and_next_questions.md`](2026-07-31_idea_evolution_and_next_questions.md)
> 为准：W9C identity fix 后 probe/coverage/readiness 已 PASS；ridge held-out gate 已按
> 预注册标准 KILL；不升级 MLP/RL，论文转 benchmark/diagnosis。特别地，task-held-out
> test 仅 8 clean states、无 failure state，不能被解读为 selector 泛化证据。

> 本文汇总从 baseline 到 W9 clean-control gate 的主线证据、主张如何改写、为什么改写，以及下一步 gate。  
> 细节实验记录仍以同目录下各 `YYYY-MM-DD_*.md` 为准；本文是叙事总览，不替代逐次实验不可变记录。

---

## 0. 一句话现状

RASE 当前最稳固的发现**不是**“SmolVLA 候选能恢复失败”，而是：

> **恢复性是 policy-relative 的：在视觉扰动诱导的 failure frontier 上，弱 policy（SmolVLA）续完接近零，强 policy（OpenVLA-OFT）在同一批 snapshot 上可救回一部分；且主要增益来自直接升级到强 policy，而不是候选动作前缀本身。**

项目已据此从 **candidate-fallback / reranker** 叙事，转向：

1. **episode-disjoint、policy-relative recoverability benchmark**（主贡献）；
2. **成本敏感的轻量 escalation selector**（`CONTINUE_SMOL / ESCALATE_OFT / ABSTAIN`，验证 benchmark 可用性的方法贡献）。

**W9-B 已按预注册 kill 规则停机：** clean-control 采满 140 ep 后成功率仅 **~7.2%**（相对 clean baseline **70%**），且成功几乎全在 Spatial；Object/Goal/Long 的 early/mid cell 全空，无法冻 clean32。下一动作是 **审计 clean 任务/环境映射并重采**，不是降低 coverage、也不是训 selector。

---

## 1. 时间线与阶段结论

### 1.1 基础设施与基线（约 7/16–7/17）

| 工作 | 结果 | 意义 |
|---|---:|---|
| SmolVLA clean LIBERO（seed 0, nas10） | **70.0%** / 2,000 ep | base policy 在干净分布有效 |
| ForkableEnv W1 gate | 4/4 通过 | deterministic snapshot/restore 可用 |
| LIBERO-Plus collapse smoke | ~2.5%（小样本） | 仅定性，不作主表 |

**当时的 idea：** 有可靠 forkable 环境后，可以在失败时刻做候选采样与 fallback，形成“学会何时换招”的 RASE。

### 1.2 扰动崩塌与状态池（约 7/17–7/18）

| 工作 | 结果 | 意义 |
|---|---:|---|
| Plus collapse full | **0.38%** / 3,142 completed（camera≈0%，robot≈0.78%） | 视觉扰动形成强烈 failure frontier |
| NGC Step-1 scale200 | 199 ep / 3666 states | 大规模 failure pool；几乎全是失败轨迹 |

**当时的 idea：** 在失败态上采 K 个候选，用统计协议（Wilson 等）筛选“可恢复候选”，再训练 selector。

**隐患已埋下：** pool 极度 failure-biased，尚无足够 success episode 支撑 clean-regret；若直接训“何时升级”，极易塌成 always-escalate。

### 1.3 候选与续完诊断（W2–W5，约 7/18–7/27）

| 阶段 | 结果 | 对主张的影响 |
|---|---|---|
| W2 candidates pilot | 多样性通过；未测续完 | 工程可行 |
| W3 Smol continuation | 多 cohort **近零**（如 0/768） | Smol→Smol 续完能力不足 |
| W3 OFT 对照 | Spatial 等有阳性；同态上 Smol 四温仍零 | **缺口在 continuation policy，不是温度/假阴性** |
| W4 ADEQUATE 32 态 | Smol **0/1536**；OFT portfolio **17/32**（候选命中 65/256） | 强证据：同一批候选，换 OFT 续完就有恢复 |
| W5 温度扫描 | t=0.3/0.7/1.0 合计 **0/576** | **关闭 proposal-temperature 诊断线** |

**idea 第一次重要转向：**

- 从“调温度 / 多采样就能救” → “**recoverability 依赖 continuation policy**”；
- 从“Smol 候选质量问题” → “至少在当前 frontier，**Smol proposal+continuation 缺乏正例**”。

原因：同一 snapshot、同一候选前缀，Smol 续完为零而 OFT 可正，温度扫描无法翻转。

### 1.4 配对 pilot 与机制否证（W6–W7 discovery，约 7/27–7/29）

| 阶段 | 结果 | 意义 |
|---|---|---|
| W6 L1–L2 采集 | 40/40 failure ep，762 states | 分层 failure-challenge；success control 仍缺 |
| W6 paired matrix | Smol **0/8** vs OFT **2/8**；McNemar **p=0.5** | 方向对，但 pilot 无显著性功效 |
| W7 discovery prefix ablation | **candidate-specific rescue = 0/2** | **机制否证：不能把 OFT 命中归因于候选内容** |

W6 两个 OFT-only 态的控制实验：

1. **continuation_sufficient**：direct OFT、zero prefix、8 个 candidate 全成功 → OFT 续跑本身充分；
2. **passive_prefix_sufficient**：direct 失败，但 zero / 部分 candidate 成功 → 被动等待/动力学已是充分替代解释。

**idea 第二次重要转向（核心否证）：**

- 停止主张 “learned Smol proposal causes rescue”；
- 候选 hit 降级为**描述统计**，inferential unit 固定为 **state / episode-group**；
- 研究问题改为 **state-conditioned continuation / escalation**，并要求 passive prefix、clean regret、strong-policy cost 控制。

原因：没有 time-matched zero / direct 对照时，会把“强 policy 续完能力”误写成“候选生成能力”。

### 1.5 Held-out 主结果与可部署臂（W7–W8，7/29）

**W7 held-out24（与 W6 episode-group 零重叠，四 cell 各 6 态）：**

| 臂 | 结果 |
|---|---:|
| Smol portfolio | **0/24** |
| Prefix + OFT portfolio（any-of-8） | **8/24 (33%)**，Wilson ≈ [0.18, 0.53] |
| Exact McNemar（state） | **p = 0.0078125** |

Suite 粗分：Long 4/10、Goal 3/8、Spatial 1/4、Object 0/2。  
`t0`：min 0 / median 10 / max 36 → **early-stage cohort**，不可外推 mid/late。

**W7 held-out 上 8 个 OFT-only 态的 causal ablation：**

| 机制 | n |
|---|---:|
| continuation_sufficient_candidate_invariant | 4 |
| continuation_sufficient_candidate_harm_possible | 3 |
| passive_prefix_sufficient | 1 |
| **candidate_specific_rescue** | **0** |

→ 再次确认：多数是 OFT continuation 够用；部分候选甚至有害；**没有“只有候选能救”的态。**

**W8 direct OFT（同一 24 snapshot，可部署升级臂）：**

| 指标 | 结果 |
|---|---:|
| Direct OFT | **9/24 (37.5%)** |
| vs Smol McNemar | **p = 0.00390625** |
| Goal / Long / Object / Spatial | 4/8、5/10、0/2、0/4 |

**W7 portfolio vs W8 direct 逐状态重叠（posthoc）：**

| both | prefix-only | direct-only | neither |
|---:|---:|---:|---:|
| 7 | 1 | 2 | 14 |

Prefix vs direct 的 McNemar **p = 1.0**（边际 8/24 vs 9/24 **不可**解读为谁更好）。  
Spatial 上唯一的 prefix-only 与 ablation 的 `passive_prefix_sufficient` 一致。

**idea 第三次收敛：**

- 主机制收敛为 **direct policy escalation**；
- any-of-K portfolio 保留为 **诊断/上界**，**禁止**当作 selector 标签（proxy）；
- 方法冻结为三动作路由器：`CONTINUE_SMOL / ESCALATE_OFT / ABSTAIN`。

### 1.6 Selector 代码与 W9 clean-control gate（7/29 晚）

已实现并同步：

- 三臂 schema、proxy/leakage/label-collapse readiness gate；
- dependency-free ridge utility baseline、episode/task split；
- W7/W8 pairing、pool-support inventory、failure-only readiness 审计；
- fingerprint `rase-task-identity/v2`（排除 mujoco 可变 XML 假阳性）。

**Failure-only readiness（预期拒绝）：** `ready: false`（无 clean-success control）。

**W9 clean collect（预注册 60+40+40，硬停 140；详见
[coverage gate 记录](2026-07-29_w9_clean_control_coverage_gate.md)）：**

| 项 | 结果 |
|---|---:|
| Pool episodes | 138（10 success / 128 failure） |
| Episode success rate | **7.2%**（vs clean baseline 70%） |
| Success by suite | Spatial 9 / Long 1 / Object 0 / Goal 0 |
| Coverage freeze | **失败**：6/8 cells `n=0`（Object/Goal/Long × early+mid） |
| Direct Smol failure24 | **0/24** |
| clean32 标注 / selector | **未启动**（exit 2） |

**idea 纪律确认：** coverage 失败是合法科学停机，不是“再多采一点就行”。同配置下相对 70% baseline 的数量级塌缩，优先怀疑 **clean 任务映射 / 环境是否真 clean**，禁止放宽 `per_cell` 或 Spatial-only 训 selector。

---

## 2. 创新点 / Idea 如何演变（对照表）

| 阶段 | 当时主打 novelty / claim | 被什么证据打穿或改写 | 现在保留什么 |
|---|---|---|---|
| 早期 | 统一 fallback 空间（多招切换） | HELM / VoLo / B2FF / AEGIS 已压缩该叙事空间；且本地缺乏多臂正例 | 不把“统一空间”本身当 novelty |
| W2–W3 | 候选采样 + 统计筛选可恢复动作 | Smol 续完多 cohort 近零 | Forkable + 候选协议作为 **基础设施** |
| W3–W5 | 调 proposal temperature / 多样性即可 | 0/576 温度扫描 | 关闭该线；强调 **policy pair** |
| W4–W6 | “Smol 提案 + 强续完”即 RASE 方法贡献 | 归因实验显示候选非因果 | 改为 **cross-oracle asymmetry / policy-relative R** |
| W6 pilot | 小样本宣称显著优越 | McNemar p=0.5 | 只作方向性 pilot |
| W7 | held-out 显著 gap + 机制澄清 | candidate-specific rescue 0 | **benchmark + 配对统计** 成主贡献 |
| W8 | 可部署 direct escalation | portfolio ≠ deployable action | 三臂 cost-sensitive router；portfolio 仅诊断 |
| W9-B collect | 预注册 140 ep 凑齐 clean32 | **7.2%** success；6/8 cell 空；exit 2 | 先审计 clean 映射再重采；禁降 gate |
| 下一阶段 | 复杂 RL selector | failure-only / 残缺 clean 会塌成 always-OFT | 有效 clean32 + ridge；不过 matched-random 不上 RL |

**贯穿始终、仍成立的贡献轴：**

1. **可复现实验基础设施**：deterministic snapshot/restore、provenance checksum、resumable suite runners、episode-disjoint split audit。  
2. **诊断性事实**：扰动崩塌 + 弱/强 policy 恢复不对称。  
3. **协议纪律**：state 为推断单位；禁止同 episode snapshot 伪独立；禁止 proxy 标签训练。

---

## 3. 当前可以 / 不可以说什么

### 可以说

- 构建了带 episode-disjoint split、配对评测与 provenance 的恢复性实验栈。  
- 在多个 failure-conditioned cohort 上，Smol proposal/continuation **一致缺乏正例**。  
- W7 held-out 上 OFT（prefix portfolio）相对 Smol 的 state 级差异显著（p≈0.008）；W8 direct 亦显著优于 Smol（p≈0.004）。  
- Prefix 控制实验表明：主要增益不应归因于候选特异性；direct escalation 已足以解释主增益量级。  
- Selector readiness 正确拒绝 failure-only 训练，避免隐瞒 clean regret。

### 暂时不能说

- 无条件任务成功率或“所有 suite 均可恢复”。  
- “Smol candidate 因果恢复状态”（当前归因 0）。  
- Failure-conditioned rate = 干净分布成功率。  
- “Selector 已有效 / 已学会何时升级”（W9-B coverage 失败；未过 readiness）。  
- “当前 Plus-adapter clean collect ≈ 70% clean baseline”（实测 ~7.2%，需先修映射）。  
- 用边际 9/24 vs 8/24 宣称 direct 优于 prefix（配对 p=1.0）。  
- mid/late recovery、L3–L5、第二 backbone 的外推。

---

## 4. 证据速查表（论文主表候选）

| 证据 | 数字 | 进入论文时的角色 |
|---|---|---|
| Clean SmolVLA LIBERO | 70.0% / 2000 | Base competence |
| Plus camera/robot collapse | 0.38% / 3142 | Failure frontier |
| W4 ADEQUATE | Smol 0/1536；OFT port. 17/32 | Early cross-oracle asymmetry |
| W5 temperature | 0/576 | Kill temperature confound |
| W6 L1–L2 pilot | 0/8 vs 2/8；p=0.5 | Directional only |
| W7 discovery ablation | rescue 0/2 | Mechanism falsification |
| W7 held-out | 0/24 vs 8/24；p=0.0078 | **主配对结果** |
| W7 held-out ablation | rescue 0/8 | 机制再确认 |
| W8 direct | 9/24；vs Smol p=0.0039 | Deployable escalation arm |
| W7↔W8 overlap | 7/1/2/14；prefix vs direct p=1.0 | 禁止误比边际率 |
| Selector readiness | NOT_READY | 负结果也要报 |
| W9 clean collect 140 | 10/138≈7.2%；coverage fail | **W9-B kill；禁训 selector** |

---

## 5. 接下来的计划（按 gate，勿跳步）

### Gate W9-A：配对与文档冻结（基本完成）

- [x] W7/W8 逐状态重叠表（`runs/ngc_w8_direct_escalation_pairing.*`）  
- [x] Pool success-support inventory（结论：需新采）  
- [x] Failure-only readiness → `NOT_READY`  
- [ ] 把 pairing / ablation / 主张边界写入论文草稿主表脚注

### Gate W9-B：Clean-success control（**采集已跑完；coverage 未过，仍阻塞**）

已执行的预注册采集见
[2026-07-29_w9_clean_control_coverage_gate.md](2026-07-29_w9_clean_control_coverage_gate.md)：
exit 2，**不得**用同一脚本盲目重跑。

当前最高优先级（诊断，不是扩采样）：

1. **审计** Plus-adapter clean 路径：task catalog、是否真 `level=0` / 无扰动、观测栈与 70% baseline 是否同一 checkpoint/horizon。  
2. 用小样本对照复现 baseline 量级 success；确认后再设计 **success-retained** 重采。  
3. 对通过 coverage 的 control 态跑与 challenge **相同**两臂：direct Smol + direct OFT。  
4. 合并 challenge + control → episode/task-heldout → readiness；不过则 **禁止训练**。

目标仍是：至少约 **30+** 个独立 **success episode-group**（跨 pool 去重；**禁止**同一成功轨迹多 snapshot 伪独立）。

### Gate W9-C：最小 selector（仅 readiness 通过后）

1. 训练 **ridge / linear** cost-sensitive utility（已有代码路径）。  
2. 对照：always-Smol、always-OFT、matched-random trigger、abstain、oracle。  
3. 主指标：**task/net success、FEB、clean regret、strong-policy usage、latency**——禁止只报恢复率。  
4. **Kill：** task-held-out 上不优于 matched-random → 先修数据/特征，**不上 MLP/RL**。

### Gate W9-D：仅 C 通过后的扩展

- 第二 backbone / 鲁棒化模型，验证 policy-relative 是否跨模型成立；  
- HELM-like rollback、B2FF-like milestone 等额外臂（信息增益驱动，非先堆满五招）；  
- mid/late `t0` 独立 cohort；Object/Spatial 加压（当前 0/6 欠功效，勿写成不可能）；  
- 最后才考虑 offline CQL / online DQN。

### 投稿定位（并行决策）

| 路径 | 条件 |
|---|---|
| **Benchmark / diagnosis 主投** | 若 clean control 或 selector gate 失败：保住配对协议、机制否证、cross-oracle matrix |
| **Benchmark + method** | selector 在成本—成功率 Pareto 上稳定优于规则基线，且无泄漏 split |
| **避免** | 用复杂 RL 叙事稀释已有最强诊断证据 |

---

## 6. 风险与纪律（提醒自己）

1. **Failure-conditioned ≠ unconditional。** 所有恢复率必须写清条件化。  
2. **Portfolio ≠ action。** any-of-K 不能当部署标签。  
3. **边际率 ≠ 配对结论。** 必须看 overlap 四格表。  
4. **Early t0 ≠ 全时段。** mid/late 另开 cohort。  
5. **NOT_READY 是保护，不是故障。** 缺 clean control 时强行训练会污染论文可信度。  
6. **观察 held-out 后不改 K/T/horizon/state keys。** 新实验新记录，不覆写历史。

---

## 7. 关键产物索引

| 类型 | 路径 |
|---|---|
| 导师简报 | `progress/2026-07-29_advisor_brief.md` |
| W7 记录 | `progress/2026-07-29_w7_causal_heldout.md` |
| W8 结果 | `progress/2026-07-29_w8_direct_escalation_results.md` |
| 执行方案 v4 | `plan/RASE_top_conference_execution_v4.md` |
| W7 矩阵 | `runs/ngc_w7_heldout24_policy_matrix.{json,md}` |
| W7 ablation | `runs/ngc_w7_heldout24_prefix_ablation.{json,md}` |
| W8 escalation 数据 | `runs/ngc_w8_direct_escalation_failure.jsonl` |
| W7↔W8 pairing | `runs/ngc_w8_direct_escalation_pairing.{json,md}` |
| Readiness / pool | `runs/ngc_w8_failure_selector_audit/readiness_audit.json`，`runs/ngc_w8_selector_pool_support.json` |

---

## 8. 结语

到今天为止，RASE 最大的“进展”不只是数字从 pilot 走到 held-out 显著，而是 **主动完成了一次机制收缩**：用控制实验否证了最初最诱人的 candidate-rescue 故事，把可发表主张收束到更硬、也更可辩护的 **policy-relative recoverability + direct escalation benchmark** 上。

下一步的胜负手很清晰：**先查清为何 clean collect ≪ 70% baseline 并重采有效 control，再谈 selector；coverage/readiness 不过关就诚实投 diagnosis/benchmark，而不是降 gate 或用更大模型掩盖标签缺陷。**
