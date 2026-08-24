# RASE 动作纠正方法调研与改进方案（2026-08-20）

> 目的：调研类似 paper 如何做动作纠正/候选仲裁，学习并优化，增强创新点（可行度高），
> 解决"fallback 支配"问题；同时说明 RASE 的 candidates 生成机制与默认动作的关系。
> 状态：文献调研完成（4 路 subagent + 直接检索）；综合与方案见下。

---

## 0. 一句话结论

> 别家的"动作纠正"已覆盖"采样-验证-重排"（RoboMonkey/EVE/V-GPS/RoVer/Pre-VLA/CheckVLA），
> RASE 不能在此重复；真正未被覆盖的是 **same-root 反事实训练 + 跨策略零样本 + 保守仲裁**
> 的组合。fallback 支配问题的解法不是"换更强的纠正策略"，而是把纠正从
> **"切换策略"改为"在风险模型引导下重新生成"**，并让支配结构本身成为训练信号。

---

## 1. 现状回顾（三行）

1. LIBERO 域：跨策略 fallback 弱支配 continue（96 状态，0 continue-only，oracle gain=0pp）→ 选择器无增益；
2. 同策略 requery/resample（用户 idea 主机制）**从未被真正测过**（确定性推理 + 采样路径已存在但未用）；
3. Novelty 边界（roadmap §3）：**跨策略反事实仲裁**，P(s_{t+1:t+H}|s_t,a,π)≈P(s_{t+1:t+H}|s_t,a) 的 policy-invariance 是核心假设。

---

## 2. 文献调研综述

<!-- 待 4 路 subagent 结果填充：方法分类表、代表 paper、机制摘要、与 RASE 的映射 -->

### 2.1 生成-验证范式（single-policy verification，拥挤区）

**代表方法**（详见 `rase_verifier_survey.md`，18 方法全表）：

| 方法 | verifier 类型 | 参与方式 | 关键数字 |
|---|---|---|---|
| RoboMonkey (2506.17811) | VLM 动作 verifier | 采样 K 候选 + 扰动/多数投票 → rerank | ID +9pp / OOD +25pp；test-time scaling law |
| RoVer (2510.10975) | 机器人 PRM | 打分 + 沿 PRM 方向扩展候选 | 同预算更多候选 |
| CoVer-VLA (2602.12281) | 对比式 verifier | 指令重述 × 动作采样二维扩展 | ID +22% / OOD +13% |
| Action-Draft-and-Verify (2603.18091) | VLM perplexity | 扩散草稿 K 块 + 单次前向打分 | sim +4.3 / real +19.7 |
| TapSampling (2605.25547) | 任务进度预测器 | Action-VAE 潜在空间采样 + rerank | +1.8~+4.3pp |
| CheckVLA (2607.26789) | 动作条件世界模型 | 执行期验证 → 保形阈值 → 改写后缀 | timely recall 48.6→77.9%；matched budget +8.5pp |
| BOKBO (2605.30660) | 违规预测器（语义特征） | K 候选保形 abstention（LIBERO+OpenVLA） | margin~σ 相关 0.98、与真实违规噪声底 |
| UPS (2602.22474) | 语义不确定性 + 动作可行性 | 三路：执行/澄清/干预 + 保形校准 + residual learning | 对应 continue/regenerate/switch |
| V-GPS (2410.13816) | offline-RL value | 跨架构 rerank | 多平台 |
| Pre-VLA (2605.22446) | validity + advantage | 动作块执行前验证 + 自适应重采样 | LIBERO 闭环 |

**三个对 RASE 最关键的发现**：
1. **主流纠正不换执行器**：rerank/rewrite/guide 全在同一策略自己的候选内完成；"切换策略"非主流且需要成功集合交叉。
2. **BOKBO = 我们困境的机制解释**：margin/disagreement 与动作噪声 σ 相关 0.98、与真实违规噪声底 → "margin 大才切换"必然跟着噪声走。
3. **oracle gain=0pp 问题在候选侧**：出路 = rerank 源自身多候选（RoboMonkey 不依赖第二策略）/ residual learning 制造优势（UPS）/ 切换 GT 标签全 0 时 verifier 只能学常数。

### 2.2 闭环纠正（correction 而非 selection，详见 `vla_correction_survey.md`，30+ 方法）

**四种纠正生成机制**：
1. **同策略重采样/重试（= RASE requery）**：TapSampling（Action-VAE 潜空间采样 + task-progress verifier）、CycleVLA（MBR 解码重采样，扰动修正 ~80%）、VLA-ATTC（按需 test-time compute）；
2. **残差修正（保 base，= RASE 冻结 VLA 之上加层）**：A2C2（加性修正头；**LIBERO Spatial 仅 +7pp vs 动态 Kinetix +23pp——静态桌面纠正空间小的直接文献证据**）、ReCoVLA（失败条件残差，sim→real 61.7%）、RedFlow（动作级纠正监督）；
3. **切换更强生成器（= RASE fallback 语义）**：Critic in the Loop（视觉 Critic 路由 VLA↔VLM，2603.05185）、SC-VLA（fast↔slow 反思）、Assistron（人类介入）；
4. **检测-触发（主流但回避仲裁）**：INSIGHT（π0-FAST 上 token 熵触发"何时求助"）、When-to-Trust-Imagination（自适应 continue 长度）、SPR（progress 停滞 rewind）。

**三个关键发现**：
1. **没有任何论文显式构建"same-root 反事实状态 → 多策略候选"数据集**——RASE 的数据协议是文献缺口，本身是贡献点；最接近的是 RePO-VLA 的 RAI（恢复段状态对齐）与 ALOE 的行动级 OPE（混合缓冲隐式反事实）。
2. **支配问题无人正面解决**：主流"检测器决定何时纠正"隐含假设"源策略在正常状态不差"——96-state 数据恰好证伪（处处更好 → 退化为无条件 fallback）；verifier 打分单调时同样退化。真正缓解支配的只有**引入纠正成本/预算**（INSIGHT 求助成本、ALOE 价值差、When-to-Trust 延长 continue、Critic-in-the-Loop 重试环终止规则）。
3. **LLM 镜像证据**：Cannot-Self-Correct-Yet（ICLR 2024）证明无外部反馈的内在纠正无效——纠正价值取决于信号源外部性与成功集合关系。

**破局三出路**（调研原话）：① 给 fallback 加成本/延迟/多目标（成功率×时间/安全）；② verifier 重采样增强源策略使成功集交叉（TapSampling 式）；③ ALOE 式 OPE 量化 ΔV(s)，把 0pp 变成严格数值证据 + 转证"检测-触发 requery + 成本预算"必要性。

### 2.3 恢复策略/安全过滤/仲裁（详见 `RASE_literature_survey.md`，7 方法族）

**四个关键发现**：

1. **支配困境有定理背书**：[Experts in an MDP](https://proceedings.nips.cc/paper_files/paper/2004/hash/421b3ac5c24ee992edd6087611c60dbb-Abstract.html)（Even-Dar et al., NIPS'04）regret 界证明：存在处处最优的固定专家时，任何状态级切换必然零增益——96-state 的 0 continue-only 是**定理推论**而非测量噪声。
2. **坐标系错位是根因**：[Recovery RL](https://arxiv.org/abs/2010.15920)/CBF/盾 的"风险"是**绝对安全**（约束违反、危险集），RASE 的"风险"是**相对策略优劣**——安全批评家不会在"OFT 更好但 SmolVLA 不违规"的状态触发，而相对比较器在单方支配下必须处处触发。
3. **任务级互补真实存在且被直接测量**：[RoboRouter](https://arxiv.org/abs/2603.07892)（语义表征+历史检索路由）、[MoIRA](https://arxiv.org/abs/2507.01843)（LIBERO 上文本路由验证）；互补性度量工具：Kuncheva & Whitaker 十种多样性度量（double-fault 最贴近）。
4. **残差修正不需要"源策略更好的状态"**：ResiP（冻结 chunked BC + 闭环残差 RL 反超）、RPD（RL 学生超越 VLA 教师）证明只需"误差可学"；不确定性门控残差（2506.17564）≈ RASE 风险模型的连续化版本。

**三个推荐方向**（调研原话）：① 任务级路由 + 状态级绝对安全保底（换坐标系避免支配）；② 从比较改修正（残差/蒸馏，绕开"需要相对优势"前提）；③ 支配变现为训练信号 + [Conformal Decision Theory](https://arxiv.org/abs/2310.05921) 校准保守切换（可证明风险上界）。

### 2.4 候选生成机制（VLA 动作分布三族，详见 `vla_candidate_generation_report.md`）

| 架构 | 生成机制 | 默认推理 | 同状态多候选来源 |
|---|---|---|---|
| π0 / π0.5 / SmolVLA | **flow matching**（随机噪声 → 10 步 Euler 积分） | 单次随机采样（**无 greedy 等价物**） | **批量 K 个噪声**（openpi `sample_actions` batch / Octo `sample_shape`），共享 prefix KV cache，最便宜 |
| π0-FAST / OpenVLA / RT-2 | 自回归离散 token | **greedy（temperature=0.0）** | 温度 0.4–0.7 采样（文献只验证 β=0.7；top-p/nucleus 无验证） |
| OpenVLA-OFT | **确定性 L1 回归** | 单前向，输出唯一 | **无**（只适合 fallback.persistent） |

关键结论：
- **SmolVLA 没有动作 temperature/top_p**（那是 VLM 文本解码参数）；动作头是 flow matching，随机性来自初始噪声。项目里 `select_env_action` 的 temperature 参数 = flow-matching 噪声缩放 `N(0, t²)`（W2 candidate sampling 先例），机制一致。
- flow 系"默认动作" = 一次随机采样；"候选" = K 个样本（关系是 1 样本 vs K 样本，不是众数 vs 样本）。
- 候选多样性几乎无直接度量文献 → 需在 LIBERO 自建度量（候选 L2 / 成功率差）；**LIBERO 任务多为单解任务，候选差异可能主要是执行时序/小幅轨迹变化**，必须先量化再设计仲裁。
- 成功先例：VGAS（best-of-N + 值模型选择，与 RASE 几乎同构——需差异化）、Do-What-You-Say、FM-Steer；**"采样无选择器"不涨成功率**。
- 落地建议：SmolVLA（450M）K=4–8 批量噪声做同策略多候选（最便宜）；π0.5-LIBERO（96.85）做质量上限/fallback；π0-FAST 温度采样做第二来源。

---

## 3. RASE candidates 生成机制与默认动作的关系

见独立文档 `progress/2026-08-20_candidate_generation_notes.md`（要点）：

| 候选 | 生成方式 | 与默认动作关系 |
|---|---|---|
| continue.source | 不生成 | = 默认动作（当前 chunk 剩余部分） |
| requery.source | 同策略重新推理（独立 seed/噪声） | 默认动作的同分布重采样 |
| resample.source | 同一次推理多次采样 | 同上 |
| fallback.persistent | 另一策略完整动作块 | 异分布候选 |
| abort.safe | 安全中止 | — |

代码事实（修正后）：
- `collect_same_root.py`：restore 同一 snapshot → 每候选生成 chunk（冻结）→ `single_chunk` 执行（禁超 native 长度）→ branch-end 快照 → recovery 评估；
- `rase_selector_loop.py`：决策点所有模型从同一状态生成 chunk → risk 打分 → LCB/UCB + abstain → 执行选中 chunk；
- `rase/collect/policy_step.py::select_env_action`：`temperature=None` → 默认推理；`temperature=t` → flow-matching 初始噪声 `N(0, t²)` 注入（**W2 candidate sampling 先例已存在**）；
- `forked_rollout.py`：SmolVLA `temperature=0.5` 随机续跑；π0-fast/π0.5 原生采样；`rollout_seed(state_key, candidate, rollout)` 派生可复现 seed。

**关键推论**：默认动作 = 分布的贪心实现；同策略候选 = 同分布的其他实现（需随机源：temperature/flow 噪声）；
跨策略候选 = 异分布。确定性策略下 continue ≡ requery（无多样性，capability mask）。

---

## 4. 借鉴与改进方案（解决 fallback 支配，按可行度排序）

### 方案 1（主线，符合 idea + 借鉴主流）：Risk-Guided Regeneration（风险引导重生成）

```text
决策点 t（同一物理状态 s_t）：
  source（SmolVLA flow matching）批量采样 K=4-8 个候选（K 个随机噪声，共享 prefix KV cache）
      ↓
  风险模型打分（same-root 反事实训练；BOKBO 式 margin 诊断 + 保形校准阈值）
      ├─ 有候选通过验证 → 执行（rerank，不切换）
      ├─ 全部高风险 → 拒绝-重采样循环（预算内；EVE 式条件引导可选）
      └─ 预算耗尽仍不行 → 才 fallback（最后手段）+ 成本记账
```

- **借鉴来源**：TapSampling/CycleVLA（requery 现成实现）、CheckVLA（reject + 保形阈值 + action-shuffled 消融）、EVE（verifier 作为生成条件）、BOKBO（保形 abstention + 分数轴诊断）、RoboMonkey（K 候选 rerank）；
- **创新组合**：same-root 反事实训练的 verifier 从"选择器"升级为"生成条件"（rerank + reject + guide 融合）——文献中无人把三者放同一仲裁框架；
- **可行性**：代码全在（`select_env_action` flow-matching 批量噪声、risk 模型管线、`decide` 保守仲裁、forked_rollout 采样）；E0/E1 先裁决多样性 + 机会。

### 方案 2（立即做，纯分析）：分数轴诊断 + 保形校准（BOKBO/CheckVLA 范本）

1. **margin vs σ vs 真实切换收益** 相关分析（BOKBO 式）——回答"我们的 abstain 阈值跟的是噪声还是真实收益"；
2. 真实切换标签的**保形校准阈值**替代启发式 margin（Conformal Decision Theory）；
3. **action-shuffled 消融**验证风险模型真用了动作信息（CheckVLA 范式）；
4. 产出 = 论文方法论贡献（switch precision-coverage 的统计化版本）。

### 方案 3（中算力）：ALOE 式行动级 OPE 量化支配

- 逐状态估计 ΔV(s) = V(fallback) − V(continue)（混合缓冲 + OPE 校正）；
- 若 96 状态全部显著为正 → "OFT 弱支配 SmolVLA"的严格数值证据（可发表负面结果）；
- 同时转证"检测-触发 requery + 纠正成本预算"的必要性（INSIGHT/When-to-Trust 路线：支配成立时剩余决策空间在"何时 requery"与"执行多长"）。

### 方案 4（中算力，创新性强）：把支配变成训练信号

- 用 fallback 的 same-root 结局训练**源策略的纠正生成器**（新组件，不改冻结 VLA 权重；UPS residual learning 路线）；
- 或 TapSampling 式：SmolVLA 侧重采样 + task-progress verifier，**扩大源策略成功集，把嵌套变交叉**。

### 方案 5（论文讨论）：成本维度的重新定义

- LIBERO 零成本 fallback 是支配域无增益的结构原因（调研确认：文献中真正缓解支配的只有"引入纠正成本/预算"）；
- 真机切换有真实成本（延迟/加载/能耗/切换风险）→ 多目标（成功率 × 时间/安全）下仲裁重新获得空间；
- 作为论文 discussion/future work，不作为主 endpoint。

### 方案 6（理论定位，支撑上述全部方案）：坐标系换位——任务级路由 + 状态级保底

- **定理**（Experts in an MDP）：状态级切换零增益 ⇔ 存在处处最优固定专家。96-state 数据 = 该定理的实证实例；
- **换坐标系**：RASE 状态级风险模型从"相对策略优劣"降级为"绝对安全保底"（Recovery RL 语义：约束违反/危险集），仲裁主层移到**任务级路由**（RoboRouter/MoIRA 模板，LIBERO 已验证）——H_within=0 恰好证明状态级冗余、任务级互补存在（51/45 任务分布）；
- 这保留了 RASE 的"跨策略 + 保守 + 可迁移"内核，但坐标系与文献对齐，不再对抗定理。

## 5. 创新点建议（增强创新性，保持 novelty boundary）

| # | 创新点 | 对应文献缺口 | 可行度 |
|---|---|---|---|
| C1 | **same-root 反事实数据集协议**（冻结状态 × 多策略候选 × 真实终局） | 调研确认：**没有任何论文显式构建**（最接近：RePO-VLA RAI、ALOE OPE） | 高（数据已有，修正协议已就位） |
| C2 | **Verifier-as-Generator-Condition**：风险模型从选择器升级为生成条件（rerank+reject+guide 融合） | RoboMonkey 只 rerank；EVE 只 guide；CheckVLA 只 reject | 中（E1 通过后） |
| C3 | **保守仲裁的统计化**：保形校准阈值 + precision-coverage 曲线 + action-shuffled 消融 | BOKBO/CheckVLA 各自部分覆盖，无人组合到跨策略场景 | 高（纯分析） |
| C4 | **支配的严格量化 + 域筛选方法论**：ΔV(s) OPE、嵌套 vs 交叉失败模式度量、H_within 统计检验 | 文献无"策略对成功集合包含关系"的度量 | 高（96-state 资产已就位） |
| C5 | **同策略候选多样性度量**（候选 L2 / 成功率差，LIBERO 自建） | 调研确认：无直接度量文献 | 高（E0 即产出） |
| C6 | **"嵌套 vs 交叉"失败模式的形式化**（Experts in an MDP 定理的实证 + double-fault 度量 + ΔV(s) OPE） | 文献无策略对成功集合包含关系的度量；0 continue-only 是定理推论 | 高（96-state 资产已就位） |

## 6. 最终建议（一句话执行顺序）

```text
1. E0：同策略候选多样性探针（SmolVLA flow matching K=4-8 批量噪声 / π0-FAST 温度采样）
   → 产出 C5（多样性度量）
2. E1：同策略 same-root pilot（真实终局）→ 候选间 H_within + oracle gain
   ├─ PASS → 方案 1（Risk-Guided Regeneration）主线：same-root verifier + rerank/reject/guide
   └─ FAIL → 方案 6 换坐标系：任务级路由（RoboRouter/MoIRA 模板）+ 状态级绝对安全保底
             + 方案 3（ALOE 式 OPE 量化支配，可发表负面结果）
3. 并行（纯分析）：方案 2 分数轴诊断 + 保形校准；C6 支配形式化
4. 中算力可选：方案 4（把支配变成训练信号：OFT 赢家轨迹蒸馏/残差）
```

核心原则不变：**先排除结构性失败（E0/E1/定理检查），再投入算力**；调研确认
same-root 反事实数据集是文献缺口（C1），RASE 的采集协议本身就是贡献。

---

## 6. 下一步实验（可行度高优先）

1. E0：同策略多样性探针（SmolVLA temp=0.5 / π0-fast 原生采样，3 次推理 L2）；
2. E1：同策略 same-root pilot（4-6 tasks × 4 states × 3 采样候选，真实终局，H_within + oracle gain）；
3. （若 E1 PASS）Verifier-Conditioned Regeneration 原型；
4. （并行，纯分析）支配/嵌套 vs 交叉失败模式的正式度量（dominance ratio / flip rate），作为域筛选统计量。
