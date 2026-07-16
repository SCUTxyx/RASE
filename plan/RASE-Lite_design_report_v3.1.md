# 技术设计报告 v3.1：面向 CVPR 2027 的轻量化 VLA 错误检测与修复外挂（RASE-Lite）

版本：v3.1　日期：2026-07-10　前置文档：v3.0（2026-07-08）

本版相对 v3.0 的核心变化（八条主线）：

1. **事实纠偏**：修正 "OFT 扰动下跌穿 30%" 与 "SmolVLA 复现 43–51%" 两处会被审稿人查表推翻的数字表述，全部改为按维度/按 suite 的精确引用。
2. **文献快照推进到 2026-07-10**：补入 v3.0 遗漏的近邻——**HELM**（frozen policy 内 rollback+replan，直接侵蚀 C3）、**VoLo**（离散恢复决策空间的 agent 版本）、ReCoVLA、VLA-FAIL、RePO-VLA、AFIL、FailSafe、LiLo-VLA、FARL、LIBERO-PRO、frozen-VLA value probing、vla-eval，并更新防攻击矩阵与四象限。
3. **C3 分界再收窄**：从"统一 fallback 空间"收窄为 "**learned（RL）统一 fallback selector** + FEB 度量协议 + per-candidate ground truth"——rule/agent 版统一恢复空间（VoLo）与 rollback+replan 组合（HELM）均已存在。
4. **FEB 措辞降级**："FEB 定理" 改称 "**FEB 恒等式（identity）**"，数学重量转移到 N-Scaling Lemma 与 partial-fallback 方法的**可测量 FEB**，主动规避 trivial-theorem 攻击。
5. **NGC 判定统计功效修复**：\(N_{\text{roll}}=5\) 的固定采样改为**两阶段自适应采样 + Wilson 上置信界判据**，并新增 per-state 物理可逆性标注。
6. **风险信号因果拆分**：\(d_t\) 拆为 \(d_t^{\text{post}}\)（滞后诊断）与 \(d_t^{\text{pre}}\)（前瞻散度），修复部署时无 fork 的因果矛盾；新增 ACC（action chunk consistency）与 frozen-VLA value probe 两路廉价信号。
7. **matched-budget 记账规则显式化**：oracle rollouts 计入 selector 总交互预算，堵公平性攻击。
8. **时间线压缩**：22 周 → 18 周（今日起算），预印本目标 9 月下旬，全文按 11 月上旬截稿规划。

---

## 目录

- §0 执行摘要与核心决策一览（v3.1 更新）
- §0.5 v3.0 问题诊断清单（本版新增）
- §1 论文市场深度调研（2026-07-10 快照）与防攻击矩阵
- §2 基础 VLA 选择：双轨制论证（数字纠偏版）
- §3 NGC-Plus benchmark：构建方法、标注协议、统计功效、成本预算
- §4 风险信号模块：四路信号与因果拆分
- §5 FEB 形式化：恒等式、覆盖范围、与 partial-fallback 阵营的边界
- §6 RL 训练框架：工程设计与预算记账
- §7 训练数据来源
- §8 实验矩阵（v3.1 调整）
- §9 CVPR 2027 投稿策略与叙事
- §10 风险矩阵与降级预案
- §11 18 周执行计划（含降级序）
- 附录 A：v3.0 → v3.1 变更速查

---

## 0. 执行摘要与核心决策一览（v3.1 更新）

### 0.1 一句话定位（CVPR 版，v3.1 修订）

在视觉扰动下（相机视角、物体布局、机器人初始位姿），LIBERO-Plus（CVPR 2026）已实证：即使在 clean LIBERO 上接近饱和（95%+）的 VLA，在 camera / robot-initial-state 维度上也会大幅塌缩（如 OpenVLA-OFT 在 robot 维度降至 31.9%、camera 维度 56.4%，总体 69.6%；较弱模型跌破 30%），并大量产生"所有候选动作都不可接受"的 No-Good-Candidate（NGC）状态。我们（i）形式化 candidate-set-internal selection / verification 类方法在 NGC 状态上 Forced-Execution Bias 恒为 1 的结构性恒等式，并对 partial-fallback 类方法给出**可测量的 FEB**；（ii）构建首个由视觉扰动诱导、带 per-candidate 可恢复性 ground-truth 标注的 **NGC-Plus** benchmark（含 FEB 度量协议、物理可逆性标注与"扰动类型→NGC 产率"因果分析）；（iii）提出一个不动 VLA 一个参数的轻量外挂：以四路廉价风险信号（DINOv2 视觉一致性、V-JEPA 2 前瞻/滞后 latent 信号、action-chunk consistency、frozen-VLA value probe）为输入，用 RL 训练的 3.5M selector 在"执行候选 \(\cup\) {ROLLBACK, RESAMPLE, REPLAN, WAIT, ABSTAIN}"的统一离散空间中决策。在**计入 warm-start 数据的**匹配环境交互预算下，selector-RL 的样本效率显著高于直接 RL 微调 VLA，且相对 rule/agent 驱动的统一恢复空间（VoLo 式）与 rollback+replan 组合（HELM 式），learned selector 在触发时机与 fallback 选型上均显著更优。

**v3.1 措辞纪律（在 v3 基础上追加两条）**：

- 不再写"强 VLA 扰动下跌穿 30%"这类笼统表述；一律写"OFT 在 robot-initial-state 维度 31.9%（LIBERO-Plus, CVPR 2026, Table X）"式的按维度精确引用。"95%→<30%" 仅用于原论文摘要引述并注明其针对的模型与维度范围。
- 不再宣称"统一离散 fallback 空间"本身是首创；首创性锚定在 **learned selector + FEB 协议 + per-candidate ground truth** 的组合上。

### 0.2 核心决策一览表

| # | 决策 | 选择 | 一句话理由 | 详见 |
|---|------|------|-----------|------|
| D1 | 主评估平台 | LIBERO（不退化验证）+ LIBERO-Plus（主战场，聚焦 L3+ 与组合扰动） | clean 已饱和；LIBERO-Plus 中 CVPR 2026；训练侧鲁棒化（VLA-GSE 81.2%）正在收窄窗口，故聚焦高难度档 | §2、§3 |
| D2 | 主线基础 VLA | SmolVLA-0.45B（官方 LIBERO checkpoint，固定 lerobot 版本与 `n_action_steps`） | 4090 可承载全栈；官方权重社区复现总体约 73%（Long 仅 43–56%），clean + perturb 双重 headroom；须自测并锁定评测配置 | §2 |
| D3 | 副线基础 VLA | OpenVLA-OFT 7B（仅推理） | OFT clean 97%+ 但 LIBERO-Plus robot 维度 31.9%——强模型在特定扰动维度同样有巨大 headroom | §2 |
| D4 | 世界模型 | V-JEPA 2 ViT-L（frozen）+ DINOv2-B | latent 预测跨域鲁棒、延迟可控；Foresight 已佐证 V-JEPA 2 类特征在失败预测上 work | §4 |
| D5 | Selector 训练 | Offline warm-start（BC+CQL）→ Online Double+Dueling DQN | 离散小动作空间 + off-policy 样本效率 + 单卡吞吐约束 | §6 |
| D6 | NGC oracle | OFT（temp 0.5）续完为主 + SmolVLA 续完交叉验证 + 两阶段自适应采样 + 人工核查 300–500 状态 | cross-oracle + Wilson 判据 + 人工核查三重堵循环性攻击 | §3 |
| D7 | 训练数据 | 底座全下载、selector 数据全自采；oracle rollouts 计入 selector budget | 无公开数据集含 (candidate set, fallback, outcome) 三元标注 | §7 |
| D8 | 目标会议 | CVPR 2027（按 2026-11 上旬截稿规划）+ arXiv 预印本 9 月下旬抢命名权 | 空白按每月 3–4 篇速度收窄（v3.0 估计 2–3 篇已偏保守） | §9 |

### 0.3 三层贡献（CVPR 口味排序，v3.1 微调）

```
C1 (Benchmark, 唯一 headline): NGC-Plus —— 首个由视觉扰动诱导、带 per-candidate
    可恢复性 ground-truth 标注的 NGC 状态基准 + FEB 度量协议 + 物理可逆性标注
    + "扰动类型→NGC 产率"因果分析 + "鲁棒训练压缩但不消灭 NGC"实证
    （对 OFT+ 与开源鲁棒化模型测 NGC 产率）。

C2 (System component): 四路廉价风险信号 (φ_t, d_t^pre, d_t^post, c_t^ACC, v_t^probe)
    作为 selector 输入。诚实定位：每一路信号都有先例（Foresight / VLA-FAIL /
    value probing），我们的贡献是"信号组合的新用途 + optimism-bias 诊断"。

C3 (System): learned RL selector 在统一 5-fallback 离散空间中决策；
    FEB 恒等式（candidate-internal 家族）+ partial-fallback 阵营的可测量 FEB
    + matched-interaction-budget（含 warm-start 记账）样本效率分析
    + learned-vs-rule/agent 对照（HELM-lite / VoLo-lite / CycleVLA-lite / B2FF-lite）。
```

---

## 0.5 v3.0 问题诊断清单（本版新增，修订依据）

| # | 问题 | 严重度 | v3.1 处置 |
|---|------|--------|----------|
| P1 | "OFT 扰动下跌穿 30%" 与 LIBERO-Plus CVPR 版数据（OFT 总体 69.6%）不符 | ★★★★★（事实错误） | §0.1、§2 全部改按维度精确引用 |
| P2 | SmolVLA "复现 43–51%" 实为 LIBERO-Long 单 suite；整体社区复现约 73%；且对 `n_action_steps`/版本极敏感 | ★★★★ | §2.4 修正 + 复现纪律强化（引 vla-eval 复现陷阱） |
| P3 | 漏引 HELM（frozen policy 内 rollback+replan）——ROLLBACK+REPLAN 组合已存在 | ★★★★★（novelty） | §1.1.9 新增；C3 分界收窄；E8 加 HELM-lite |
| P4 | 漏引 VoLo（CONTINUE/REPLAN/REWRITE/... 离散恢复决策空间，agent 驱动）——"统一离散恢复空间"概念已有 rule/agent 版 | ★★★★★（novelty） | §1.1.10 新增；C3 措辞改"learned selector"为核心；E8 加 VoLo-lite |
| P5 | 漏引 ReCoVLA / VLA-FAIL / RePO-VLA / AFIL / FailSafe / LiLo-VLA / FARL / LIBERO-PRO / value-probing / vla-eval | ★★★ | §1.2 表补齐 |
| P6 | 训练侧鲁棒化进展快于预期（VLA-GSE 81.2%、ABot-M0 80.5%、VLANeXt 80.1% zero-shot on LIBERO-Plus）——motivation 窗口收窄 | ★★★★ | D1 聚焦 L3+ 与组合扰动；E5 扩为"对鲁棒化 SOTA 测 NGC 产率" |
| P7 | FEB "定理"是定义的重言式，会被攻击 trivial | ★★★★ | §5 降级为"恒等式"，重量转移到 N-Scaling + 可测量 FEB |
| P8 | \(N_{\text{roll}}=5\) 在 \(\tau=0.5\) 边界统计功效不足，Set C 假阳性高 | ★★★★（benchmark 质量） | §3.4 两阶段自适应采样 + Wilson 上置信界判据 |
| P9 | \(d_t\) 部署时无 fork、无"实际观测"可比——因果矛盾 | ★★★★（设计缺陷） | §4 拆分 \(d_t^{\text{pre}}\)/\(d_t^{\text{post}}\) |
| P10 | matched-budget 未计 warm-start 的 oracle rollouts，E3 公平性可被攻击 | ★★★ | §6.7 显式记账规则 |
| P11 | ROLLBACK 逆动作重放对物理不可逆事件无效 | ★★★ | §3.3 Step 5 增加物理可逆性标注；§6.1 ROLLBACK 可达域声明 |
| P12 | REPLAN-lite（文字改写）在 SmolVLA 上大概率无效（LIBERO-Plus 证明模型忽略语言） | ★★★ | §6.1 增加 goal-image REPLAN 备选；E9 增加两种 REPLAN 对照 |
| P13 | 22 周计划起点已过，且 CVPR 全文截稿通常 11 月上旬 | ★★★ | §11 压缩为 18 周 |

---

## 1. 论文市场深度调研（2026-07-10 快照）与防攻击矩阵

**结论先行（v3.1 更新）**：C1（扰动诱导 NGC benchmark + per-candidate 可恢复性 ground truth + FEB 协议）截至 2026-07-10 仍是空白。但右下象限（冻结 VLA + fallback）的密度比 v3.0 评估的还要高：除 CycleVLA、B2FF、AEGIS、FAR、RC-NF、KnowNo 外，还有 **HELM**（rollback+replan within frozen policy）、**VoLo**（agent 式离散恢复决策空间）、**ReCoVLA**（VLM 检测 + per-category residual 恢复）、LiLo-VLA（retry+backtrack）。"统一恢复决策空间"与 "rollback+replan 组合" 都已有 rule/agent 实例。我们的组合空白精确化为：

\[
\text{learned RL selector} + \text{per-candidate recoverability ground truth} + \text{FEB 协议} + \text{扰动诱导 NGC benchmark}.
\]

近邻出现速度实测约每月 3–4 篇（v3.0 估计 2–3 篇偏保守）。预印本时间从 9–10 月提前锁定为 **9 月下旬**。

### 1.1 第一梯队：必须正面交锋的近邻

#### 1.1.1 LIBERO-Plus（CVPR 2026）——叙事前提（数字纠偏）

- 已被 CVPR 2026 接收（Open Access 可查）。7 维 21 子维扰动，10,030 任务，动态难度分层。核心发现：模型对 camera viewpoint 与 robot initial state 极度敏感；语言指令大量被忽略；**组合扰动呈负组合泛化**。
- **v3.1 数字纪律**：论文中引用其 per-model per-dimension 数字（如 OpenVLA 总体 15.6%、OFT 总体 69.6%、OFT camera 56.4% / robot 31.9%），"95%→<30%" 仅作原文摘要转述。我们的 headroom 论证改为："即使总体 69.6% 的 OFT，在 robot 维度也只有 31.9%——NGC 状态在该维度富集"。
- 官方仓库提供 OpenVLA-OFT+（LIBERO-plus 数据 mix-SFT 版本）与无腕部观测/mix-SFT 变体——E5 直接使用。
- 攻击话术与应对同 v3.0（per-candidate 标注新维度 + 因果分析 + built-on-top）。

#### 1.1.2 LIBERO-PRO（v3.1 新增，同类平台）

- 另一个 LIBERO 鲁棒/公平评测扩展（"beyond memorization"）。related work 必引，并说明选择 LIBERO-Plus 的理由（CVPR 正式出版、难度分层、pip 直换、社区采用度：StarVLA-α / VLA-GSE / LoopVLA / Afford-VLA 等均已在其上评测）。

#### 1.1.3 FOREWARN（RSS 2025）——WM-based selection 最近邻

同 v3.0（WM-as-Decision vs WM-as-Sensor 表格；FOREWARN-lite baseline 实测 FEB）。

#### 1.1.4 Foresight（2606.23085）——威胁 C2

同 v3.0（信号同源承认 + "信号的新用途"话术 + C2 降格）。v3.1 补充：C2 的信号扩为四路（§4），单一信号首创性问题进一步稀释。

#### 1.1.5 B2FF（2606.09258）——C1/C3 交界最危险近邻

- 补充实测数字：failure-injected LIBERO 上把 baseline 从 56.3% 提到 74.0%；机制是 pre-imagined milestone bank + recoverability-aware selector + 固定视觉目标锚定。
- 四点差异不变（视觉扰动诱导 vs 失败注入；发布 per-candidate ground truth vs 选择启发式；FEB 协议；含 ABSTAIN/REPLAN）。
- **v3.1 新增借鉴**：B2FF 的 milestone-as-visual-goal 机制是 REPLAN 的 goal-image 变体的现成实现思路（应对 P12）。

#### 1.1.6 AEGIS（2606.06660）——escalation 范式

- 同 v3.0。**v3.1 关键补充**：AEGIS 的 related work 是本方向最新的系统梳理（FIPER、FAIL-Detect、ReconVLA、Sentinel、Pre-VLA、INSIGHT、HELM、LiLo-VLA、FailSafe、FPC-VLA、Confidence-Gated Autonomy、FARL），我们的 related work 以其为底本查漏。
- AEGIS 自述其空白："对 frozen VLA 内部逐步早期预警 + 升级到更强独立 policy + 恢复度量 + 因果控制"的四合一无人占据——注意它声称的空白与我们不冲突（它无 benchmark、无 per-candidate ground truth、无 learned 统一空间），citation 时点明。

#### 1.1.7 HELM（v3.1 新增，P3，直接侵蚀 C3）

- **它做了什么**：在同一个 frozen policy 内做恢复（recover-within-the-same-policy）：rollback 到 checkpoint + replan。被 AEGIS 认定为其最近邻并作 budget-matched baseline 复现。
- **危险性**：我们 5-fallback 中最重的两个（ROLLBACK、REPLAN）的组合已被单独成文。若我们仍把"ROLLBACK+REPLAN 于 frozen VLA"当卖点，HELM 是直接反例。
- **应对**：
  1. C3 卖点收窄为 learned selector（HELM 的触发与恢复流程是 rule/固定的）+ 统一空间联合优化（\(\alpha_b\) sweep 证明的 trade-off 性质）+ FEB 协议下的量化对比。
  2. E8 增加 **HELM-lite baseline**（rollback-to-checkpoint + replan，rule 触发），预期其 FEB 低于 candidate-internal 但 broken-success 显著高于我们（固定触发规则误伤成功轨迹）。
  3. related work 单列 "recover-within-policy" 小节：HELM、LiLo-VLA、FailSafe、Pre-VLA 归此类。

#### 1.1.8 VoLo（2606.07723）——（v3.1 新增，P4，动摇"统一空间"首创性）

- **它做了什么**：开放词汇长时程操作的 physical orchestrator；恢复阶段进入 deliberation context，在 CONTINUE / REPLAN / REWRITE / GRASP / PLACE 等离散选项中选择；REWRITE 即"用新的子目标指令重跑 VLA"（正是我们 REPLAN-lite 的语义）。
- **危险性**："在统一离散空间中做恢复决策（含指令改写）"已有 agent/LLM-deliberation 实现。若审稿人拿 VoLo 对线，"5-fallback 统一空间"作为概念不再新。
- **应对**：
  1. 论文措辞：统一空间的**概念**归功于 orchestrator 类工作（VoLo 及经典 LLM-planner 线），我们的贡献是（a）把该决策**学习化**（3.5M RL selector，无在线 LLM，延迟低两个数量级）；（b）在 NGC-Plus 上给出该决策问题的第一个带 ground truth 的量化基准与 FEB 协议；（c）证明 learned 决策在触发时机/选型上优于 agent-deliberation。
  2. E8 增加 **VoLo-lite baseline**（VLM deliberation 在同一 5-fallback 空间上选择），与我们同空间对照——这是最干净的 learned-vs-agent 实验，比 CycleVLA-lite 更有说服力（动作空间完全对齐，只变决策器）。
- **附带红利**：VoLo-lite 同空间对照使 "5-fallback 是 engineering" 的攻击也被间接回应（同一空间下决策器差异即是科学问题）。

#### 1.1.9 CycleVLA / FAR / SimpleVLA-RL

同 v3.0（分别为 partial-fallback rule 阵营、retry 阵营、matched-budget 假想敌；三层防御不变）。v3.1 对 SimpleVLA-RL 防御的补充见 §6.7 记账规则。

### 1.2 第二梯队：related work 必引、择优做 lite baseline（v3.1 增补行加粗）

| 工作 | 一句话定位 | surgical 差异 | 处理方式 |
|------|-----------|--------------|---------|
| SAFE（CoRL 2025） | VLA 内部特征失败检测器 | 只检测不恢复 | 复用作 \(p^{\text{fail}}\) 输入 + baseline |
| **VLA-FAIL（2606.21386）** | 高效失败检测：action chunk consistency（ACC）+ LLM 描述检测 | 只检测；ACC 是免费信号 | **ACC 直接纳入 selector 输入 \(c_t^{\text{ACC}}\)（§4）**；引用 |
| **ReCoVLA（2606.09630）** | frozen VLA + VLM 语义失败分类 + per-category residual policy 恢复 | 恢复靠预训练 residual 库（每类一个 policy），非统一决策空间；VLM 在线 | 引用；归 escalation/residual 阵营；讨论其"未知类别不恢复"与我们 ABSTAIN 的关系 |
| **RePO-VLA（2605.09410）** | 恢复驱动的 policy 优化（训练 VLA 学恢复，TSHR 切片） | training-time，改 policy | 引用；training-side 恢复代表作 |
| **AFIL（2605.08434）** | 失败轨迹作 diffusion/flow VLA 的负向指导 | training-time | 引用 |
| **FailSafe（2510.01642）** | VLM 推理失败原因 + 修正末端位姿 nudge | recover-within-policy，连续修正 | 引用；归 recover-within-policy |
| **LiLo-VLA** | planner+VLA 栈内 retry + backtrack（LIBERO-Long） | 模块化 rule 恢复 | 引用 |
| **FARL** | RL post-training WM safety critic + offline recovery | training-time regime | 引用 |
| **Frozen-VLA value probing（2605.28527）** | frozen VLA/DINOv2/CLIP 特征线性 probe 可读出 MC outcome，probe 读数可在线改变动作选择 | 只做 probing 与 reranking（candidate-internal） | **probe 作为 selector 第四路输入 \(v_t^{\text{probe}}\)（§4）**；其 reranking 用法归 FEB=1 家族 |
| **vla-eval（2603.13966）** | VLA 统一评测 harness；记录复现陷阱（单参数 55pp 波动） | 工具性工作 | §2 复现纪律引用；考虑直接用其 harness |
| **StarVLA-α / ABot-M0 / VLANeXt / VLA-GSE / LoopVLA / Afford-VLA** | LIBERO-Plus 上的鲁棒化/泛化模型（VLA-GSE 81.2% zero-shot） | training-time 鲁棒化 | 关键引用：论证"训练侧在追赶但压缩≠消灭"（E5 扩容） |
| Sentinel-VLA / VLA-ATTC / VeriSpace / Pre-VLA / FPC-VLA / AHEAD / VLA-JEPA / MiraBench / VLA-RFT / RC-NF / KnowNo / rerankers / PLD | 同 v3.0 | 同 v3.0 | 同 v3.0 |

### 1.3 防攻击总矩阵（v3.1 更新版，新增行加粗）

| 攻击 | 强度 | 应对（实验/话术） |
|------|------|------------------|
| "为什么不直接 RL 微调 VLA（SimpleVLA-RL 97.6%）" | ★★★★★ | E3 matched-budget（含 warm-start 记账，§6.7）+ N-Scaling + E4 FEB 实测 + cross-backbone + 诚实声明单卡 LoRA 非原版复现 |
| **"HELM 已做 frozen policy 内 rollback+replan"** | ★★★★★ | learned vs rule 触发/选型；E8 HELM-lite 实测 broken-success 与 FEB；C3 卖点 = learned selector + FEB 协议 |
| **"VoLo 已有统一离散恢复决策空间（含指令改写）"** | ★★★★★ | 概念归功 orchestrator 线；贡献 = 学习化（3.5M vs 在线 LLM）+ ground-truth 基准 + E8 VoLo-lite 同空间对照 |
| "NGC-Plus 是 LIBERO-Plus 换皮" | ★★★★ | 同 v3.0 |
| "B2FF 已做 recoverability-aware 恢复" | ★★★★ | 同 v3.0 四点差异 + B2FF-lite |
| "Foresight 已用 V-JEPA 2 做失败信号" | ★★★★ | 同 v3.0；四路信号组合进一步稀释单信号首创性问题 |
| **"你引用的塌缩数字不成立（OFT 总体 69.6%）"** | ★★★★（自伤型） | v3.1 全文按维度精确引用；headroom 论证锚定 robot 维度 31.9% |
| "鲁棒训练（VLA-GSE 81.2%）会消灭 NGC" | ★★★★ | E5 扩容：对 OFT+ 与至少一个开源 80%+ 模型测 NGC 产率与 FEB；N-Scaling 论证 |
| "FEB 定理 trivial（定义重言式）" | ★★★★ | §5 主动降级为恒等式；数学重量在 N-Scaling 与 partial-fallback 可测量 FEB；恒等式的价值 = 度量协议而非定理 |
| **"Set C 判定统计功效不足（5 次 rollout 噪声）"** | ★★★★ | §3.4 两阶段自适应采样 + Wilson 上置信界；发布 per-state 置信度 |
| **"d_t 部署时没有实际观测可比"** | ★★★★ | §4 拆分 pre/post 两路信号，各自延迟语义明确 |
| **"matched-budget 没算你的 oracle rollouts"** | ★★★ | §6.7 记账规则：warm-start 数据计入 selector 总预算 |
| "oracle 循环性" | ★★★★ | 同 v3.0 三重防御 + Wilson 判据 |
| "WM optimism bias 污染信号"（MiraBench） | ★★★ | E6 诊断不变 |
| "ROLLBACK 不可部署 / 物理不可逆" | ★★★ | 主表逆动作重放语义 + **per-state 物理可逆性标注**（§3.3）+ ROLLBACK 可达域声明 |
| "REPLAN-lite 不算真正 replan / 对 SmolVLA 无效" | ★★★ | E9 双 REPLAN 对照（text-rewrite vs goal-image）；若 text 版无效如实报告并以 goal-image 版为主 |
| "sim 数值不代表真机" | ★★★ | 同 v3.0（FEB 恒等式 simulator-independent） |
| "5 个 fallback 是 engineering" | ★★ | VoLo-lite 同空间对照将问题转化为决策器科学问题 + per-fallback ablation |

### 1.4 文献空白确认（2026-07-10 四象限，v3.1 更新）

\[
\begin{array}{c|cc}
 & \text{训练 VLA} & \text{冻结 VLA} \\
\hline
\text{无 fallback 概念} & \text{SimpleVLA-RL, VLA-RFT, VLA-JEPA, VLA-GSE, RePO-VLA, AFIL} & \text{rerankers, Pre-VLA, Foresight, AHEAD, VeriSpace, VLA-FAIL, value-probing} \\
\text{有 fallback/恢复} & \text{PLD, FPC-VLA, FAR, Sentinel-VLA, FARL} & \text{CycleVLA, B2FF, AEGIS, RC-NF, KnowNo, HELM, VoLo, ReCoVLA, LiLo-VLA, FailSafe} \\
\end{array}
\]

右下象限密度已很高，但按"决策器类型 × 是否有 ground-truth 基准"再切一刀：所有右下工作的决策器均为 rule / heuristic / VLM-agent / 预训练 residual 库；**无一是 learned RL selector**；**无一发布 per-candidate 可恢复性 ground truth**；**无一给出 FEB 类度量协议**。空白仍在，但只剩这一个精确组合点，且窗口以每月 3–4 篇的速度收窄。

---

## 2. 基础 VLA 选择：双轨制论证（数字纠偏版）

### 2.1 headroom 悖论（表述修正）

- 强 VLA（OFT 7B）：clean 97%+，clean 无 success gain。
- 弱 VLA（SmolVLA）：headroom 充足但会被攻击"弱模型假象"。

### 2.2 破局点：LIBERO-Plus 的按维度 headroom（v3.1 精确化）

\[
\text{headroom}_{\text{clean}}(\text{OFT}) \approx 3\%,\qquad
\text{headroom}_{\text{robot-dim}}(\text{OFT}) \approx 68\%,\qquad
\text{headroom}_{\text{camera-dim}}(\text{OFT}) \approx 44\%.
\]

叙事锚点：OFT 总体 69.6%，但 robot initial state 维度 31.9%、camera 维度 56.4%（LIBERO-Plus CVPR 2026 数据；OpenVLA 非 OFT 版总体仅 15.6%）。NGC 状态采集聚焦这两个维度 + L3 以上难度 + 组合扰动（负组合泛化使 NGC 富集）。

**训练侧追赶的对冲**：VLA-GSE 类 PEFT 鲁棒化已把 LIBERO-Plus zero-shot 推到 81.2%。我们的回应：（1）81.2% 意味着仍有约 19% 失败测度，其中的 NGC 子集正是我们的对象；（2）E5 扩容为对鲁棒化模型直接测 NGC 产率；（3）N-Scaling：压缩测度 ≠ 消灭测度。

### 2.3 双轨制方案（不变）+ 诊断轨（OFT+）

同 v3.0 表格。

### 2.4 主线 SmolVLA 复现纪律（v3.1 强化，P2）

- **修正后的事实**：官方 `HuggingFaceVLA/smolvla_libero` checkpoint 社区复现：object ≈ 93%、goal ≈ 81%、spatial ≈ 63–73%、long ≈ 43–56%，总体 ≈ 73%（论文报告均值 87.3%）。"43–51%" 仅指 LIBERO-Long。
- **复现敏感性**（vla-eval 记录的教训）：`n_action_steps`（1 vs 10 结果差异巨大）、lerobot 版本（0.5.x）、mujoco 版本、数据集版本（HuggingFaceVLA/libero v3.0）、proprio 来源、绝对/增量动作模式、评测 crop——任何一项错配可造成两位数百分点波动。
- **纪律**：W1 用固定 commit + 固定 `n_action_steps=10, num_steps=10` 实测四 suite（每任务 \(\ge 50\) episodes），论文报自测数并附完整配置；全项目 pin 死 `lerobot`、`mujoco`、`LIBERO-plus` 三个版本号并写入 repo 的 lockfile。详见《VLA 复现与微调指南》。

### 2.5 写进论文的标准话术（v3.1 修订）

> "We deliberately evaluate on two backbones at opposite ends of the capability spectrum. On SmolVLA-0.45B, headroom exists even in the clean setting; on OpenVLA-OFT (97%+ clean success, 69.6% overall on LIBERO-Plus), headroom is exposed dimension-wise: success drops to 31.9% under robot-initial-state perturbations and 56.4% under camera perturbations (LIBERO-Plus, CVPR 2026). We further evaluate OpenVLA-OFT+ (mix-SFT on LIBERO-Plus data) and a state-of-the-art robustness-hardened open model, showing that NGC states persist under robustness training and that FEB remains 1 for all candidate-internal selection—forced-execution bias is structural, not an artifact of weak or non-hardened base policies."

---

## 3. NGC-Plus benchmark：构建方法、标注协议、统计功效、成本预算

### 3.1 NGC 状态的形式化定义（不变）

给定扰动状态 \(s\)、候选集 \(\mathcal{A}(s)=\{a_1,\dots,a_K\}\)（\(K=8\) 主设定），候选可恢复性价值：

\[
r(s,a_i) = \mathbb{E}\big[R^{\pi_{\text{cont}}}(s,a_i)\big],\qquad R\in\{0,1\}.
\]

阈值 \(\tau\)（主设定 0.5，敏感性 \(\{0.3,0.5,0.7\}\)）。NGC 状态：\(\max_i r(s,a_i)<\tau\)。Set A / B / C 三分同 v3.0。

### 3.2 FEB 度量协议（表述随 §5 调整）

对 candidate-internal 方法 \(m\)（\(\forall s,\ m(s)\in\mathcal{A}(s)\)）：

\[
\text{FEB}(m) = \mathbb{E}_{s\in\text{NGC}}\big[\mathbb{1}[m(s)\in\mathcal{A}(s)]\big] \equiv 1 \quad (\text{恒等式，见 §5}),
\]

对 partial-fallback 方法，FEB 是**可测量的经验量**（触发规则未覆盖的 NGC 状态比例）。报告协议：Set C 上报 FEB、broken-success、net-success；所有方法同时报 Set A/B 上的 clean-regret（在可恢复状态上因 fallback 损失的成功率），防止"全 ABSTAIN 刷 FEB"。

### 3.3 状态采集流水线（v3.1 两处修改）

Step 1–4 同 v3.0（详细施工见《NGC-Plus 数据采集指南》）。修改：

- **Step 1 修改**：扰动采样聚焦 camera + robot-initial-state 两维的 L3–L5 档 + 双因子组合扰动（利用 LIBERO-Plus 的负组合泛化发现），使 NGC 产率最大化、并保证对 80%+ 鲁棒化模型仍有足够 NGC 密度。
- **Step 5 扩充（P11）**：因果分析标注之外，为每个 NGC 状态增加**物理可逆性标签** \(\rho(s)\in\{\text{reversible},\ \text{contact-irreversible},\ \text{task-irreversible}\}\)（自动规则：接触事件检测 + 物体位姿变化阈值 + 人工抽查），用于（a）ROLLBACK 可达域声明；（b）分析 fallback 各自的适用域。

### 3.4 Set C 判定的统计功效（v3.1 新增，P8）

问题：\(N_{\text{roll}}=5\) 时 \(\hat{r}\) 的 95% Wilson 区间宽度约 ±0.35–0.45，\(\tau=0.5\) 边界附近误判率不可接受。

**两阶段自适应判据**：

1. 第一阶段每个 \((s,a_i)\) 跑 \(N_1=3\) 次。若 \(\hat{r}\) 的 Wilson 95% 区间完全在 \(\tau\) 一侧（例如 0/3 成功 → 上界 0.56 仍跨界，3/3 → 下界 0.44 跨界；实际早停主要发生在极端计数），直接定案；否则进入第二阶段补至 \(N_2=10\)。
2. **Set C 判据（保守）**：\(s\in\) Set C 当且仅当对所有 \(i\)，\(\hat{r}(s,a_i)\) 的 Wilson 95% **上界** \(<\tau\)。即宁可漏判不可错判——Set C 假阳性直接伤害 benchmark 公信力，漏判只损失规模。
3. 发布物中附 per-state 的 \(\{\hat{r}_i, N_i, \text{CI}_i\}\) 全量数据，允许使用者用自己的判据重切。

成本影响：期望 rollout 数从 \(K\times 5\) 变为 \(K\times(3\sim10)\)，边界状态占比按 30% 估计，期望约 \(K\times 5.1\)，与原预算基本持平（因为大量明显失败的候选 3 次即定案）。

### 3.5 oracle 循环性应对（v3.0 三重防御不变）

cross-oracle Cohen's \(\kappa\)、人工核查 300–500 状态、附录透明度——不变，且与 §3.4 的置信度发布叠加。

### 3.6 成本预算（随 §3.4 微调）

期望 rollout 总量与 v3.0 持平：约 \(2{,}000\times 8\times 5.1 \approx 82{,}000\) 条（Set C 部分），加 Set B、cross-oracle、复核，总量 **约 1,000–1,900 GPU·h**。W2 前实测单条 rollout 耗时后重估（行动项不变）。

---

## 4. 风险信号模块：四路信号与因果拆分（v3.1 重写）

### 4.1 信号清单（\(d_t\) 拆分，P9）

| 信号 | 定义 | 时机 | 先例（诚实归因） |
|------|------|------|------------------|
| \(\varphi_t\) | DINOv2-B 当前观测与近期观测/目标观测的特征距离 | 实时 | 通用 |
| \(d_t^{\text{pre}}\) | **前瞻**：对各候选 \(a_i\) 用 V-JEPA 2 做短程 latent rollout，取候选间预测终态的散度 + 与"成功流形"参考 latent 的距离 | 执行前，可部署 | Foresight（动作条件 WM 失败特征） |
| \(d_t^{\text{post}}\) | **滞后**：上一个已执行 chunk 的 WM 预测 latent vs 实际观测 latent 的误差 | 执行后一拍，可部署 | Foresight |
| \(\sigma_t\) | \(d^{\text{pre}}\) 的多次采样方差（不确定性） | 执行前 | 通用 |
| \(c_t^{\text{ACC}}\) | action chunk consistency：受控重叠水平线下相邻 chunk 的动作不一致度 | 实时，免费 | VLA-FAIL |
| \(v_t^{\text{probe}}\) | frozen VLA（或 DINOv2）特征上的线性 value probe 读数（MC outcome 回归） | 实时，免费 | frozen-VLA value probing（2605.28527） |
| \(p_t^{\text{fail}}\) | SAFE 式内部特征失败概率 | 实时 | SAFE |

部署语义澄清（写进论文）：训练时可用 state-fork 计算逐候选的真实 \(d^{\text{post}}\) 作监督/特征；**部署时 selector 只消费可实时获得的 \(\{\varphi, d^{\text{pre}}, d^{\text{post}}(\text{上一拍}), \sigma, c^{\text{ACC}}, v^{\text{probe}}, p^{\text{fail}}\}\)**。训练/部署特征集合完全一致（fork 仅用于 reward 与标注，不进特征），避免 train-test 信息泄漏攻击。

### 4.2 optimism-bias 诊断（不变，作用域扩大）

MiraBench 诊断三件套（Set C 上 \(d^{\text{pre}}\) 分布、Set A/B/C AUROC、DINOv2-only ablation）不变；v3.1 追加对 \(v_t^{\text{probe}}\) 做同样的三档分离度报告（probing 论文只在 LIBERO-Goal 验证，扰动域下的迁移性是我们的增量观察）。

### 4.3 自动降权机制（不变）

---

## 5. FEB 形式化：恒等式、覆盖范围、边界（v3.1 措辞降级，P7）

### 5.1 主命题（从"定理"降为"恒等式 + 引理"）

**FEB 恒等式**：若 \(m\) 是 candidate-set-internal 的（\(\forall s,\ m(s)\in\mathcal{A}(s)\)），则 \(\text{FEB}(m)\equiv 1\)。这是定义的直接推论——**我们在论文中明确承认这一点**，其价值不在数学深度，而在（a）给出一个此前无人度量的量的**度量协议**；（b）把"reranking 类方法在某类状态上结构性零贡献"从直觉变成可在 benchmark 上逐方法核算的账目。

**N-Scaling Lemma（数学重量所在）**：只要 \(\Pr_s[q(s)=0]>0\)，任何候选质量提升（policy-RL、鲁棒训练、test-time scaling、增大 \(K\)）都只能压缩、不能消灭 NGC 测度。给出形式化证明 + E4/E5 实证。

**可测量 FEB（v3.1 新增支柱）**：对 partial-fallback 方法（HELM、VoLo、CycleVLA、B2FF、FAR、AEGIS、RC-NF、KnowNo），FEB \(\in(0,1)\) 是经验量，反映其触发规则对 NGC 状态的覆盖率。E8 逐方法实测——这使 FEB 从"打击 reranker 的锤子"升级为"整个恢复方法谱系的统一坐标轴"，叙事更正面。

### 5.2 覆盖清单（v3.1 增补）

恒等式内（FEB \(\equiv 1\)）：rerankers（ADV/RoVer/SCALE/RoboMonkey/V-GPS/MG-Select）、Pre-VLA、VeriSpace、Do-What-You-Say、FOREWARN、**value-probe reranking（2605.28527 的在线用法）**。

恒等式外（可测量 FEB）：CycleVLA、B2FF、FAR、AEGIS、RC-NF、KnowNo、**HELM、VoLo、ReCoVLA、LiLo-VLA、FailSafe**。

learned-vs-rule/agent 论点（v3.1 版）：这些方法的触发是 rule / heuristic / VLM-deliberation，fallback 选型是固定的或分层的。§8 用（1）触发时机（FEB 偏高）、（2）选型质量（broken-success 偏高）、（3）\(\alpha_b\) sweep（联合优化）三个维度实证 learned selector 的优势；其中 **VoLo-lite 同空间对照**是最干净的证据。

---

## 6. RL 训练框架：工程设计（v3.1 两处修改）

### 6.1 动作空间（P11、P12 修订）

\[
\mathcal{U}(s) = \mathcal{A}(s)\ \cup\ \{\text{ROLLBACK},\ \text{RESAMPLE},\ \text{REPLAN},\ \text{WAIT},\ \text{ABSTAIN}\},\qquad |\mathcal{U}|\le 13.
\]

- **ROLLBACK 可达域声明**：主表语义为逆动作重放，仅对 \(\rho(s)=\text{reversible}\) 状态有语义保证；对 contact-irreversible 状态 ROLLBACK 退化为"后撤脱困"，其效果如实报告。快照式 set_state 仅出现在 oracle 上界。
- **REPLAN 双实现（P12）**：REPLAN-text（LLM 改写指令，即 v3.0 的 REPLAN-lite）与 **REPLAN-goal（B2FF 式：切换到 goal-image 条件或子目标锚定）**。主线 SmolVLA 上预期 text 版弱（LIBERO-Plus 的语言忽略发现），E9 双版本对照，主表用较优者并如实讨论。

### 6.2–6.6 selector 架构 / 训练配比 / 奖励 / infra / 稳定性预案

同 v3.0（3.5M、BC+CQL warm-start → Double+Dueling DQN、\(R = R_{\text{task}} - \alpha_b\mathbb{1}[\text{broken}] - \alpha_c\,\text{cost}\)、并行 env、发散与 fallback 崩溃预案）。工程细节全部移入《RL 训练框架搭建指南》。

### 6.7 matched-budget 记账规则（v3.1 新增，P10）

E3 的交互预算记账：

\[
B_{\text{selector}} = B_{\text{warm-start rollouts}} + B_{\text{online RL episodes}},\qquad B_{\text{policy-RL}} = B_{\text{SFT rollouts}} + B_{\text{RL episodes}}.
\]

selector 一侧把 NGC-Plus 标注中被 warm-start 复用的 oracle rollouts **全额计入** \(B_{\text{selector}}\)；benchmark 构建中未被训练复用的部分（纯评测集）不计。论文附记账表。预期结论不变（selector 参数量 3.5M vs 0.45B-LoRA，小预算区间仍显著占优），但公平性无懈可击。

---

## 7. 训练数据来源（微调）

| 数据 | 来源 | 规模 | 说明 |
|------|------|------|------|
| SmolVLA LIBERO checkpoint | HF `HuggingFaceVLA/smolvla_libero` | — | 固定 lerobot 版本 + `n_action_steps` |
| OpenVLA-OFT / OFT+ checkpoint | 官方仓库 / LIBERO-plus 仓库 | — | 副线 + 诊断轨 |
| 开源鲁棒化模型（VLA-GSE 或同级，若开源） | 官方 | — | E5 扩容用；不开源则退化为 OFT+ 单模型 |
| V-JEPA 2 ViT-L / DINOv2-B | 官方 | — | frozen |
| LIBERO / LIBERO-Plus 环境 | 官方 pip | 10,030 任务 | 状态生成器 |
| NGC-Plus 状态 + 三元标注 + 可逆性标签 | 全自采 | ~4,000 状态 | 本工作数据贡献 |
| selector RL 交互数据 | 全自采 | 10–16K episodes（含 warm-start 记账） | matched-budget |

---

## 8. 实验矩阵（v3.1 调整：E8 扩容、E5 扩容）

| 实验 | 缘由（堵哪个攻击） | 优势 | 缺陷/caveat |
|------|--------------------|------|------------|
| E1 主结果：SmolVLA + selector vs baselines（LIBERO-Plus 高难度子集） | 核心 gain | headroom 充足 | 弱 baseline 质疑 → E2 兜 |
| E2 OFT 7B 副线（camera/robot 维度 suite） | "base VLA 太弱" | 强模型复现 gain | 仅推理迁移 |
| E3 matched-budget 三方（含 §6.7 记账） | SimpleVLA-RL 质疑 + 预算公平性 | 立场之争→实证 | 单卡 LoRA 非原版复现（诚实声明） |
| E4 SimpleVLA-RL ckpt 在 NGC-Plus 测 FEB | 正交性 | 预期 FEB=1 | 需其开源 ckpt |
| E5（扩容）鲁棒化模型 NGC 产率 + FEB：OFT+ **及一个 80%+ 开源鲁棒化模型** | "鲁棒训练消灭 NGC"（VLA-GSE 81.2% 时代必须双模型） | 直接对冲叙事窗口收缩 | 开源可得性依赖 |
| E6 optimism-bias 诊断（\(d^{\text{pre}}\)、\(v^{\text{probe}}\) 双信号三档分离度） | MiraBench | 主动堵 | 分离差需承认 |
| E7 cross-oracle \(\kappa\) + 人工核查 + Wilson 置信度发布 | oracle 循环性 + 统计功效 | benchmark 质量背书 | 人工成本 |
| E8（扩容）decision-maker 对照：**VoLo-lite（同空间 agent）**、**HELM-lite（rollback+replan rule）**、B2FF-lite、CycleVLA-lite、FOREWARN-lite | HELM/VoLo/B2FF/CycleVLA/FOREWARN 五路近邻 | VoLo-lite 是同空间最干净对照 | lite 版非原版（逐一声明实现差异） |
| E9 per-fallback ablation（含 REPLAN-text vs REPLAN-goal、DINOv2-only、去 ACC、去 probe） | "5-fallback 是工程" + P12 | 最小覆盖论证 + 信号边际贡献 | — |
| E10 \(\alpha_b\) sweep | 统一空间联合优化 | rule/agent 分层做不到 | — |
| E11 cross-backbone（第三 backbone） | 部署性 | 即插即用 | 降级序最先砍 |

---

## 9. CVPR 2027 投稿策略与叙事

### 9.1 贡献叙事排序（不变）

headline C1；method C3（learned selector + FEB 协议 + 可测量 FEB 坐标轴）；C2 作组件 + 双信号诊断。

### 9.2 时间线（v3.1 压缩，P13）

- 2026-07 中～08：环境 + baseline 实测 + NGC-Plus 流水线跑通与成本重估（W1–W3）。
- 2026-08～09 中：NGC-Plus 全量采集标注 + 因果分析（W4–W9）。
- **2026-09 下旬：arXiv 预印本挂出**（benchmark + FEB 协议 + 初步 selector 结果即可挂，抢 NGC/FEB 命名权）。
- 2026-09 下～10：selector 训练 + E1/E5/E6（W10–W14）。
- 2026-10 下～11 初：E2/E3/E4/E8/E9/E10 + 写作（W15–W18）。
- **2026-11 上旬：CVPR 全文截稿**（以官网公布为准，按 11 月第一周末预留 buffer 规划；不要按"中旬"规划）。
- 投稿前一周：再做一轮同今日的文献扫描。

### 9.3 化敌为友定位（v3.1 追加）

LIBERO-Plus built-on-top；Foresight/B2FF 正面引用；**VoLo/HELM 定位为"统一恢复空间与 rollback+replan 概念的先行者，我们将其决策学习化并给出第一个 ground-truth 基准"**——把两个最危险近邻转化为 motivation 的一部分。

---

## 10. 风险矩阵与降级预案（v3.1 更新）

| 风险 | 概率 | 影响 | 预案 |
|------|------|------|------|
| NGC-Plus 标注延期 | 高 | headline 崩 | W2 实测耗时；超预算砍 Set B 保 Set C；自适应采样已控成本 |
| 叙事窗口收缩（鲁棒化模型 85%+ 出现） | 中高 | motivation 弱化 | 聚焦 L3+/组合扰动；E5 双模型实证"压缩≠消灭"；FEB 坐标轴叙事不依赖绝对塌缩幅度 |
| HELM/VoLo 型近邻再抢发（learned selector 版出现） | 中 | novelty 崩 | 9 月下旬预印本是唯一硬对冲；W1 起每两周文献扫描 |
| oracle 循环性被攻破 | 中 | benchmark 质量 | 三重防御 + Wilson 置信度发布 |
| \(d^{\text{pre}}\) optimism bias | 中 | C2 组件失效 | E6 + 四路信号冗余（ACC/probe 免费兜底） |
| SmolVLA 复现 gap | 中 | baseline 质疑 | 固定版本自测 + vla-eval harness 校验 |
| REPLAN-text 无效 | 中高 | fallback 缺口 | REPLAN-goal 备选已入设计 |
| 单卡算力不足 | 中 | 实验量 | 降级序 |

### 降级序（v3.1 微调）

不可裁剪核心：NGC-Plus（Set C-consensus + Wilson 置信度）+ FEB 协议 + E1 + E5/E6/E7 + **E8 中的 VoLo-lite 与 HELM-lite**（这两个是 v3.1 新增的 novelty 生命线）。

1. 砍 E11 第三 backbone。
2. E2 降为 L3+ 代表性子集推理评测。
3. E8 从五个 lite 降为三个：**保 VoLo-lite、HELM-lite、B2FF-lite**，砍 CycleVLA-lite、FOREWARN-lite（v3.0 的"保 B2FF-lite 一个"已不够——HELM/VoLo 威胁等级更高）。
4. Set B 规模砍半。
5. \(\tau\) 敏感性降为主设定 + 附录。

绝不降级：per-candidate ground-truth 标注协议、FEB 恒等式的精确覆盖表述、cross-oracle \(\kappa\)、**Wilson 置信度发布**、**§6.7 预算记账表**。

---

## 11. 18 周执行计划（2026-07-13 起算）

- W1–W3（7/13–8/2）：环境搭建（版本 pin）、SmolVLA/OFT baseline 自测、NGC-Plus 流水线跑通 + 单条 rollout 耗时实测 + 成本重估、文献扫描机制建立。
- W4–W9（8/3–9/13）：NGC-Plus 全量采集与标注（自适应采样 + E7 同步）、可逆性标注、因果分析。
- W10（9/14–9/20）：**预印本冲刺**：benchmark 章节 + FEB 协议 + 因果分析 + FEB 恒等式，9 月下旬挂 arXiv。
- W10–W13（9/14–10/11）：selector 训练（warm-start → online DQN）、E1、E6。
- W14–W16（10/12–11/1）：E2、E3（含记账表）、E4、E5、E8（VoLo-lite / HELM-lite 优先）。
- W17–W18（11/2–11/13）：E9/E10 补齐、写作、图表、附录复现配置、投稿（按官网实际截稿日倒排，预留 72h buffer）。

---

## 附录 A：v3.0 → v3.1 变更速查

| 位置 | v3.0 | v3.1 变更 |
|------|------|----------|
| 塌缩数字 | "OFT 级模型跌穿 30%" | 按维度精确引用（OFT 总体 69.6%，robot 31.9%，camera 56.4%） |
| SmolVLA 复现 | "43–51%" | 总体 ≈73%（Long 43–56%）；复现敏感性纪律（引 vla-eval） |
| 文献快照 | 2026-07-08 | 2026-07-10：补 HELM、VoLo、ReCoVLA、VLA-FAIL、RePO-VLA、AFIL、FailSafe、LiLo-VLA、FARL、LIBERO-PRO、value-probing、vla-eval、StarVLA-α/ABot-M0/VLANeXt |
| C3 分界 | 统一 fallback 空间 + learned selector | **learned RL selector** 为核心（统一空间概念归功 VoLo/orchestrator 线；rollback+replan 组合归功 HELM） |
| FEB | "定理" | "恒等式" + N-Scaling Lemma + partial-fallback 可测量 FEB 坐标轴 |
| Set C 判定 | 固定 \(N_{\text{roll}}=5\) | 两阶段自适应（3→10）+ Wilson 95% 上界判据 + per-state 置信度发布 |
| 风险信号 | \((\varphi, d, \sigma)\) 三元组 | 七路信号，\(d\) 拆 pre/post；新增 ACC 与 value probe |
| 动作空间 | REPLAN-lite（文字改写） | REPLAN-text + REPLAN-goal 双实现；ROLLBACK 可达域声明 + 物理可逆性标注 |
| matched-budget | 未计 warm-start | §6.7 显式记账 |
| E5 | OFT+ 单模型 | OFT+ + 一个 80%+ 开源鲁棒化模型 |
| E8 | CycleVLA/B2FF/FOREWARN 三 lite | 五 lite，新增 **VoLo-lite（同空间对照）**、**HELM-lite**；降级序保这两个 |
| 计划 | 22 周，预印本 9–10 月 | 18 周（7/13 起算），预印本 9 月下旬，截稿按 11 月上旬规划 |
