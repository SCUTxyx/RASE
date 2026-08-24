# RASE Idea 说明（2026-08-21 修订：Adaptive Cross-Policy Counterfactual Arbitration）

> 本文件是 RASE 方法的**不变定义**与最新定位。实验只决定"方法在哪些域可行、
> 在哪些域无空间"，不改变方法本身。
>
> **2026-08-21 重要修订**：竞争对手 BCP（北大+微软亚研，arXiv:2608.03483，
> 2026-08）已实现"自适应执行 horizon（continue-or-replan 学习）"，在
> LIBERO-PRO +6.8%、RoboTwin 2.0 +4.06%、真机 +11~40pp。**RASE 新定位：
> BCP 是 RASE 的一维特例；RASE 在其上增加跨策略反事实仲裁、same-root 校准
> 与统计保证，并保持 zero-shot 跨 VLA 与不依赖任务记忆。**

---

## 1. 一句话定义

RASE 是一个**轻量的、model-agnostic 的实时自适应仲裁层**：

> 在多个冻结 VLA 之上，基于**同一物理状态下多个候选动作**的反事实价值比较，
> 在动作块边界做自适应决策：**继续（continue，自适应 horizon）/ 重新生成
> （replan）/ 切换纠正候选（switch，跨策略）/ 安全中止（abort）**，全部由
> 同一**反事实校准**的风险信号驱动，并在**保守规则（统计保证）**下执行。

核心主张（不变）：**动作轨迹本身携带可学习的 outcome 风险信号**——不是依赖
policy 的置信度，而是直接比较"如果执行这个候选动作会发生什么"。

**设计目标（必须满足才算是 RASE）**：
1. **zero-shot 到多个 VLA**：仲裁 head 不编码候选身份，未见 VLA 的候选动作
   可直接打分（只依赖候选动作/状态的可观测特征）；
2. **风险驱动决策**：continue / replan / switch / abort 由风险预测结果决定，
   不依赖"记住任务谁强"的查询表；
3. **跨平台**：特征规范化（相对动作、通用状态表示），不绑定单一仿真平台。

---

## 2. 系统组件与决策空间（2026-08-21 扩展）

```text
source VLA proposal + task + 近期观察/动作
                    │
                    ▼
    轻量统一仲裁头（continuation + arbitration core）
    （反事实校准：same-root 标签训练；canonical 特征）
                    │
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
   continue     replan           switch          abort
（自适应 horizon）（同策略重生成）（跨策略纠正候选）（保形校准中止）
```

| 决策 | 语义 | 状态 |
|---|---|---|
| continue | 继续执行当前 chunk（何时截断 = 自适应 horizon） | **BCP 层（基线组件，2026-08 起吸收）** |
| replan | 同策略重新生成候选 | 同上（自适应触发的重采样） |
| switch | 切换到跨策略纠正候选（fallback.persistent） | **核心差异化层（BCP 没有）** |
| abort | 安全中止 | **保形校准的决策级保证（BCP 没有）** |

| 组件 | 职责 | 关键设计 |
|---|---|---|
| 候选动作 provider | 同一边界生成多个候选 | `continue.source` / `requery.source` / `resample.source` / `fallback.persistent` / `abort.safe` |
| 同步冻结(capture) | 候选不可变、可审计保存 | inference-event provenance + queue cursor + seed ledger + hash |
| 同根分叉执行 | 候选在相同物理状态执行 | simulator restore,matched execution seeds |
| 统一仲裁头 | 预测 continue/replan/switch 的相对价值 | 轻量头（<1M 参数），same-root 反事实标签（监督+校准），RL 仅作补充 |
| 保守仲裁 | 只在有把握时干预 | 保形校准阈值（false-replan/false-switch 上界）；默认 continue |
| 概率标签 | 小 K 下不硬编码确定性 | K≥3 软成功概率 + 置信权重；capability mask 分层 |

---

## 3. 与 BCP（arXiv:2608.03483）的差异化矩阵

| 维度 | BCP | RASE-ACCA | 差异性质 |
|---|---|---|---|
| 决策空间 | continue / replan（二值，同策略） | + switch（跨策略纠正臂）+ abort（保形） | **决策空间扩展** |
| 训练信号 | trajectory-level RL（最优 horizon 不可观测） | same-root 反事实标签（同状态多候选→真实结局，密集可观测） | **监督升级：反事实校准** |
| 校准/保证 | 无统计保证（efficiency reward 抑制过度重规划） | 保形/决策级校准（false-replan/false-switch 上界） | **统计保证** |
| 跨 VLA | per-policy head（LingBot/π0.5 分别训练） | canonical 特征 + 无 policy identity → zero-shot 未见 VLA | **泛化主张** |
| 数据协议 | 标准 rollout 标签 | same-root 冻结状态 × 多候选（文献确认无先例） | **数据协议贡献** |
| 验证域 | RoboTwin 2.0、LIBERO、LIBERO-PRO、真机 | 复用上述 + 跨策略臂评估 | 继承 + 扩展 |

**定位声明**：BCP 验证了"自适应执行 horizon 在扰动域有增益"（LIBERO-PRO +6.8%）。
RASE 将其吸收为 continue/replan 层（一维特例），主贡献落在 BCP 未覆盖的：
**跨策略反事实仲裁（switch）、same-root 校准（监督+保证）、zero-shot 跨 VLA**。

---

## 4. 决策时点与候选语义（不变）

边界（boundary / decision point）：source 策略执行到预注册步数后、执行下一
动作前。候选语义（严格区分，不得混用）：

- `continue.source`：当前 VLA 已生成动作块的**继续执行**（可自适应截断）；
- `requery.source`：同一边界、同一 VLA 的**重新推理**（独立 seed）；
- `resample.source/candidate.{0,1}`：同一次推理的两个独立采样候选；
- `fallback.persistent`：纠正策略（如 OFT/π0-fast）的完整动作块，持续接管；
- `abort.safe`：安全中止（control event，不是零动作伪装）。

**capability 规则**：候选不可执行/无多样性 → 记录确定性 capability mask
（`incapable_*` / `control_only_abort`），绝不当作普通失败、绝不从分母剔除。

**候选生成机制事实（代码确认）**：
- flow matching 系（SmolVLA/π0/π0.5）：默认动作 = 单次噪声采样（无 greedy
  等价物）；K 候选 = 批量噪声（共享 prefix KV cache）；
- 自回归系（π0-FAST）：默认 = greedy（temperature=0）；候选 = temperature
  0.4–0.7 采样（注意：`predict_action_chunk` 读取 `config.temperature` 而非 kwargs）；
- 确定性回归系（OpenVLA-OFT）：无分布，只能做 switch 臂。

---

## 5. 方法原则（不可动摇，增补 2 条）

1. **可学习性 ≠ 可部署性**：部署价值必须由"相对最强固定策略的增益"证明。
2. **成本只在部署层**：训练标签用 Δsuccess；query/replan/switch 成本只进入
   部署效用 `U_λ = success − λ·cost`，不污染动作信号。
3. **sunk vs incremental**：决策点之前成本是沉没成本，不得惩罚任何候选。
4. **禁止物理回滚**：simulator restore 仅是监督用途。
5. **无免费午餐**：纠正策略若免费且处处最优，仲裁冗余（域边界，非方法否定）。
6. **任务记忆 ≠ 选择器**：禁止 task→best-policy 查询表；决策必须风险驱动。
7. **（增补）域不确定性优先**：先验证域存在执行/观测不确定性（候选结局分叉、
   oracle@K > best-of-1），再投入仲裁训练——clean 确定性域直接跳过。
8. **（增补）反事实校准优于纯 RL**：same-root 反事实标签提供密集可观测监督；
   RL（BCP 式）只作补充（长程/不可观测 horizon 的兜底）。

---

## 6. 已吸收的历史负证据（方法约束，不改变 idea）

| 结果 | 教训 |
|---|---|
| R3 | 成功条件选择不足；必须包含失败样本 |
| R6-C / R7-A | 单 root 支持不足；task-held-out、多 seed、bootstrap 下界是底线 |
| R9-C | 无条件 hazard 标签太稀疏；改用条件化、平衡标签 |
| R10-B | 闭环发散 → 小 K 硬标签不可靠；概率标签 + matched seeds |
| Deployment-v2 | fallback 统治域无 headroom；必须先做 domain mining + Opportunity Gate |
| π0.5 challenge | 高成功率 policy 上信号不可观测（ceiling） |
| Dynamic route | 确定性 policy 无 candidate diversity → disagreement detector 结构性不可用 |
| libero_90 | source 全败域无 outcome 差异 → 单纯增加难度不创造 selector opportunity |
| OFT 机会窗口 | 异构模型对存在教科书级互补（100%/0%），headroom +50pp——Goldilocks 实证 |
| OPD 蒸馏 | 风险预测可学且校准，但同状态早期决策点无适配区分；查询表 99.2% 是记忆 |
| v3 Gate A | 冻结 v3 对 π0-fast 0.837（跨架构信号存在）；VLA-ID probe 0.894（指纹泄漏） |
| v3 Gate B/C | 同状态候选未来分叉、真实未来可 0.9998 排序——反事实协议成立 |
| v3 Gate D / W1 / W2 | WM 摘要无增益；learned 未来两次失败——WM 路线搁置 |
| 闭环 v2 | 风险驱动闭环 = best-fixed（0 切换全 abstain）——静态特征无法区分同状态候选 |
| **G1（08-20）** | SmolVLA 多尺度噪声重生成：1/8 rescue，主结构 = task difficulty |
| **G2a/b（08-20）** | π0-fast Long 86.25%（过强）；Spatial 16 对 SmolVLA 弱支配（0 continue-only） |
| **E3/E3-B（08-20/21）** | BC 残差：离线可学（MSE 改善 37%）但闭环无 residual-only；DAgger 修好分布偏移后仍 0 candidate-only（教师天花板 + 不可救状态） |
| **E4-0（08-21）** | π0-fast T=0.7 候选池：oracle@8 = best-of-1（0/24 rescue）——生成侧多样性不产生结局级互补 |
| **R0（08-21）** | 执行期偏差检测：proprio 演化 99.3% 可预测、偏差与结局无关——确定性仿真杀死验证类方法 |
| **R0b（08-21）** | 噪声下重规划频率：k4 50% vs k10 67%——固定频率重规划无增益（BCP 的前提验证）；噪声本身有探索价值（t01-i00 无噪 0/8 → 带噪成功） |
| **BCP 对照（08-21）** | 学习自适应 horizon（非固定频率）在扰动域有增益：LIBERO +1.7%、**LIBERO-PRO +6.8%**、RoboTwin +4.06%、真机 +11~40pp |

---

## 7. 核心科学假设（2026-08-21 更新）

- **H1（不变）**：`P(s_{t+1:t+H} | s_t, a, π) ≈ P(s_{t+1:t+H} | s_t, a)`——
  Action Consequence 是 cross-VLA transferable bottleneck。
- **H2（新增，实验支撑）**：**扰动/不确定性是仲裁价值出现的必要条件**。
  - clean 确定性域：候选结局无分叉（G1/E4-0）、偏差不可检测（R0）、固定
    重规划频率无增益（R0b）；
  - 扰动域（LIBERO-PRO / RoboTwin Randomized / 真机）：自适应 horizon 有
    增益（BCP）——**候选结局在扰动域分叉，仲裁才有信息**。
- **H3（新增）**：continue/replan/switch 共享同一反事实价值信号——统一仲裁
  head 优于 BCP 的单一 replan 头（跨决策一致性 + 参数共享 + 反事实校准）。

---

## 8. 经验识别的适用条件（2026-08-21 更新）

### 8.1 必要条件（缺一不可）

```text
1. 域存在执行/观测不确定性（扰动、噪声、随机初始化）      ← H2（新）
2. 候选结局分叉：oracle@K > best-of-1 且 rescue ≥2         ← 候选池 gate（新）
3. 0 < P(source success | x) < 1                           ← E1
4. 决策边界存在 comparative-advantage heterogeneity       ← H_within（扰动域重测）
5. candidate provider 非退化
```

### 8.2 四条失败机制（实验确认，论文贡献）

```text
Diversity = 0            → 同策略候选不可分（确定性策略）
Outcome variability = 0  → clean 确定性域候选结局无分叉（oracle gain=0）
Comparative advantage = 0 → 嵌套失败模式（Experts-in-an-MDP 定理）
Uncertainty = 0          → 验证类方法无信号（偏差恒≈0）← 新（R0/R0b）
```

### 8.3 重开条件（写死）

任何新域/新配置必须通过 Pre-RASE Eligibility Screen：
**E0'（候选池 oracle gate：oracle@K > best-of-1 且 rescue ≥2）→ E1（source
10%–90%）→ E2（H_within ≥5% 且 oracle gain ≥5pp）**。
**优先域**：LIBERO-PRO（物体位置扰动）、RoboTwin 2.0 Randomized、真机。

### 8.4 验证状态（2026-08-21 更新）

| 主张 | 状态 | 证据 |
|---|---|---|
| same-root 反事实协议 | ✅ PASS | P0 审计 + Gate B/C |
| 风险预测 OFT 架构内转移 | ✅ PASS | LOVO 0.94+ |
| 风险预测跨架构转移 | ⚠️ 未确证 | π0-fast 0.837（高分域） |
| clean 确定性域仲裁无空间 | ✅ PASS | 闭环 v2 / G1 / E4-0 / R0 / R0b 五环一致 |
| **扰动域自适应 horizon 有增益** | ✅ **外部证据** | **BCP：LIBERO-PRO +6.8%**（待我们复现） |
| **扰动域跨策略仲裁（我们的主贡献）** | ⏳ 未验证 | R2 计划 |
| 身份无关（无指纹） | ⚠️ 未达成 | VLA-ID probe 0.894 |
| 跨平台 | ⏳ 未验证 | 特征含 LIBERO 特定量 |

---

## 9. 目标终点（不变）

**主 endpoint：closed-loop success improvement**（相对 best-fixed /
always-fallback / **BCP 基线**，paired task-bootstrap 95% CI 下界 > 0 且绝对
增益 ≥ 3pp）。

成本是约束与次指标；论文主线是"成功率提升"，不是"省调用"。
**前置条件（用户明确）**：提升必须由风险预测驱动（zero-shot 到未见 VLA/任务），
不是任务记忆；并尽可能跨平台成立。

## 10. 论文定位（2026-08-21）

> **"Adaptive Cross-Policy Counterfactual Arbitration"**：用 same-root 反事实
> 数据训练的统一仲裁层（continue/replan/switch/abort），在扰动域
> （LIBERO-PRO / RoboTwin Randomized / 真机）上相对 BCP 与固定策略取得可证明
> 增益，并 zero-shot 到未见 VLA。

- 贡献 1：same-root 反事实数据协议（文献空白）；
- 贡献 2：统一仲裁 head（自适应 horizon + 跨策略臂，BCP 的严格超集）；
- 贡献 3：反事实校准 + 保形统计保证（BCP 无）；
- 贡献 4：zero-shot 跨 VLA（canonical 特征，无 policy identity）；
- 贡献 5（负向）："何时仲裁不可能"的系统刻画（九环证据链 + 定理锚点）。
