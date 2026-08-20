# RASE Idea 说明(不变:风险预测 + 回退 Selector)

> 本文件是 RASE 方法的**不变定义**。它不被任何实验负结果修改;实验只决定
> "方法在哪些域可行、在哪些域无空间",不改变方法本身。
> 对应 canonical 文档:`outputs/RASE_CANONICAL_IDEA_stochastic_multi_vla_risk_control_2026-08-13.md`。

---

## 1. 一句话定义

RASE 是一个**轻量的、model-agnostic 的实时风险控制层**:

> 在多个冻结 VLA 之上,基于**同一物理状态下多个候选动作**的反事实价值比较,
> 预测"继续当前动作"的风险与"切换纠正策略"的恢复概率,并在保守规则下
> 选择继续 / 重新采样 / 回退纠正 / 安全中止。

核心主张:**动作轨迹本身携带可学习的 outcome 风险信号**——不是只依赖 policy
的置信度,而是直接比较"如果执行这个候选动作会发生什么"。

**设计目标(必须满足才算是 RASE)**:
1. **zero-shot 到多个 VLA**:风险预测器不编码候选身份,新 VLA 的候选动作
   可直接打分(只依赖候选动作/状态的可观测特征);
2. **风险驱动决策**:继续(continue)/ 回退(fallback)/ 更换候选(switch)由
   风险预测结果决定,不依赖"记住任务谁强"的查询表;
3. **跨平台**:特征规范化(相对动作、通用状态表示),不绑定单一仿真平台。

---

## 2. 系统组件(冻结)

```text
source VLA proposal + task + 近期观察/动作
                    │
                    ▼
   轻量 policy-conditioned 短视界风险模型
   (shared causal history / action risk core)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  two-boundary safe dwell     persistent corrective takeover
  (继续 source)              (回退纠正,可恢复窗口内)
        │                       │
        ▼                       ▼
  continue.source          fallback.persistent
```

| 组件 | 职责 | 关键设计 |
|---|---|---|
| 候选动作 provider | 同一边界生成多个候选 | `continue.source` / `requery.source` / `resample.source` / `fallback.persistent` / `abort.safe` |
| 同步冻结(capture) | 候选不可变、可审计保存 | inference-event provenance + queue cursor + seed ledger + hash |
| 同根分叉执行 | 候选在相同物理状态执行 | simulator restore,matched execution seeds |
| 风险预测 | 预测每个候选的成功/恢复概率 | 低容量模型(ridge/logistic),same-root pairwise Δsuccess |
| 保守切换 | 只在有把握时切换 | margin-based abstain,默认继续;回退训练集内 best-fixed |
| 概率标签 | 小 K 下不硬编码确定性 | K≥3 软成功概率 + 置信权重;capability mask 分层 |

---

## 3. 决策时点与候选语义

边界(boundary / decision point)定义:source 策略执行到预注册步数
(`source.step.8` / `source.step.16`,或"临界前"动态点)后、执行下一动作前。

候选语义(严格区分,不得混用):

- `continue.source`:当前 VLA 已生成动作块的**继续执行**;
- `requery.source`:同一边界、同一 VLA 的**重新推理**(独立 seed);
- `resample.source/candidate.{0,1}`:同一次推理的两个独立采样候选;
- `fallback.persistent`:纠正策略(如 OFT)的完整动作块,持续接管;
- `abort.safe`:安全中止(control event,不是零动作伪装)。

**capability 规则**:候选不可执行/无多样性 → 记录确定性 capability mask
(`incapable_*` / `control_only_abort`),绝不当作普通失败、绝不从分母剔除。

---

## 4. 方法原则(不可动摇)

1. **可学习性 ≠ 可部署性**:动作信号可预测,不等于选择器有部署价值;
   部署价值必须由"相对最强固定策略(best-fixed/always-fallback)的增益"证明。
2. **成本只在部署层**:训练标签用 Δsuccess;query/fallback/latency 成本只进入
   部署效用 `U_λ = success − λ·cost`,不污染动作信号。
3. **sunk vs incremental**:决策点之前已发生的 source 成本是沉没成本,
   不得惩罚任何候选;只比较决策点起新增成本。
4. **禁止物理回滚**:不安全的候选可在执行前拒绝;simulator restore 仅是监督用途。
5. **无免费午餐声明**:OFT 等纠正策略若免费可用且近乎完美,选择器冗余——
   这正是实验发现的域边界,不构成方法否定。
6. **任务记忆 ≠ 选择器**:用"训练期查询表记住每个任务谁强"不是 RASE;
   RASE 的决策必须由风险预测驱动,并可迁移到未见任务/VLA。

---

## 5. 与"语义扰动"实验(D0)的区分

| 层 | 内容 | 用途 |
|---|---|---|
| selector-level provider(K3) | continue/requery/resample/fallback/abort | 候选来源选择,学习动作→结局 |
| semantic perturbation(D0) | translation/rotation sign flip、temporal reverse、gripper shift | 可行性诊断,证明动作变化能改变结局 |

两者不得混入同一实验层级,否则无法归因"来源差异"与"变换效果"。

---

## 6. 已吸收的历史负证据(方法约束,不改变 idea)

| 结果 | 教训 |
|---|---|
| R3 | 成功条件选择不足;必须包含失败样本 |
| R6-C / R7-A | 单 root 支持不足;task-held-out、多 seed、bootstrap 下界是底线 |
| R9-C | 无条件 hazard 标签太稀疏;改用条件化、平衡标签 |
| R10-B | 闭环发散 → 小 K 硬标签不可靠;概率标签 + matched seeds |
| Deployment-v2 | fallback 统治域无 headroom;必须先做 domain mining + Opportunity Gate |
| π0.5 challenge | 高成功率 policy 上信号不可观测(ceiling);信号是 policy-dependent |
| Dynamic route | 确定性 policy(temperature=0)无 candidate diversity → disagreement detector 结构性不可用 |
| libero_90 | source 全败域无 outcome 差异 → 单纯增加难度不创造 selector opportunity |
| OFT 机会窗口 | 异构模型对(domain-specific OFT)存在**教科书级互补**(100%/0%),headroom +50pp——Goldilocks regime 首次实证 |
| OPD 蒸馏(2026-08-19) | 风险预测可学且校准(0.7→97.6%),但**同状态早期决策点无适配区分**;任务级查询表 99.2% 是"记忆"不是"风险驱动"——明确为偏离 idea 的中间产物 |
| v3 零样本(Gate A,08-19) | 冻结 v3 对 π0-fast(异构)预测 0.837/0.07(跨架构信号存在);VLA-ID probe 0.894(指纹泄漏);goal/oft10 近全败域不可测 |
| v3 same-root + oracle(Gate B/C) | 同状态候选未来确实分叉(物体位姿 0.0795);真实未来可 0.9998 排序候选——反事实协议与信息上限成立 |
| v3 WM(Gate D/W1/W2) | 摘要未来瓶颈无增益(B2≈B1);真实未来仅 +3pp(2/3 折);**learned 未来两次改进均失败**(排序信号对预测误差极敏感)→ WM 视频路线在当前域不划算 |
| **闭环 v2(08-20)** | **风险驱动闭环 49.2%(=best-fixed)**:同状态双候选的早期动作 chunk 特征无法区分适配(训练/部署分布 gap)→ 0 切换全 abstain——**风险驱动选择在 LIBERO 同状态场景结构性失败** |
| **架构内转移限定(08-20)** | LOVO 0.94+ 全部为 **OFT 架构内**(同骨架不同微调域);跨架构唯一证据是 π0-fast 0.837(高分域,信号可能部分来自成功率水平)——**跨架构转移未确证** |

---

## 7. 经验识别的适用条件(Empirically Identified Applicability Conditions)

> **方法定义 ≠ 方法适用条件。** 本文件的 1-6 节是方法定义(不变);
> 本节是实验链之后**经验识别**的适用条件,记录方法何时成立。

### 7.1 必要条件(缺一不可)

```text
1. 0 < P(source success | x) < 1        source outcome 不饱和(ceiling/floor 均不行)
2. P(argmax_a V(a|x) != fallback) > 0   候选间存在比较优势异质(fallback 不处处统治)
3. candidate provider 具有真实 action diversity(非确定性退化)
4. corrective fallback 不近乎处处最优(否则无 deployment headroom)
5. decision boundary 上存在足够 comparative-advantage heterogeneity(≥5%)
```

核心条件可概括为:

```text
0 < P(source success | x) < 1
P(argmax_a V(a|x) != fallback) > 0
candidate action distribution 非退化
```

### 7.2 最终定位

> **RASE 不是一个"所有机器人状态上的 universal selector",而是一个针对
> comparative-advantage heterogeneous regimes(Goldilocks regime)的实时
> 反事实选择层。**

三条失败机制(经验确认):

```text
Diversity = 0           → disagreement detector 不可能(π0-fast)
Outcome variability = 0 → 学习目标消失(ceiling:π0.5;floor:libero_90)
Comparative advantage = 0 → deployment headroom 消失(fallback 统治:LIBERO)
```

### 7.3 重开条件(写死,防止无目的扩展)

未来任何新 policy/domain 必须**先通过 Pre-RASE Eligibility Screen**
(E0 candidate capability → E1 source competence 10%-90% → E2 opportunity
hetero ≥5% / oracle headroom ≥5pp),才允许进入完整 RASE 流程。三类可重开
trigger:①真正 stochastic 的 source(temperature>0 / native diversity);
②intermediate-difficulty policy(如 π0-fast 微调 libero_90,source 30%-70%);
③fallback 不再近乎完美教师(真实机器人:昂贵/慢/异质失败模式)。

### 7.4 当前验证状态(2026-08-20 更新)

| 主张 | 状态 | 证据 |
|---|---|---|
| Goldilocks regime 真实存在 | ✅ PASS | OFT 模型对:互补 100%/0%,oracle headroom +50pp,异质率 100% |
| 风险预测可学 | ✅ PASS | OPD 蒸馏 ridge:校准 0.7→97.6%;same-root B1 AUROC 0.98 |
| **风险预测 OFT 架构内跨 VLA 转移** | ✅ PASS | LOVO(留出 OFT 家族成员)unseen AUROC 0.94+,retention 0.90+ |
| **风险预测跨架构转移(π0/SmolVLA)** | ⚠️ **未确证** | 唯一证据:π0-fast 0.837(Gate A,高分域);OFT→异构未用 same-root 协议验证 |
| **风险驱动闭环选择** | ❌ **FAIL** | 闭环 v2:49.2%(=best-fixed),0 切换全 abstain——同状态双候选早期 chunk 特征无法区分适配(训练/部署分布 gap) |
| 任务级比较优势可捕获(记忆) | ✅ PASS(非 idea) | 查询表闭环 99.2%;但这是"记忆",非风险驱动 |
| 真实未来信息价值 | ✅ PASS(温和) | oracle 0.9998;真实未来 2/3 折 +3.4~3.8pp(W1) |
| learned 未来(WM)捕获增益 | ❌ FAIL | W2 两次改进均失败(预测误差破坏排序信号);增量上限 +3pp 配不上成本 |
| 任务级文本泛化(未见任务) | ❌ FAIL | bigram/embedding LOO ~60%;此前 93.5% 系特征标准化 bug 假象 |
| 身份无关(无指纹) | ⚠️ 未达成 | VLA-ID probe 0.894(随机 0.33) |
| **跨平台** | ⏳ 未验证 | 特征含 LIBERO 特定量 |

### 7.5 核心经验教训(2026-08-20,约束后续方向)

```text
1. 反事实协议成立,但"风险驱动选择"需要能区分【同状态双候选】的特征——
   当前 proprio+chunk 统计做不到(训练/部署分布 gap 是结构性的);
   候选:视觉嵌入(图像含任务信息)、决策粒度改任务级、或更难/更长任务域。
2. 跨架构转移证据不足:LOVO 0.94+ 是 OFT 架构内;跨架构只有 π0-fast 一个
   证据(高分域);OFT→π0/SmolVLA 的 same-root 验证是必须补的实验。
3. WM 未来增强在当前域不划算:增量上限 +3pp、learned 无法保留、视频成本高;
   除非换"B1 基线低 + 长 horizon + 接触丰富"的域,否则不再投入 WM。
4. 查询表 99.2% 证明"记忆"可行但非 RASE;任何新闭环必须先过
   "非记忆对照"(task-router 基线)。
```

---

## 8. 目标终点(不变)

**主 endpoint:closed-loop success improvement**(相对 best-fixed / always-fallback,
paired task-bootstrap 95% CI 下界 > 0 且绝对增益 ≥ 3pp)。

成本是约束与次指标;论文主线是"成功率提升",不是"省 fallback 调用"。
**前置条件(用户明确)**:提升必须由风险预测驱动(zero-shot 到未见 VLA/任务),
不是任务记忆;并尽可能跨平台成立。
