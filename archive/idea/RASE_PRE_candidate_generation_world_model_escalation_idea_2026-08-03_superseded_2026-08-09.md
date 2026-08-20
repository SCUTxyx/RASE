# RASE-PRE：基于世界模型的候选动作生成、风险评估、排序与分层恢复方案

**版本日期：** 2026-08-03  
**工作名称：** RASE-PRE  
**英文全称：** **Risk-Aware Action Proposal, Ranking, and Escalation for Frozen Vision-Language-Action Policies**  
**中文定位：** 面向冻结 VLA 的风险感知候选动作生成、世界模型证据评估与分层恢复系统  
**目标论文形态：** Same-State Benchmark + Candidate-Conditioned Method + World-Model Audit + Cross-VLA Generalization + Real-Robot Validation

---

# 0. 一句话结论

RASE 不再把主要研究问题定义为：

> 在 immediate OFT 与 deferred OFT 之间选择切换时间。

新的核心问题是：

> **当冻结 VLA 提出的当前动作可能导致失败时，系统能否先判断候选动作的风险与价值，在同策略重采样、同策略重规划、其他策略候选和局部纠正中找到可靠替代；若当前状态下所有候选都不可靠，再执行回退、稳定化、切换强策略或请求人工。**

系统的基本原则是：

> **先在当前状态的候选空间中寻找低侵入的可行尝试；只有当前候选空间无解时，才改变执行策略或物理状态。**

因此，新的方法主线是：

```text
Current VLA proposal
        ↓
Candidate-conditioned risk/value evaluation
        ↓
Safe and useful?
   ├── Yes → Execute
   └── No
        ↓
Generate same-state alternatives
        ↓
Resample / Replan / Fallback / Local correction
        ↓
Any reliable candidate?
   ├── Yes → Execute best candidate
   └── No
        ↓
State-changing recovery
        ↓
Retreat / Stabilize / Rewind / Human escalation
        ↓
Observe new state and regenerate all candidates
```

---

# 1. Idea 如何从原路线演化

## 1.1 最初的真实想法

最初的研究设想并不是单纯的 timing selector，而是一个完整闭环：

```text
预测当前动作风险
→ 判断当前 action chunk 是否值得执行
→ 若有风险，生成多个替代候选
→ 预测每个候选的短时后果与最终价值
→ 选择最佳可执行候选
→ 若当前状态的所有候选都不可靠，再升级到恢复或回退
```

其中世界模型负责预测候选动作的未来证据，selector 负责选择：

- 原动作；
- 重采样动作；
- 重新规划动作；
- fallback VLA 动作；
- 局部纠正；
- 回退或停止。

## 1.2 Phase 0 实际验证了什么

Phase 0 主要验证的是一个更窄的子问题：

```text
已经准备使用 OFT
→ immediate OFT
vs
→ deferred OFT
```

实验表明：

- immediate OFT 已经是强固定策略；
- timing oracle 相对最佳固定策略的提升不足；
- timing disagreement 缺乏统一、单调、共享机制；
- learned timing selector 没有足够的 oracle headroom。

因此被关闭的是：

> **learned timing selector**

而不是：

> **动作级风险评估、候选动作选择和分层恢复**

## 1.3 新路线与 timing selector 的本质区别

Timing selector 比较的是两条高度相似的轨迹：

\[
\text{OFT now}
\quad \text{vs} \quad
\text{OFT after a short delay}
\]

新的候选系统比较的是行为差异更大的候选：

\[
\begin{aligned}
&\text{current source chunk},\\
&\text{same-policy resamples},\\
&\text{fresh same-policy replan},\\
&\text{alternate-policy proposal},\\
&\text{local correction primitive},\\
&\text{state-changing recovery}.
\end{aligned}
\]

因此新的机会空间来自：

1. 单次采样偶然性；
2. action chunk 过时；
3. 不同 VLA 的能力互补；
4. 局部接触和几何修正；
5. 当前状态已离开策略支持集，需要先改变状态。

---

# 2. 新的核心科学问题

## Q1：当前动作失败，是生成问题还是评价问题？

给定当前候选 \(a_t^0\)，失败可能来自：

- VLA 的条件分布根本没有可行动作；
- 条件分布中有好动作，但当前采样不好；
- 生成了好动作，但系统没有识别出来；
- 当前 action chunk 已经因新观察或接触变化而过时；
- 当前物理状态已经不可由原策略直接恢复。

因此需要区分：

\[
\text{proposal failure}
\neq
\text{evaluation failure}
\neq
\text{state-support failure}.
\]

## Q2：同一 VLA 分布中是否存在可利用的候选多样性？

定义严格同分布重采样：

\[
a_t^{(k)}
\sim
\pi_s(a\mid h_t,l;\xi_k),
\]

其中除采样随机种子 \(\xi_k\) 外，其他条件全部保持一致。

需要测量：

\[
H_{\text{sampling}}
=
S_{\text{oracle-resample}}
-
S_{\text{original}}.
\]

如果该差值很小，则训练复杂 candidate selector 没有意义。

## Q3：重新条件化是否比单纯重采样更重要？

Fresh replan 使用最新 observation、proprioception 和 history：

\[
a_t^{replan}
\sim
\pi_s(a\mid h_t^{fresh},l).
\]

它与严格 resample 不属于同一条件分布。

需要测量：

\[
H_{\text{reconditioning}}
=
S_{\text{oracle-resample+replan}}
-
S_{\text{oracle-resample}}.
\]

## Q4：不同 VLA 是否具有真正的候选互补性？

给定 fallback VLA：

\[
a_t^{fallback}
\sim
\pi_f(a\mid h_t^{pub},l),
\]

测量：

\[
H_{\text{executor}}
=
S_{\text{oracle-all-policies}}
-
S_{\text{oracle-source-only}}.
\]

该结果回答：

> alternate VLA 带来的是新的能力支持，还是仅仅更多计算？

## Q5：什么时候应该停止继续挑候选，转而改变物理状态？

如果当前状态下所有候选都失败，则需要区分：

- 候选生成不足；
- 当前状态已不可恢复；
- 需要 retreat、stabilize、rewind；
- 所有自动操作都无支持，应 abstain。

定义：

\[
H_{\text{state-restoration}}
=
S_{\text{oracle-after-recovery}}
-
S_{\text{oracle-same-state}}.
\]

## Q6：世界模型是否提供真实增量？

核心比较：

\[
V(h_t,c)
\quad \text{vs} \quad
V(h_t,c,\hat Z_c^{WM},\sigma_c^{WM}).
\]

世界模型只有在以下条件下才成为主贡献：

- 超过 history-only candidate critic；
- 在 held-out task、perturbation 或 VLA 上仍有增量；
- imagined ranking 与真实 continuation ranking 稳定相关；
- 增益足以覆盖额外计算和延迟；
- 能识别自身 OOD 与错误自信。

---

# 3. “同一分布”必须严格定义

“候选是否来自同一分布”不能只看是否使用同一个模型。

至少要区分三层：

## 3.1 同一个策略模型

候选是否由同一个冻结 VLA 生成。

例如：

```text
Source VLA sample 1
Source VLA sample 2
Source VLA replan
```

这三者都来自同一个模型，但不一定来自同一个条件分布。

## 3.2 同一个条件动作分布

严格同分布要求：

- 同一个物理状态；
- 同一个 RGB / depth；
- 同一个 proprioception；
- 同一个历史窗口；
- 同一个语言指令；
- 同一个 policy cache 初始化；
- 同一个 sampling temperature；
- 同一个 top-p / top-k；
- 同一个 diffusion schedule；
- 同一个 action horizon；
- 只改变随机 seed。

此时：

\[
a_t^{(1)},\ldots,a_t^{(K)}
\overset{i.i.d.}{\sim}
\pi_s(a\mid h_t,l).
\]

## 3.3 同一个物理状态

即使策略相同，只要执行了 retreat、stabilize 或 rewind，新的候选就来自新状态：

\[
h_t
\xrightarrow{u_{\text{recovery}}}
h_{t'}.
\]

然后：

\[
a_{t'}^{(k)}
\sim
\pi(a\mid h_{t'}^{pub},l).
\]

因此回退后绝不应继续声称候选来自原来的动作条件分布。

---

# 4. 候选生成的完整分层设计

## 4.1 候选不是单一动作数组

定义一个 candidate proposal：

\[
c=
(
u,
e,
a_{t:t+L},
g,
H,
B,
M
),
\]

其中：

- \(u\)：operator family；
- \(e\)：executor；
- \(a_{t:t+L}\)：具体可执行 action sequence；
- \(g\)：目标状态、目标 keyframe 或恢复目标；
- \(H\)：候选执行 horizon；
- \(B\)：计算、时间和安全预算；
- \(M\)：生成参数与元数据。

Selector 学习的是：

\[
V_\theta(h_t,c),
\]

而不是只根据状态输出一个固定标签。

---

## 4.2 C0：Current suffix

当前 source VLA 已经生成但尚未执行完的 action suffix：

\[
c_t^{current}.
\]

用途：

- 默认候选；
- 作为风险评估对象；
- 所有干预的参照；
- 测量 false intervention harm。

必须记录：

- suffix 长度；
- action chunk 的生成时间；
- 已执行前缀；
- 剩余可取消部分；
- action latency；
- policy seed。

---

## 4.3 C1：严格同分布 resample

从完全相同条件重新采样：

\[
\mathcal C_t^{resample}
=
\left\{
c_t^{(1)},\ldots,c_t^{(K)}
\right\}.
\]

唯一变化是：

\[
\xi_1,\ldots,\xi_K.
\]

它回答：

> 当前失败是否只是一次坏采样？

建议第一版：

- \(K=4\) 或 \(K=8\)；
- 保持相同采样参数；
- 单独保存 policy random seed；
- 不混入 temperature sweep。

### 采样配置变体

若改变：

- temperature；
- guidance scale；
- diffusion noise schedule；
- sampling steps；
- prompt augmentation；

则候选来自不同的 proposal profile：

\[
q_{\text{profile-}j}(c\mid h_t).
\]

这类实验可以做，但必须单独命名：

```text
STRICT_RESAMPLE
TEMPERATURE_RESAMPLE
GUIDANCE_RESAMPLE
DIFFUSION_PROFILE_RESAMPLE
```

不能把所有候选都统称为“同一分布”。

---

## 4.4 C2：Fresh same-policy replan

截断旧 suffix，使用最新 observation 和 history 重新调用 source VLA：

\[
c_t^{replan}
\sim
\pi_s(a\mid h_t^{fresh},l).
\]

它通常改变：

- observation timestamp；
- 当前 proprioception；
- history window；
- cache；
- chunk boundary；
- 接触状态；
- 当前已完成进度。

因此它测量的是：

> 原 action chunk 是否因为信息陈旧而失败？

而不是：

> 是否能从同一个分布中采到更好动作？

---

## 4.5 C3：Fallback / alternate VLA proposal

候选由另一个冻结 VLA 或 controller 生成：

\[
c_t^{fallback,j}
\sim
\pi_j(a\mid h_t^{pub},l).
\]

Fallback 候选与 source 候选可能在以下方面不同：

- 模型架构；
- 训练数据；
- action representation；
- chunk horizon；
- 控制频率；
- gripper semantics；
- 视觉输入；
- action smoothness；
- latency；
- 任务能力。

### 公共 handoff contract

所有 fallback 只能访问：

\[
h_t^{pub}
=
\{
I_{t-K:t},
q_{t-K:t},
a_{t-K:t},
l,
M_t^{pub}
\}.
\]

禁止：

- 读取 source VLA 私有 hidden state；
- 复制 source cache；
- 使用测试 outcome；
- 使用 simulator privileged pose；
- 省略 fallback warm-up；
- 忽略 action normalization 和 first-action latency。

Fallback 必须从公共历史重建自身上下文。

---

## 4.6 C4：Local correction candidates

局部纠正可以来自：

- fixed primitive；
- motion planner；
- residual controller；
- task-conditioned corrector；
- safety controller。

候选例子：

```text
RETREAT
LIFT
OPEN_GRIPPER
CLOSE_GRIPPER
REALIGN
REGRASP
STABILIZE
OBSTACLE_AVOID
```

它们不属于 VLA 采样分布，而属于独立 proposal distribution：

\[
c_t^{local,j}
\sim
q_j(c\mid h_t).
\]

主 benchmark 不应允许无限自由生成 local correction。

建议固定 3–5 个标准 profile：

1. retreat-short；
2. stabilize-hold；
3. lift-and-reobserve；
4. open-reset-gripper；
5. align-and-regrasp。

---

## 4.7 C5：Rewind / state-restoration proposal

Rewind 是一个宏观候选：

\[
c_t^{rewind}
=
(
\text{target keyframe},
\text{planner trajectory},
\text{termination rule}
).
\]

目标可包括：

- 最近 contact-free keyframe；
- 最近高质量状态；
- 子任务边界；
- 历史 milestone；
- familiar state；
- safe set。

部署时 rewind 必须通过真实控制实现，不能使用 simulator teleport。

Simulator restore 只能用于生成同状态标签。

---

## 4.8 C6：Abstain / request help

当所有自动候选的保守价值都不足时：

```text
SAFE_HOLD
ABSTAIN
REQUEST_HUMAN
RESET
```

必须把 abstention 作为正式动作，而不是系统失败后的隐式处理。

---

# 5. 回退后如何处理候选

## 5.1 回退会改变状态分布

执行：

```text
RETREAT
STABILIZE
REWIND
LOCAL_REGRASP
```

都会改变：

- 机器人位姿；
- 物体位姿；
- 接触关系；
- gripper 状态；
- 摄像机观测；
- 任务进度；
- 可达区域；
- 策略支持度。

因此：

\[
h_t \neq h_{t'}.
\]

旧候选和旧 world-model prediction 必须失效。

## 5.2 严禁复用回退前候选

错误流程：

```text
At h_t:
candidate A
candidate B
candidate C

Execute rewind
→ arrive at h_t'
→ directly execute old candidate B
```

正确流程：

```text
At h_t:
evaluate candidates
→ choose rewind
→ physically execute rewind
→ observe h_t'
→ invalidate all old candidates
→ invalidate all old WM predictions
→ rebuild public history
→ rebuild policy cache
→ regenerate all candidate families
→ rerank at h_t'
```

## 5.3 Fresh re-anchoring

推荐主版本：

回退后使用：

- 当前真实 observation；
- 当前 proprioception；
- recovery 过程中真实执行的动作；
- 最近几帧 recovery history；
- 原任务指令；
- 可选公开 milestone memory。

然后重新调用 source 或 fallback VLA。

优点：

- 符合真实部署；
- 不隐藏 recovery history；
- 不依赖不现实的 cache restore；
- 能正确反映 handoff 和状态偏差。

## 5.4 Milestone context restoration

可作为扩展 operator：

```text
REWIND + FRESH_REPLAN
REWIND + MILESTONE_CONTEXT
```

Milestone context 可以包含：

- 历史关键帧；
- 对应语言子目标；
- 已完成进度摘要；
- 当时的公开动作历史。

但必须同时保留当前真实 observation，不能让旧历史覆盖当前物理证据。

该方案需要审计：

- milestone 与当前状态是否一致；
- 是否引入虚假任务记忆；
- 是否造成 context-state mismatch；
- 是否比 fresh re-anchoring 更好。

---

# 6. 世界模型的正确角色

## 6.1 世界模型不是最终裁判

世界模型提供：

- action-conditioned short-horizon future；
- object relation；
- contact；
- grasp stability；
- collision；
- progress proxy；
- uncertainty；
- support distance。

不提供：

- 最终 ground-truth success；
- 最优候选标签；
- 最终实验真值；
- 长期任务完成的可靠证明。

## 6.2 Candidate-conditioned world model

对每个具体候选：

\[
\hat Z_{t+1:t+H}^{c}
=
W_\phi(
h_t,
c
).
\]

输入必须包含具体 action proposal，而不是只有 operator label。

例如：

- Current：当前剩余 suffix；
- Resample：具体新 action chunk；
- Replan：fresh VLA chunk；
- Fallback：fallback 首个 chunk；
- Local correction：primitive sequence；
- Rewind：planner trajectory prefix；
- Abstain：hold / stop action。

## 6.3 推荐预测目标

优先预测结构化短时结果：

\[
\hat y^{WM}_c=
\{
\text{contact},
\text{object motion},
\text{grasp attachment},
\text{collision},
\text{progress delta},
\text{support distance}
\}.
\]

推荐 horizon：

- 一个 action chunk；
- 4–16 个 control steps；
- 或覆盖一个关键接触阶段。

不建议把长视频像素生成质量作为主要指标。

## 6.4 世界模型证据编码

\[
r_c^{WM}
=
E_{WM}
(
z_t,
\hat Z^c,
c,
\sigma_c^{WM}
).
\]

可提取：

- predicted progress；
- predicted contact quality；
- predicted collision；
- predicted object displacement；
- predicted grasp retention；
- imagined support distance；
- uncertainty；
- action sensitivity。

## 6.5 数据隔离

需要三套独立数据：

### WM Train

训练动态预测。

### Candidate Value Train

使用真实 continuation outcome 训练 candidate critic。

### Calibration / Test

完全独立。

禁止：

- 用 test branch outcome 更新世界模型；
- 让世界模型自生成训练标签再自我验证；
- 用 imagined success 作为最终部署成功；
- 在 test candidate outcome 上构造 executor fingerprint。

---

# 7. Risk Model 与 Candidate Value Model

## 7.1 不建议只预测“最终会不会失败”

单一 terminal risk：

\[
P(Y^{terminal}=0\mid h_t,c)
\]

存在：

- credit assignment 过长；
- 类别不平衡；
- 局部风险和最终成功混淆；
- 可恢复错误和不可恢复错误混淆。

## 7.2 多头风险输出

推荐预测：

\[
\mathbf r(c)
=
\begin{bmatrix}
P(\text{collision})\\
P(\text{object loss})\\
P(\text{grasp instability})\\
P(\text{progress regression})\\
P(\text{task inconsistency})\\
P(\text{irreversible transition})\\
P(\text{unsupported state})
\end{bmatrix}.
\]

同时预测：

\[
p_c^{success},
\quad
p_c^{harm},
\quad
p_c^{feasible},
\quad
\Delta progress_c,
\quad
\sigma_c.
\]

## 7.3 Candidate-conditioned critic

模型：

\[
F_\theta(
h_t,
c,
r_c^{WM}
)
\rightarrow
\{
\hat p_c^{success},
\hat p_c^{harm},
\hat p_c^{feasible},
\hat r_c^{progress},
\hat c_c^{latency},
\hat c_c^{compute},
\hat\sigma_c
\}.
\]

## 7.4 为什么不直接训练平面 selector

不推荐：

\[
\pi_\theta(h_t)
\rightarrow
\{
RESAMPLE,
REPLAN,
FALLBACK,
REWIND
\}.
\]

问题：

- 候选数量可变；
- 不同 operator 可有多个实现；
- 无法接入新 VLA；
- 容易学习 task ID 和 policy ID shortcut；
- 无法评价具体 action chunk；
- 无法输出候选不确定性；
- 所有候选都差时仍会被迫选择一个。

推荐学习：

\[
V_\theta(h_t,c).
\]

---

# 8. 候选排序与保守执行规则

## 8.1 Candidate utility

建议报告独立指标，不应只依赖人为 utility。

内部选择可定义：

\[
\hat U_c
=
\hat p_c^{success}
+
\alpha_p \hat r_c^{progress}
-
\alpha_h \hat p_c^{harm}
-
\lambda_l \hat c_c^{latency}
-
\lambda_g \hat c_c^{compute}
-
\lambda_s \hat c_c^{safety}.
\]

主论文仍需单独报告：

- terminal success；
- harm；
- latency；
- compute；
- intervention frequency；
- human requests。

## 8.2 保守排序

\[
c^*
=
\arg\max_{c\in\mathcal C_t}
\operatorname{LCB}
(
\hat U_c
).
\]

仅当：

\[
\operatorname{LCB}
(
\hat U_{c^*}
-
\hat U_{\text{current}}
)
>
\delta
\]

并满足：

\[
\hat p_{c^*}^{harm}<\eta_h,
\]

\[
\hat p_{c^*}^{feasible}>\eta_f,
\]

才替换当前候选。

## 8.3 所有候选都差时必须升级

如果：

\[
\max_{c\in\mathcal C_t^{same-state}}
\operatorname{LCB}(\hat U_c)
<
\tau,
\]

不能硬选“相对最好”的候选。

应进入：

```text
same-state candidates exhausted
→ state-changing recovery
→ regenerate candidates
```

---

# 9. 分层恢复阶梯

## Tier 0：Execute current

条件：

\[
LCB(V(h_t,c^{current}))\ge \tau_{execute}.
\]

## Tier 1：Strict same-policy resample

生成 \(K\) 个严格同分布候选。

适用于：

- 单次采样不稳定；
- 生成分布中可能存在成功候选；
- 当前状态未明显改变。

## Tier 2：Fresh same-policy replan

适用于：

- 当前 suffix 过时；
- 新观察已经包含重要变化；
- source VLA 仍可能支持当前状态。

## Tier 3：Alternate-policy proposal

适用于：

- source VLA candidate pool 整体低价值；
- fallback 在该任务或状态具有能力互补；
- handoff contract 可满足；
- 额外延迟和算力可接受。

## Tier 4：Local correction

适用于：

- 几何或接触局部异常；
- 不需要改变高层任务语义；
- 有明确可行的 primitive。

## Tier 5：State restoration

```text
RETREAT
STABILIZE
REWIND
MILESTONE RECOVERY
```

适用于：

- 当前状态已离开 VLA 支持集；
- 所有同状态 action candidate 都低价值；
- 存在物理可达恢复目标。

## Tier 6：Abstain / human

适用于：

- 无自动候选支持；
- 不确定性过高；
- 不可逆风险；
- 预算耗尽；
- 回退不可达。

---

# 10. 机会空间的分解

定义嵌套候选集：

\[
\mathcal C_0
=
\{
current
\},
\]

\[
\mathcal C_1
=
\mathcal C_0
\cup
resamples,
\]

\[
\mathcal C_2
=
\mathcal C_1
\cup
replan,
\]

\[
\mathcal C_3
=
\mathcal C_2
\cup
fallback,
\]

\[
\mathcal C_4
=
\mathcal C_3
\cup
local\ corrections.
\]

对应 oracle：

\[
S_k
=
P(
\exists c\in\mathcal C_k:Y_c=1
).
\]

增量：

\[
H_{\text{sampling}}
=
S_1-S_0,
\]

\[
H_{\text{replan}}
=
S_2-S_1,
\]

\[
H_{\text{fallback}}
=
S_3-S_2,
\]

\[
H_{\text{local}}
=
S_4-S_3.
\]

状态改变后：

\[
H_{\text{recovery}}
=
S_{\text{post-recovery}}
-
S_4.
\]

这套分解回答：

1. VLA 是否只是采样偶然失败；
2. fresh observation 是否足够；
3. 是否需要更强或互补 VLA；
4. 是否需要局部控制器；
5. 是否必须改变物理状态。

---

# 11. Same-State Candidate Benchmark

## 11.1 Benchmark query

一个 query：

\[
x=
(
h_t,
s_t^{restore},
\mathcal C_t,
\mathcal B
),
\]

其中：

- \(h_t\)：部署可见历史；
- \(s_t^{restore}\)：仅用于 simulator fork；
- \(\mathcal C_t\)：候选 proposal 集；
- \(\mathcal B\)：计算、时间与安全预算。

标签：

\[
\mathcal Y_x
=
\{
Y_{c,\omega}
:
c\in\mathcal C_t,
\omega\in\Omega
\}.
\]

## 11.2 每个 snapshot 的候选建议

第一版 pilot：

```text
1 × current suffix
4 × strict source resamples
1 × fresh source replan
1 × fallback VLA proposal
2 × fixed local corrections
1 × abstain
```

Rewind 作为 state-changing operator 独立执行。

## 11.3 分支标签

每个 candidate arm 记录：

- terminal success；
- short-horizon risk event；
- progress delta；
- first irreversible event；
- object drop；
- collision；
- grasp loss；
- candidate completion；
- continuation steps；
- GPU time；
- wall-clock latency；
- human reset；
- final support；
- failure reason。

## 11.4 Source trajectory 类型

必须包含：

- clean successes；
- natural failures；
- controlled failures；
- near-failure states；
- benign high-risk-score states；
- silent failures；
- different task stages；
- different source VLA；
- contact and non-contact states。

不能只从失败状态采样，否则无法学习：

- continue 的真实价值；
- false alarm；
- harmful correction；
- redundant intervention。

## 11.5 Snapshot fidelity

验证：

1. physics state hash；
2. controller state；
3. RNG state；
4. rendering parity；
5. source replay parity；
6. action chunk parity；
7. sensor delay parity；
8. contact parity；
9. task-state parity；
10. policy seed parity。

---

# 12. 三个必须先做的 Opportunity Screen

## Screen A：Candidate Proposal Headroom

### 目标

回答：

> 候选集合中是否经常存在比原动作更好的动作？

### 建议规模

- 12–16 tasks；
- 100–200 independent snapshots；
- clean + natural failure + 2 perturbation families；
- 每状态 6–10 candidates；
- 关键边界状态 3 seeds。

### 指标

\[
S_{base}
=
P(Y_{current}=1),
\]

\[
S_{oracle@K}
=
P(
\exists c\in\mathcal C_t:Y_c=1
),
\]

\[
H_{candidate}
=
S_{oracle@K}-S_{base}.
\]

还需报告：

- rescue coverage；
- resample-only success；
- replan-only success；
- fallback-only success；
- local-only success；
- all-candidates-fail；
- candidate diversity；
- task concentration。

### Go 条件

建议至少满足：

1. candidate oracle headroom \(\ge 8\)–\(10\) pp；
2. base failure 中至少约 20% 存在成功替代候选；
3. 独有成功不只集中于一个 task；
4. 至少两个 generator family 有独有成功；
5. task-held-out pilot 方向保持。

若不通过：

- 不训练复杂 world model；
- 不训练 selector；
- 先改善候选生成或增加互补 executor。

---

## Screen B：Candidate Learnability

### Baselines

- first candidate；
- random candidate；
- VLA likelihood / confidence；
- action magnitude；
- action smoothness；
- policy seed heuristic；
- current-frame-only；
- history-only；
- action-only；
- task ID；
- policy ID；
- privileged state upper bound；
- candidate-conditioned critic；
- WM-conditioned critic；
- oracle candidate。

### 指标

\[
G_{\text{captured}}
=
\frac{
U_{\text{learned}}
-
U_{\text{first}}
}{
U_{\text{oracle}}
-
U_{\text{first}}
}.
\]

### Go 条件

建议同时满足：

- learned ranking 显著超过 random；
- 超过 VLA confidence；
- task-held-out 方向稳定；
- 捕获至少 30%–50% oracle candidate gain；
- 部署成功提高约 5 pp，或显著降低 harm；
- 额外 latency 可接受；
- 不依赖 test task ID / policy ID shortcut。

---

## Screen C：Residual Recovery Headroom

只在：

\[
\forall c\in\mathcal C_t^{same-state},\quad Y_c=0
\]

的状态上测试：

```text
RETREAT
STABILIZE
REWIND
STRONG_FALLBACK
RESET
ABSTAIN
```

定义：

\[
H_{\text{residual}}
=
U(
\text{oracle recovery}
\mid
\text{all same-state candidates fail}
)
-
U(
\text{best fixed recovery}
\mid
\text{all same-state candidates fail}
).
\]

若 residual oracle gap 小：

- 使用固定 recovery ladder；
- 不训练 recovery selector；
- learned 模块只负责 candidate evaluation。

---

# 13. 世界模型增量价值审计

## 13.1 模型对比

```text
History-only critic
Action-only critic
Observation + action critic
VLA-hidden probe
WM residual only
WM predicted latent
WM structured events
Full WM-conditioned critic
Privileged simulator dynamics upper bound
```

## 13.2 关键问题

- WM 是否真正感知 action 差异？
- action-shuffled 后性能是否下降？
- operator-shuffled 后性能是否下降？
- contact states 是否显著更差？
- OOD candidate 是否错误自信？
- predicted progress 是否与真实 progress 相关？
- imagined ranking 是否与真实 ranking 一致？
- WM 增量是否仅来自更多参数？

## 13.3 World Model Gate

作为主贡献必须满足：

1. full WM critic > history-only；
2. held-out perturbation 或 VLA 上仍有增量；
3. imagined-real ranking 有稳定相关；
4. false-confidence 可被 uncertainty 识别；
5. 增益覆盖额外 compute；
6. real-robot calibration 不严重失效。

若不通过：

- WM 降为 optional evidence 或 baseline；
- 主方法保留 true-outcome candidate critic；
- 不为了叙事强行保留 WM。

---

# 14. Cross-VLA 设计

## 14.1 Source generalization

固定 fallback：

```text
Source A → Fallback X
Source B → Fallback X
Source C → Fallback X
```

测试：

> 同一个 candidate evaluator 能否评价不同 source VLA 的候选？

## 14.2 Fallback generalization

固定 source：

```text
Source A → Fallback X
Source A → Fallback Y
```

测试：

> 评价模型是否依赖某一个强 fallback checkpoint？

## 14.3 Unseen policy-pair

训练：

```text
A → X
B → X
A → Y
```

测试：

```text
B → Y
```

## 14.4 Held-out source VLA

训练：

```text
Source A
Source B
```

测试：

```text
Source C
```

候选 evaluator 不读取 source VLA 私有 latent。

## 14.5 推荐输入

模型无关输入：

\[
h_t^{public}
=
\{
RGB,
proprioception,
action\ history,
language,
candidate\ action
\}.
\]

可加入 behavior fingerprint：

- action scale；
- smoothness；
- horizon；
- gripper behavior；
- control frequency；
- latency；
- visual input support。

不建议主模型直接依赖 one-hot policy ID。

---

# 15. 训练目标

## 15.1 Success loss

\[
\mathcal L_{success}
=
BCE(
\hat p_c^{success},
y_c^{success}
).
\]

## 15.2 Risk loss

\[
\mathcal L_{risk}
=
\sum_k
BCE(
\hat r_{c,k},
r_{c,k}
).
\]

## 15.3 Feasibility loss

\[
\mathcal L_{feas}
=
BCE(
\hat p_c^{feasible},
y_c^{feasible}
).
\]

## 15.4 Pairwise ranking

同一 snapshot 的两个 candidate：

\[
d_{ij}
=
U_i-U_j.
\]

\[
\mathcal L_{rank}
=
\log
\left[
1+
\exp(
-\operatorname{sign}(d_{ij})
(\hat U_i-\hat U_j)
)
\right].
\]

近似 tie：

\[
|d_{ij}|<\epsilon
\]

可以忽略或使用 tie-aware loss。

## 15.5 Calibration

使用独立 calibration split：

- temperature scaling；
- beta calibration；
- deep ensemble；
- bootstrap；
- split conformal；
- group-conditional conformal。

## 15.6 总损失

\[
\mathcal L
=
\mathcal L_{success}
+
\lambda_r\mathcal L_{risk}
+
\lambda_f\mathcal L_{feas}
+
\lambda_{rank}\mathcal L_{rank}
+
\lambda_c\mathcal L_{cost}
+
\lambda_{wm}\mathcal L_{wm}.
\]

推荐分阶段训练：

1. 训练 / 冻结 WM；
2. 训练 candidate critic；
3. 独立校准；
4. 闭环部署；
5. 不在 test outcome 上继续更新。

---

# 16. 推荐算法

## 16.1 核心伪代码

```python
def act(history, source_vla, fallback_vlas, budget):
    # 1. 当前候选
    current = source_vla.current_suffix(history)
    current_value = critic.evaluate(history, current)

    if current_value.lcb >= EXECUTE_THRESHOLD:
        return execute(current)

    # 2. 严格同分布重采样
    resamples = source_vla.strict_resample(
        history=history,
        num_candidates=budget.num_resamples,
        keep_sampling_profile_fixed=True,
    )

    same_state_candidates = [current, *resamples]

    # 3. Fresh replan
    if budget.allow_replan:
        replanned = source_vla.replan(history)
        same_state_candidates.append(replanned)

    # 4. Fallback VLA 候选
    for fallback in fallback_vlas:
        if fallback.feasible(history, budget):
            proposal = fallback.propose_from_public_history(history)
            same_state_candidates.append(proposal)

    # 5. Local correction 候选
    for corrector in local_correctors:
        if corrector.feasible(history, budget):
            same_state_candidates.append(corrector.propose(history))

    # 6. 世界模型证据 + candidate critic
    scored = []
    for candidate in same_state_candidates:
        wm_evidence = world_model.imagine(history, candidate)
        value = critic.evaluate(history, candidate, wm_evidence)
        scored.append((candidate, value))

    best_candidate, best_value = max(
        scored,
        key=lambda item: item[1].lcb,
    )

    # 7. 只有绝对可靠时才执行
    if (
        best_value.lcb >= EXECUTE_THRESHOLD
        and best_value.harm_ucb <= HARM_THRESHOLD
        and best_value.feasible_prob >= FEASIBILITY_THRESHOLD
    ):
        return execute(best_candidate)

    # 8. 当前状态候选空间无解，升级恢复
    recovery = fixed_recovery_ladder.propose(history, budget)

    if recovery is not None:
        execute(recovery)

        # 回退改变状态，旧候选全部失效
        new_history = observe_and_reanchor()
        invalidate_all_old_candidates()
        invalidate_all_old_world_model_predictions()

        return act(
            history=new_history,
            source_vla=source_vla,
            fallback_vlas=fallback_vlas,
            budget=budget.after(recovery),
        )

    return abstain_safely()
```

## 16.2 第一版不要训练端到端 escalation policy

推荐：

```text
Learned candidate critic
+
Fixed escalation ladder
```

只有 residual recovery oracle 证明：

\[
H_{\text{residual}}
\]

足够大，才训练 recovery router。

---

# 17. 核心实验

## E0：Candidate Opportunity Audit

- current；
- strict resample；
- fresh replan；
- fallback；
- local correction；
- same-state oracle。

回答：

- 是否有候选 headroom；
- 哪类 generator 有独有成功；
- 候选成功是否集中在少数 task；
- 所有候选失败的比例；
- 当前候选被替换是否会造成 harm。

## E1：Same Distribution vs Reconditioning

比较：

```text
Strict same-distribution resampling
Sampling-profile variation
Fresh replan
```

回答：

- 收益来自采样随机性还是新观察；
- temperature sweep 是否只是更多计算；
- replan 是否真正有独立价值。

## E2：Candidate Generator Decomposition

嵌套 oracle：

```text
Original
+ Resamples
+ Replan
+ Fallback
+ Local correction
+ State recovery
```

展示每一级增量 headroom。

## E3：Risk Detection ≠ Candidate Selection

比较：

- 当前动作 failure AUROC；
- candidate ranking accuracy；
- closed-loop success；
- harm reduction。

验证：

> 能检测风险，不代表能找到更好的动作。

## E4：World Model Incremental Value

比较：

- no WM；
- WM residual；
- WM latent prediction；
- structured contact/progress prediction；
- full critic。

## E5：Matched Compute

固定：

- total policy calls；
- total GPU seconds；
- total wall-clock；
- candidate count；
- intervention count。

比较：

```text
Source only
Fixed N-resample
Always replan
Always fallback
Immediate OFT
Risk-triggered resample
DREAMSTEER-style ranking
Full RASE-PRE
```

## E6：Calibration and Abstention

画：

- success–coverage；
- harm–coverage；
- compute–coverage；
- human request–coverage。

## E7：Held-Out Task

严格 task-disjoint。

## E8：Held-Out Perturbation

测试新的 failure family。

## E9：Held-Out VLA

训练 A/B，测试 C。

## E10：Unseen Policy Pair

训练 A→X、B→X、A→Y，测试 B→Y。

## E11：Second Simulator

验证结果不只来自 LIBERO。

## E12：Real Robot

验证：

- success；
- harm；
- intervention latency；
- fallback handoff；
- rewind 可执行性；
- calibration；
- human workload。

---

# 18. Baselines

## 18.1 Candidate generation

- Original only；
- Fixed \(K\)-resample；
- Best-of-\(K\) by VLA likelihood；
- Random candidate；
- Action smoothness heuristic；
- Fresh replan；
- Always fallback；
- Local primitive only。

## 18.2 Risk and verification

- VLA uncertainty；
- WM residual threshold；
- current-frame risk；
- history-only risk；
- action-only risk；
- progress stall；
- CheckVLA-style verification。

## 18.3 Candidate ranking

- random ranking；
- VLA likelihood；
- lowest predicted risk；
- highest predicted progress；
- history-only critic；
- WM-only imagined ranking；
- candidate-conditioned critic；
- full WM-conditioned critic；
- same-state oracle。

## 18.4 Recovery

- immediate OFT；
- fixed local recovery；
- fixed rewind；
- fixed escalation ladder；
- Agentic-style recovery manager；
- abstain-only；
- human upper bound。

## 18.5 Shortcut probes

- task ID；
- policy ID；
- time index；
- perturbation ID；
- candidate generator ID；
- action magnitude only；
- privileged simulator state upper bound。

---

# 19. 主要指标

## 19.1 Terminal success

\[
S=P(\text{task success}).
\]

## 19.2 Oracle candidate headroom

\[
H_{candidate}
=
S_{oracle@K}
-
S_{original}.
\]

## 19.3 Captured oracle gap

\[
\rho
=
\frac{
S_{learned}-S_{original}
}{
S_{oracle@K}-S_{original}
}.
\]

## 19.4 Ranking regret

\[
R
=
U_{oracle\ candidate}
-
U_{selected}.
\]

## 19.5 Harmful replacement rate

\[
HIR
=
P(
U_{selected}
<
U_{current}-\epsilon
\mid
replace
).
\]

## 19.6 Futile search rate

\[
FSR
=
P(
\forall c,\ Y_c=0
\mid
candidate\ search
).
\]

## 19.7 Beneficial replacement precision

\[
BRP
=
P(
U_{selected}
>
U_{current}+\epsilon
\mid
replace
).
\]

## 19.8 Candidate search efficiency

\[
\frac{
\text{additional successes}
}{
\text{extra GPU seconds}
}.
\]

## 19.9 Escalation efficiency

对 Tier \(k\)：

\[
\Delta_k
=
P(
\text{Tier }k\text{ rescues}
\mid
\text{Tier }0{:}k-1\text{ fail}
).
\]

## 19.10 Risk–coverage

报告：

- autonomous coverage；
- success；
- harm；
- compute；
- abstention；
- human requests。

---

# 20. 统计协议

## 20.1 独立单位

- task；
- source episode；
- perturbation cluster；
- policy pair；
- real-world initial-condition block。

同一 snapshot 的候选不是独立样本。

## 20.2 配对统计

同一 snapshot 的候选天然配对。

使用：

- paired bootstrap；
- cluster bootstrap；
- McNemar；
- randomization test；
- mixed-effects logistic regression；
- task-level bootstrap。

## 20.3 混合效应模型

\[
\operatorname{logit}P(Y=1)
=
\beta_0
+
\beta_1 Method
+
\beta_2 Generator
+
\beta_3 SourceVLA
+
\beta_4 FallbackVLA
+
\beta_5 Perturbation
+
\beta_6 Method\times SourceVLA
+
(1|Task)
+
(1|Episode)
+
(1|PolicyPair).
\]

## 20.4 主假设

预注册：

> 在 matched total compute 下，RASE-PRE 的 terminal success 高于最强 fixed candidate-search baseline，并降低 harmful replacement。

## 20.5 次假设

- WM 提高 held-out candidate ranking；
- strict resampling 有真实 oracle headroom；
- fresh replan 提供独立增量；
- cross-VLA evaluator 在 unseen pair 上仍有效；
- state-changing recovery 只对 residual failures 有价值。

---

# 21. Real Robot 方案

## 21.1 实机不做 exact same-state counterfactual

实机使用：

- randomized block；
- repeated initial condition bins；
- matched perturbation；
- task × method balanced design；
- calibration split 与 test split 分离。

## 21.2 推荐任务

1. pick-and-place；
2. drawer / cabinet；
3. insertion；
4. grasp–transport–place 多阶段任务。

## 21.3 扰动

- object displacement；
- grasp slip；
- occlusion；
- moving obstacle；
- action latency；
- gripper timing error。

## 21.4 候选池

实机第一版：

```text
Current
2–4 same-policy resamples
Fresh replan
1 fallback proposal
2 local corrections
Abstain
```

Rewind 作为独立扩展。

## 21.5 安全

- workspace bounds；
- speed limits；
- force / torque limits；
- emergency stop；
- collision checker；
- safe-hold pose；
- object damage policy；
- human takeover protocol。

---

# 22. 最大风险与止损

## 风险一：候选空间没有 headroom

表现：

```text
Original fails
All resamples fail
Replan fails
Fallback fails
Local correction fails
```

处理：

- 不训练 selector；
- 改善 proposal generator；
- 增加真正互补的策略或 primitive；
- 研究 unsupported-state detection。

## 风险二：Oracle@K 很高，但 learned evaluator 不会选

处理：

- 先使用真实 candidate outcomes 做 pairwise critic；
- 增加 candidate-conditioned action representation；
- 检查 task/policy shortcut；
- world model 只作为增量；
- 若仍不行，论文转向 proposal–evaluation gap。

## 风险三：世界模型在接触状态最不准

处理：

- 短 horizon；
- 结构化 contact prediction；
- uncertainty-aware abstention；
- action sensitivity audit；
- WM 降为 optional evidence。

## 风险四：检测准确但没有可纠正候选

这证明：

\[
\text{failure detectability}
\neq
\text{recoverability}.
\]

处理：

- 单独报告 detection–recovery gap；
- 不把 risk AUROC 当闭环贡献；
- 扩充 generator 或状态恢复。

## 风险五：Fallback 全面支配

如果 strong fallback 从头到尾更好且成本也低：

- recovery framing 被削弱；
- 必须做 policy replacement audit；
- source policy 只能以成本、端侧部署或资源约束作为存在理由。

## 风险六：系统过于复杂

控制主贡献：

1. same-state candidate benchmark；
2. candidate-conditioned critic；
3. world-model incremental-value audit。

Fixed escalation ladder 是部署机制，不作为第四个算法贡献。

---

# 23. 分阶段执行计划

## Phase A：Candidate Generator Audit

完成：

- strict resample；
- fresh replan；
- fallback proposal；
- local correction；
- same-state branch runner；
- oracle decomposition。

输出：

- Candidate Proposal Headroom；
- generator unique-success matrix；
- Go / No-Go。

## Phase B：Simple Candidate Critic

先做：

- observation + action；
- history + action；
- pairwise ranking；
- no WM；
- task-held-out；
- calibration。

输出：

- learned evaluator 是否捕获 oracle gap。

## Phase C：World Model Evidence

完成：

- action-conditioned short-horizon WM；
- structured contact/progress heads；
- uncertainty；
- imagined-real ranking audit；
- full vs no-WM。

## Phase D：Closed-Loop Predict–Rank–Escalate

完成：

- conservative execution；
- fixed escalation ladder；
- candidate invalidation；
- post-recovery re-anchoring；
- matched compute。

## Phase E：Cross-VLA

完成：

- multiple source VLA；
- multiple fallback VLA；
- held-out pair；
- held-out source；
- no-private-latent generalization。

## Phase F：Second Platform and Real Robot

完成：

- second simulator；
- randomized-block real experiments；
- latency；
- human workload；
- safety；
- calibration。

---

# 24. 顶会版最低条件

## CoRL / RSS 强版本

- same-state candidate benchmark；
- 明确 proposal/evaluation/recovery decomposition；
- 至少两个 source policies；
- 强 fixed baseline；
- closed-loop deployment；
- 第二平台或正式实机；
- candidate oracle 显著高于 original；
- learned critic 超过 fixed search heuristic。

## CVPR 强版本

除上述内容，还需要：

- 视觉时序和 action-conditioned prediction 是核心；
- WM structured visual/contact evidence 有明确增量；
- held-out task / perturbation / VLA 泛化；
- 大规模视觉候选数据；
- 真实机器人或强外部视觉验证；
- 方法不是已有 candidate ranking 的简单复现。

## NeurIPS / ICLR 强版本

需要强调：

- offline candidate-value learning；
- distributional prediction；
- uncertainty；
- hierarchical decision under budgets；
- unseen executor / proposal distribution generalization；
- proposal–evaluation–recovery decomposition 的一般性。

---

# 25. 可提出与不能提出的 Claim

## 可以提出

- 我们将冻结 VLA 的运行时可靠性分解为候选生成、候选评价和状态恢复三个不同问题；
- 严格同分布 resample、fresh replan、fallback 和 rewind 属于不同 proposal regimes；
- 回退改变物理状态，因此必须重新观察、重新生成并重新评价候选；
- same-state candidate outcomes 可直接测量 proposal headroom 和 evaluator regret；
- 世界模型作为 candidate-conditioned predictive evidence，而真实 continuation 提供最终监督；
- 系统只在存在绝对可靠候选时执行，否则逐级升级或 abstain；
- 跨 VLA 评价必须使用 public-history handoff 和统一 action contract。

## 不能提前提出

- 多采样一定能显著提升所有 VLA；
- 世界模型一定优于 history-only critic；
- fallback 一定与 source 互补；
- learned selector 一定超过 best fixed；
- 回退一定优于 immediate OFT；
- 仿真 same-state oracle 等于真实机器人反事实；
- 所有候选都来自同一分布；
- 使用同一个 VLA 就意味着候选同分布。

---

# 26. 推荐论文叙事

## 标题方向一

> **Try Before You Recover: Risk-Aware Candidate Selection and Escalation for Frozen VLA Policies**

## 标题方向二

> **Where Does VLA Recovery Come From? Decomposing Proposal, Evaluation, and State-Restoration Headroom**

## 标题方向三

> **RASE-PRE: Same-State Candidate Evaluation with World-Model Evidence for Frozen VLA Policies**

## 核心叙事

现有 VLA recovery 系统通常把失败检测直接连接到固定恢复：

```text
risk high
→ replan / rewind / switch
```

但高风险并不能说明：

- 当前 VLA 的分布中没有好候选；
- replan 一定优于 resample；
- fallback 一定支持当前状态；
- 当前状态真的需要回退；
- 所有自动候选都失败。

RASE-PRE 先从相同状态生成异构候选，利用真实 continuation labels 学习 candidate-conditioned value，并用短时世界模型提供未来证据。只有当前状态下没有可靠候选时，系统才改变策略或物理状态。回退后，所有旧候选和旧想象全部失效，系统从新的真实状态重新开始。

---

# 27. 最终推荐

最值得立即执行的实验不是训练完整 world model，而是：

```text
Current candidate
vs
4 strict same-distribution resamples
vs
1 fresh replan
vs
1 fallback proposal
vs
2 local corrections
```

从相同 snapshot 运行真实 continuation。

先得到：

\[
S_0,S_1,S_2,S_3,S_4.
\]

如果：

\[
S_4-S_0
\ge 8\text{–}10\text{ pp}
\]

且至少两个 generator family 产生稳定独有成功，再训练 candidate-conditioned critic。

随后验证：

\[
V(h_t,c,\hat Z_c^{WM})
>
V(h_t,c)
\]

是否成立。

最终系统应坚持：

> **同分布 resample 用于处理采样偶然性；fresh replan 用于处理信息陈旧；fallback 用于处理能力不足；local correction 用于处理局部几何和接触错误；rewind 用于改变不受支持的物理状态。任何状态改变后，都必须废弃旧候选并从新状态重新生成、重新想象和重新排序。**

这条路线既恢复了最初的研究想法，又吸收了 Phase 0 的核心经验：

> **在训练任何 selector 之前，先用真实同状态 outcome 证明候选空间确实存在足够大的、可学习的收益。**
