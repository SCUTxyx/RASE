<!-- 来源:用户 2026-08-19 提供的顶会冲刺版规划 v3(取代 v2;v2 存档于 RASE_ROADMAP_2026_08_19_v2.md)。执行以本文为准。 -->
# RASE 顶会冲刺版后续研究规划 v3

## Predicting Action Consequences, Not Policy Identities

> 日期：2026-08-19  
> 目标：以 **CVPR / CoRL / RSS 等顶会标准**推进 RASE，同时优先保证科学问题清晰、实验可执行、失败可快速止损。  
> 核心愿景不变：**zero-shot 到多个未见 VLA + 风险驱动 continue/switch/fallback + 跨平台带来闭环成功率收益**。  
> 本文档综合当前 RASE idea、截至 2026-08-19 的实验进展、已发现问题，以及上一版 World-Model 后续规划重新收敛。  
> 本版本的关键原则是：**不堆模块；先证明一个足够新的核心机制，再逐层扩展。**

---

# 0. Executive Decision

## 0.1 最终主线

后续 RASE 的核心不再是：

> 训练一个模型判断“某个 VLA 最后会不会成功”。

而是：

> **在同一个真实物理状态下，对多个冻结 VLA 给出的候选动作进行反事实未来预测，判断每个动作会把世界带向什么状态，以及该未来是否正在进入失败、偏移累积或不可恢复区域，再进行保守选择。**

正式建模为：

$$
(s_t,a_i)
\rightarrow
\hat{\tau}_{t+1:t+H}^{\,i}
\rightarrow
R_i
\rightarrow
\{\text{continue, switch, fallback, abort}\}
$$

其中：

- \(s_t\)：当前共享物理状态；
- \(a_i\)：任意冻结 VLA 产生的 candidate；
- \(\hat{\tau}^{\,i}\)：candidate-conditioned predicted future；
- \(R_i\)：与 candidate source identity 无关的未来风险；
- selector：根据相对风险与不确定性做保守 arbitration。

---

## 0.2 顶会主贡献只保留三件事

主论文优先只讲：

### Contribution A — Same-State Cross-VLA Counterfactual Protocol

在**完全相同的物理状态**上比较来自不同冻结 VLA 的候选动作未来，而不是比较各自 on-policy trajectory。

### Contribution B — Policy-Agnostic Predictive Risk

通过 World Model 显式学习：

$$
(s,a)\rightarrow future\ consequence
$$

再学习：

$$
future\ consequence\rightarrow risk
$$

目标是学习 action consequence，而不是 policy competence。

### Contribution C — Zero-Shot Heterogeneous VLA Arbitration

训练 risk/world model 时不见测试 VLA，测试时：

- 不输入 VLA identity；
- 不使用 task lookup；
- 不使用 test-VLA calibration；
- 不 fine-tune 测试 VLA；
- 直接比较 unseen VLA candidate risk；
- 最终提升闭环成功率。

---

## 0.3 OPD、Bandit、RL 的新定位

### OPD

不是论文中心。

定位为：

> **将昂贵 World-Model predictive risk 蒸馏成轻量实时 risk predictor。**

如果 OPD distillation 有显著 latency/throughput 价值，则进入主文；否则放 appendix。

### Contextual Bandit

作为：

> **unseen VLA / platform distribution shift 下的 residual online adaptation。**

不是 zero-shot claim 的替代品。

### Full RL

不是当前优先方向。

只有 selector 存在明显长期依赖、switch cost、probe cost、delayed reward 时再启用。

---

# 1. 当前实验到底已经证明了什么

后续所有工作必须建立在已有证据上，而不是重新从零开始。

---

## 1.1 已经成立：Action risk signal 可学习

已有：

- K3 same-root pairwise accuracy ≈ 0.750；
- K5 raw-action accuracy ≈ 0.787；
- Semantic Selector nested OOF ≈ 0.871；
- OPD 在训练分布上的风险 calibration 有明显信号；
- failure trajectory risk score 会随失败进程下降。

因此：

> **“动作/状态中不存在任何 outcome signal”这个否定假设已经基本排除。**

无需继续大量重复证明 IID learnability。

---

## 1.2 已经成立：Selector opportunity 不是天然存在

已有实验说明：

```text
π0.5 ceiling
fallback domination
π0-fast deterministic diversity collapse
libero_90 all-fail
```

都会让 selector 没有部署空间。

因此后面所有新 benchmark / policy 必须先过：

```text
E0 candidate diversity
E1 non-ceiling / non-floor competence
E2 oracle headroom / comparative advantage
```

---

## 1.3 已经成立：OFT 模型对提供极强机会窗口

当前：

```text
oft_spatial:
spatial 100%
object   0%

oft_object:
spatial   0%
object  100%
```

说明：

- heterogeneous frozen policies 可以存在巨大 comparative advantage；
- oracle headroom 非常高；
- selector 有可能显著提高成功率。

但是：

> 这个结果主要是 **task-level complementarity**，不能单独证明实时 action selector。

因此 OFT pair 继续作为：

```text
debug / opportunity / controlled benchmark
```

但不能成为顶会 paper 的唯一主结果。

---

## 1.4 当前最重要的负结果：99.2% 不是 RASE 最终答案

当前 lookup 闭环 99.2% 本质是：

```text
task -> which OFT model is good
```

而不是：

```text
candidate future -> risk -> selection
```

换 unseen task / VLA / platform 会失效。

因此：

> **99.2% lookup 保留为“shortcut upper bound / diagnostic”，不再作为 RASE 方法主结果。**

---

## 1.5 当前真正悬空的 Claim

目前完全没有正式证明：

$$
Risk_{train\ VLA}(s,a)
\rightarrow
Risk_{unseen\ VLA}(s,a)
$$

即：

> **risk 是否 transferable across heterogeneous policies。**

这必须成为下一阶段第一科学问题。

---

# 2. 顶会竞争环境：必须主动避开的已有路线

截至 2026-08-19，以下方向已经比较拥挤。

---

## 2.1 V-GPS

**Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance**  
arXiv:2410.13816

已经证明：

- offline-RL value function 可以 rerank actions；
- 同一个 value function 可以改善多个不同架构的 generalist policies；
- 在多个 robotic platforms 上验证。

因此 RASE 不能只声称：

> “一个外部 value/risk model 可以提升多个 policy。”

这不够新。

---

## 2.2 RoboMonkey

**RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models**  
arXiv:2506.17811

已经覆盖：

- VLA candidate sampling；
- verifier；
- candidate selection；
- OOD improvement。

因此不能把创新点写成：

> “多采几个 candidate，再用 verifier 选。”

---

## 2.3 RoVer

**RoVer: Robot Reward Model as Test-Time Verifier for VLA**  
arXiv:2510.10975

已经覆盖：

- process reward；
- candidate scoring；
- candidate refinement；
- frozen VLA test-time verification。

因此仅仅：

$$
(s,a)\rightarrow scalar\ reward\rightarrow select
$$

也不够。

---

## 2.4 Pre-VLA

**Pre-VLA: Preemptive Runtime Verification for Reliable VLA and World-Model Rollouts**  
arXiv:2605.22446

已经覆盖：

- action chunk 执行前 validity assessment；
- safety + advantage prediction；
- adaptive resampling；
- LIBERO 闭环成功率提升。

因此不能仅声称：

> “执行动作之前先做 risk verification。”

---

## 2.5 CheckVLA

**CheckVLA: Execution-Time Verification with Action-Conditioned World Model**  
arXiv:2607.26789

已经非常接近：

- action-conditioned world model；
- expected consequence；
- deviation propagation；
- runtime intervention。

因此：

```text
VLA action
-> world model
-> detect deviation
-> repair
```

已经不是足够的新颖点。

---

## 2.6 World Pilot / RISE 等

World Pilot 已经使用 world/action priors steer VLA；  
RISE 已经使用 imagined world rollout + value model 改善机器人 policy。

因此：

> **World Model + VLA 本身不是 novelty。**

---

# 3. RASE 必须死守的 Novelty Boundary

真正可以形成顶会差异的是：

# “Cross-policy counterfactual arbitration”

而不是：

# “single-policy verification”

---

## 3.1 RASE 的 candidate 来源不同

目标场景：

```text
same physical state s_t

        OpenVLA candidate
       /
s_t --- π0 candidate
       \
        SmolVLA candidate
         \
          fallback candidate
```

所有 candidate 都转换到 canonical representation，然后：

$$
WM(s_t,a_i)
\rightarrow
Future_i
\rightarrow
Risk_i
$$

World Model / Risk Critic：

> **不知道 candidate 来自哪个 VLA。**

---

## 3.2 顶会核心假设

我们真正要验证：

$$
P(s_{t+1:t+H}|s_t,a,\pi)
\approx
P(s_{t+1:t+H}|s_t,a)
$$

也就是说：

> **物理世界对动作的响应应该比 policy 的最终 competence 更 policy-invariant。**

于是：

$$
Action\ Consequence
$$

有可能成为：

$$
Cross\text{-}VLA\ transferable\ bottleneck
$$

---

## 3.3 最关键 novelty experiment

必须出现这样的实验：

### Direct Risk Baseline

$$
(s,a)\rightarrow P(success/risk)
$$

### Predictive Risk

$$
(s,a)
\rightarrow
future
\rightarrow
risk
$$

训练：

```text
VLA A + VLA B
```

测试：

```text
unseen heterogeneous VLA C
```

我们真正希望得到：

```text
IID:
Direct Risk ≈ Predictive Risk

Unseen VLA:
Direct Risk ↓
Predictive Risk remains stable
```

如果这条成立：

> **World Model 才不是装饰，而是一个真正提高 policy transfer 的 causal bottleneck。**

这是整篇论文最重要的实验。

---

# 4. 顶会冲刺的“必做 / 强烈建议 / 可选”分层

为了保证可行性，不能所有模块同时做。

---

## Tier A — 必做，缺一项都不建议投主会

1. Same-root multi-VLA counterfactual dataset；
2. 至少 3 个 VLA，其中至少 1 个 held-out architecture；
3. Direct Risk vs Predictive Risk；
4. unseen-VLA zero-shot ranking；
5. risk-driven closed-loop selector；
6. 相对 best-fixed 的显著收益；
7. task-router baseline；
8. candidate identity / shortcut probe；
9. paired statistical evaluation。

---

## Tier B — 强烈建议，显著提高顶会竞争力

1. latent visual world model；
2. within-task comparative advantage；
3. 第二仿真平台；
4. simulator-ID probe；
5. uncertainty-aware selective imagination；
6. failure anticipation / lead-time analysis。

---

## Tier C — 可选增强，不能阻塞主线

1. OPD distillation；
2. contextual bandit；
3. full RL；
4. photorealistic video generation；
5. real robot；
6. 大规模 candidate search。

---

# 5. 项目主架构：保持最简

顶会版主模型优先采用：

```text
Observation Encoder
      │
      ▼
Canonical State z_t
      │
      ├──── candidate a_1 ─── World Model ─── future z^1
      ├──── candidate a_2 ─── World Model ─── future z^2
      ├──── candidate a_3 ─── World Model ─── future z^3
      └──── fallback     ─── World Model ─── future z^f
                                      │
                                      ▼
                                Risk Critic
                                      │
                       risk + uncertainty + recoverability
                                      │
                                      ▼
                            Conservative Selector
```

不优先加入复杂 RL。

---

# 6. Phase 0：冻结当前结果，建立不可回退的实验纪律

## 6.1 Freeze

冻结：

```text
oft_risk_model_v3
current feature pipeline
current lookup result
current action-loop result
current OFT matrix
K3/K5 results
```

禁止为了新实验重新调旧结果。

---

## 6.2 保存为正式 baselines

后续始终保留：

```text
Direct OPD-v3
Task lookup
Best fixed
Random
Oracle
```

特别是：

### Task Lookup

不是最终方法，但必须作为强 baseline。

原因：

> 如果新 selector 只是在重新识别 task/domain，那么它应该无法系统性超过 task-router。

---

## 6.3 数据泄露纪律

测试 unseen VLA 时：

- 不允许 test VLA rollout 参与 risk training；
- 不允许用 test-VLA statistics 做 normalization；
- 不允许 test-VLA-specific threshold；
- 不允许 candidate source ID；
- 不允许 task→policy table。

---

# 7. Phase 1：Current OPD Zero-Shot Falsification

目的：

> 用最小代价快速判断当前 OPD 是否已经具有任何 cross-VLA transfer。

---

## 7.1 Train

保持：

```text
oft_spatial
oft_object
```

完全冻结。

---

## 7.2 Test hierarchy

### Level 1 — Same-family checkpoint shift

```text
oft_goal
oft_10
```

### Level 2 — Architecture shift

优先：

```text
π0-fast
SmolVLA
```

如果接口成本过高，至少完成一个真正异构 architecture。

---

## 7.3 Metrics

必须统一计算：

- AUROC；
- AUPRC；
- Brier score；
- ECE；
- calibration curve；
- pairwise ranking accuracy；
- score distribution；
- score drift relative to train VLA；
- VLA-ID probe。

---

## 7.4 这里不追求闭环

Phase 1 只回答：

> 当前 representation transfer 不 transfer？

不要花时间重新调 threshold。

---

## 7.5 Gate A

### A-PASS

跨 architecture：

- ranking 明显高于随机；
- calibration 没有完全崩；
- score 不只是按 VLA family 分群。

进入 Phase 2，同时保留 direct OPD 作为强 baseline。

### A-FAIL

如果：

```text
AUROC ~ 0.5
pairwise ~ random
VLA-ID probe 很高
```

则正式得到论文 motivation：

> **Direct risk prediction exploits policy-specific shortcuts and fails under policy shift.**

停止 OPD-v3 deployment tuning。

---

# 8. Phase 2：Same-Root Multi-VLA Counterfactual Dataset

这是整个新路线最关键的工程。

---

## 8.1 必须复用已有 capture 基础设施

当前已经完成：

- immutable inference event；
- queue cursor；
- force requery；
- RNG snapshot restore；
- capability mask；
- same-root fork 基础设施。

因此不要重写 capture stack。

只扩展：

> **同一 snapshot 请求多个异构 VLA candidate。**

---

## 8.2 样本结构

每个 root：

```text
Root r
│
├── current state / observation
├── task
├── history
│
├── candidate A from VLA-A
├── candidate B from VLA-B
├── candidate C from VLA-C
└── fallback
```

每个 candidate 分别：

```text
restore same root
matched execution seed
execute H steps
save future trajectory
```

---

## 8.3 Candidate source identity

保留在 metadata：

```text
candidate_source = OpenVLA / π0 / SmolVLA
```

但正式 World Model / Risk Critic 输入：

```text
candidate_source MUST NOT ENTER
```

---

## 8.4 Future trajectory 保存内容

最少：

```text
RGB_t:t+H
proprio_t:t+H
EEF pose
gripper state
success/subgoal state
```

如果 simulator 可获取，teacher/diagnostic 额外保存：

```text
object poses
contacts
collision
object-goal relation
grasp relation
```

这些 simulator privileged state 第一阶段可以用于：

- label 构建；
- oracle analysis；
- world-model MVP；

但最终 cross-platform model 不应强依赖它们。

---

# 9. Phase 2.5：先测 Counterfactual Opportunity，不急着训练模型

对每一个 root，统计：

$$
\Delta_{future}
=
d(
future_i,
future_j
)
$$

以及：

$$
\Delta_{outcome}
$$

目的：

> candidate 虽然 action chunk 很接近，未来是否真的会分叉？

---

## Gate B

必须满足：

### Candidate diversity

candidate 在 canonical action / future state 上有非退化差异。

### Within-state comparative advantage

同一个 root 上：

```text
不同 candidate 的 future quality / outcome
```

存在足够异质。

如果多数 root：

```text
所有 candidate 未来几乎一样
```

不要训练 World Model selector。

先换 candidate provider / decision boundary。

---

# 10. Phase 3：Oracle Future Risk Upper Bound

这是非常重要的可行性保护。

在训练任何复杂 World Model 之前先问：

> **如果我已经知道真实未来，我能不能准确判断哪个 candidate 更好？**

---

## 10.1 Oracle input

使用真实：

$$
\tau^{real}_{t+1:t+H}
$$

---

## 10.2 Risk target

第一版不要设计太复杂。

建议只保留三个核心维度：

### Progress

$$
R_{progress}
$$

candidate 是否让任务状态更接近目标。

### Drift / Failure Attraction

$$
R_{drift}
$$

candidate 是否产生持续增长偏移。

### Recoverability

$$
R_{recover}
$$

candidate 执行后由统一 fallback/reference controller 接管还能否成功。

---

## 10.3 最终 scalar

简单线性组合即可：

$$
Score_i
=
w_p Progress_i
-
w_d Drift_i
+
w_r Recoverability_i
$$

权重只在 train split 确定。

---

## 10.4 Gate C — 最关键早期止损

如果：

> **Ground-truth future 都无法可靠排序 candidate**

则：

- 不训练大 World Model；
- 不做 video generation；
- 优先重定义 decision boundary / risk target / candidate set。

只有 Oracle Future ranking 明显成立，才进入 Phase 4。

---

# 11. Phase 4：World Model MVP —— 先追求可行，不追求视频质量

顶会冲刺版不应该一开始训练大视频生成模型。

---

## 11.1 MVP 输入

使用 simulator teacher state：

```text
EEF-object relative pose
object-goal relative pose
gripper state
canonical action chunk
```

---

## 11.2 MVP 模型

优先：

```text
MLP
Temporal MLP
Small Transformer
```

不使用大型 diffusion/video model。

---

## 11.3 MVP 输出

预测 H-step 后的：

```text
relative object pose
relative EEF pose
grasp/contact state
task progress proxy
```

不要求每一帧 photorealistic。

---

## 11.4 MVP 目的

只回答一个问题：

$$
\boxed{
Does modeling future consequence improve candidate ranking?
}
$$

---

## 11.5 三条 baseline

### B0 — Action Statistics

当前 raw action / chunk statistic 风险模型。

### B1 — Direct State-Action Risk

$$
(s,a)\rightarrow risk
$$

### B2 — Future-Bottleneck Risk

$$
(s,a)\rightarrow \hat s_{future}\rightarrow risk
$$

---

## Gate D

只有：

$$
B2 > B1
$$

特别是在：

```text
held-out VLA
```

上出现稳定优势，才继续升级视觉 World Model。

如果 B2 只在 IID 更好：

> 说明 World Model 可能只是增加容量，不足以支撑顶会主 claim。

---

# 12. Phase 5：Latent Visual World Model —— 正式 CVPR 版本

如果 Phase 4 PASS，再进入视觉版。

---

## 12.1 不以 RGB 重建质量作为第一目标

正式模型优先：

```text
RGB / multi-view
      ↓
frozen or lightly trained visual encoder
      ↓
latent z_t
      +
canonical action
      ↓
action-conditioned latent dynamics
      ↓
z_hat_{t+1:t+H}
      ↓
risk critic
```

---

## 12.2 为什么优先 latent

比完整 video generation：

- 训练成本更低；
- latency 更低；
- 更容易 multi-candidate rollout；
- 不需要预测 texture/lighting；
- 更容易做 cross-domain feature alignment；
- 更适合作为 risk bottleneck。

---

## 12.3 视频生成只作为可解释性增强

如果资源允许：

```text
latent future -> decoder -> future visualization
```

用于 paper figure / qualitative。

但不要让视频 fidelity 成为主方法的工程瓶颈。

---

# 13. Phase 6：专门验证“微小偏移 → 错误累积”

这是非常适合 CVPR 的核心实验。

---

## 13.1 Controlled perturbation

对正确 candidate：

$$
a
$$

构造微小扰动：

$$
a+\epsilon
$$

控制：

$$
\|\epsilon\|
$$

逐级变化。

---

## 13.2 记录 future divergence

$$
D_h
=
d(
s_{t+h}^{a},
s_{t+h}^{a+\epsilon}
)
$$

研究：

$$
D_h
$$

是否随 horizon 放大。

---

## 13.3 Failure anticipation lead time

定义：

$$
LeadTime
=
t_{actual\ failure}
-
t_{risk\ alarm}
$$

希望证明：

> Predictive Risk 可以在真正失败发生前若干 decision boundaries 发现危险。

---

## 13.4 与 direct verifier 比较

测试：

```text
Direct Risk
Observation-only anomaly
Action statistics
Predictive Future Risk
```

最有价值的结果不是最终 accuracy，而是：

> **Predictive Future Risk 更早识别 compounding error。**

---

# 14. Phase 7：Unseen-VLA Zero-Shot Risk Evaluation

这是论文最关键实验之一。

---

## 14.1 正式 Split

至少三个 candidate policy family。

例如：

```text
Train:
OFT/OpenVLA family
+ π0 family

Test:
SmolVLA
```

然后 rotate：

```text
Train A+B -> Test C
Train A+C -> Test B
Train B+C -> Test A
```

形成：

# Leave-One-VLA-Out (LOVO)

---

## 14.2 测试时严格 frozen

测试 VLA：

```text
0 risk training samples
0 calibration samples
0 task lookup
0 VLA-specific normalization
0 model identity
```

---

## 14.3 主要指标

### Prediction

- AUROC；
- AUPRC；
- calibration；
- Brier；
- pairwise ranking。

### Transfer

定义：

$$
TransferRetention
=
\frac{Metric_{unseenVLA}-Metric_{random}}
{Metric_{seenVLA}-Metric_{random}}
$$

用于比较 Direct Risk vs Predictive Risk。

---

## 14.4 顶会关键图

最重要的一张图建议：

```text
              Seen VLA       Unseen VLA

Direct Risk      high            collapse
Predictive Risk  high            retained
```

如果这个 pattern 很强：

> 论文核心成立。

---

# 15. Phase 8：Risk-Driven Closed-Loop Selector

只有 unseen-VLA predictive ranking 成立后再做闭环。

---

## 15.1 Selector 必须简单

不要让 reviewer 怀疑收益来自复杂 selector。

推荐：

$$
LCB_i=\mu_i-\beta\sigma_i
$$

选择：

$$
i^*=\arg\max_i LCB_i
$$

---

## 15.2 Conservative switching

只有：

$$
LCB_{new}>
UCB_{current}+\delta
$$

才切换。

否则：

```text
continue
```

---

## 15.3 Fallback

如果：

```text
all primary candidates risky
AND fallback low-risk/recoverable
```

则 fallback。

如果：

```text
all candidates + fallback high-risk
```

则 abort。

---

## 15.4 Hysteresis

switch 后：

```text
minimum dwell K
```

避免 oscillation。

---

# 16. Closed-Loop Baselines：一定要强

主表至少有：

1. each VLA only；
2. best fixed；
3. random router；
4. task-router；
5. current Direct OPD；
6. V-GPS-style value/reranking baseline（尽可能复现相同协议）；
7. direct verifier；
8. Predictive Risk Selector；
9. oracle future selector；
10. oracle best candidate。

---

# 17. 顶会成功标准

原 RASE endpoint 保留：

$$
Success(RASE)-Success(best-fixed)
$$

最低科学标准：

- paired bootstrap 95% CI lower bound > 0；
- absolute gain ≥ 3pp。

但是内部“顶会 ready”标准建议更严格：

### Strong

$$
+5pp
$$

以上，且在多个 held-out VLA 上稳定。

### Very Strong

不仅平均成功率提升，而且：

- failure detection 更早；
- fallback 次数合理；
- unseen-VLA transfer retention 明显高于 direct risk。

---

# 18. 必须新增：Within-Task Comparative Advantage

OFT 当前最强异质主要是：

```text
task-level
```

这容易被 task-router 解决。

真正 real-time selector 更需要：

```text
same task:
state 1 -> A
state 2 -> B
state 3 -> fallback
```

---

## 18.1 定义

$$
H_{within}
=
P(
argmax_i V(s_t,a_i)
\ changes\ inside\ the\ same\ task
)
$$

---

## 18.2 Gate E

如果：

$$
H_{within}\approx0
$$

这个 benchmark 只用于：

```text
routing sanity check
```

不要把它当实时 selector 主 benchmark。

---

# 19. Phase 9：跨平台验证 —— 但用真正不同平台

原规划里“robosuite native task”作为第二平台说服力不足。

LIBERO 与 robosuite / MuJoCo 高度同源，因此不能强支撑：

> simulator-independent transfer。

---

## 19.1 第二平台选择原则

必须满足：

1. 与 LIBERO 不同 simulator / physics stack；
2. 支持 RGB；
3. 支持 restore/reset 或可复制状态；
4. 至少能运行 2 个 candidate policy；
5. action 可以映射到 canonical representation；
6. task 数量不需要大，但要能形成 failure/success heterogeneity。

---

## 19.2 优先候选：ManiSkill3

理由：

- SAPIEN stack，与 MuJoCo/robosuite 不同；
- GPU parallel simulation；
- contact-rich manipulation；
- 任务和视觉接口丰富。

正式采用前先做兼容性 probe：

```text
VLA observation preprocessing
action mapping
candidate inference
state restore
```

如果 VLA adapter 成本过高，则不要强行迁移。

---

## 19.3 RoboTwin 2.0

可以作为备选。

优点：

- 当前 VLA 社区使用较多；
- domain randomization 强；
- benchmark 较新。

风险：

- 双臂；
- action space / embodiment 更复杂；
- 可能显著增加工程成本。

因此只在已有 compatible VLA pipeline 时选择。

---

# 20. 跨平台正式实验

最强形式：

```text
Train:
Platform A = LIBERO
VLA A/B

Test:
Platform B
unseen VLA C
unseen tasks
```

Risk model frozen。

---

## 20.1 如果直接跨平台太难

采用两级证据：

### Cross-Policy First

```text
LIBERO:
train VLA A/B
test VLA C
```

### Cross-Platform Second

```text
train Platform A + small Platform B train split
test held-out tasks/VLA on Platform B
```

第二个只能作为 intermediate claim。

---

# 21. Cross-Platform Feature Design

---

## 21.1 Canonical Action

每个平台实现：

$$
a^{canonical}
=
T_{platform}(a^{native})
$$

核心量：

```text
Δx, Δy, Δz normalized
canonical rotation delta
gripper intent
velocity
acceleration
trajectory curvature
chunk smoothness
```

---

## 21.2 Canonical State

优先：

```text
vision embedding
EEF-relative geometry
gripper state
goal semantic embedding
history
```

不要把：

```text
LIBERO-specific 8d proprio
```

作为最终主 feature。

---

# 22. Identity Shortcut Audits

这是顶会版必须有的“诚实性”实验。

---

## 22.1 VLA-ID Probe

$$
z\rightarrow VLA-ID
$$

比较：

```text
Direct Risk representation
Predictive Risk representation
```

希望：

> predictive representation 更难识别 VLA ID，同时保留 risk signal。

---

## 22.2 Simulator-ID Probe

第二平台后：

$$
z\rightarrow simulator-ID
$$

如果 simulator ID 几乎完美可预测：

> 不能直接声称 platform-invariant。

---

## 22.3 Task-ID Probe

如果 task ID 可以极高准确率恢复：

需要特别检查：

> risk gain 是否来自 task shortcut。

---

# 23. OPD 的最终位置

只有 World Model 路线已成立，再做 OPD。

---

## 23.1 Teacher

$$
Teacher(s,a)
=
Risk(
WorldModel(s,a)
)
$$

---

## 23.2 Student

$$
R_{OPD}(s,a)
$$

---

## 23.3 最有价值的部署方式

```text
candidate
   │
   ▼
OPD fast risk
   │
confident?
 /       \
yes       no
 |         |
select   World Model
          imagination
             |
          refined risk
```

这可以形成：

> **Adaptive imagination**

即只在高不确定性决策点调用昂贵 World Model。

---

# 24. Bandit / RL：从主线移到扩展

---

## 24.1 Contextual Bandit 什么时候值得加

如果：

```text
zero-shot risk 有一定用
但 unseen VLA calibration 有残余偏移
```

再使用：

$$
Q^{online}
=
Q^{predictive\ prior}
+
\Delta^{bandit}
$$

---

## 24.2 论文里必须拆开

### Episode 0

zero-shot。

### Episode > 0

adaptation。

绝不混淆。

---

## 24.3 Full RL 触发条件

只有：

- selector choice 改变未来长期 opportunity；
- fallback/switch 有明显成本；
- query World Model 有成本；
- delayed reward 明显；
- contextual bandit 不足；

才进入 RL。

否则不做。

---

# 25. 顶会核心实验矩阵

| Exp               | Same-root | Held-out Task | Held-out VLA | Held-out Platform | World Future | Closed-loop |
| ----------------- | --------: | ------------: | -----------: | ----------------: | -----------: | ----------: |
| Current OPD       |      部分 |          部分 |            ✗ |                 ✗ |            ✗ |           ✓ |
| E1 Direct Risk    |         ✓ |             ✓ |            ✓ |                 ✗ |            ✗ |        部分 |
| E2 Oracle Future  |         ✓ |             ✓ |            ✓ |                 ✗ |           GT |           ✗ |
| E3 WM-MVP         |         ✓ |             ✓ |            ✓ |                 ✗ |        state |           ✓ |
| E4 Latent WM      |         ✓ |             ✓ |            ✓ |                 ✗ |       latent |           ✓ |
| E5 Cross-platform |         ✓ |             ✓ |            ✓ |                 ✓ |       latent |           ✓ |
| E6 OPD distill    |         ✓ |             ✓ |            ✓ |              可选 |    distilled |           ✓ |
| E7 Bandit         |         ✓ |             ✓ |            ✓ |              可选 |        prior |     ✓+adapt |

---

# 26. Ablation Matrix

必须回答 reviewer 的每个疑问。

---

## A. World Model 是否真的需要？

```text
action only
state + action direct risk
state + action -> future -> risk
```

---

## B. Future horizon

```text
H = short
H = medium
H = long
```

看：

- ranking；
- prediction error；
- anticipation lead time；
- compute。

---

## C. Future representation

```text
privileged simulator state
object-centric
latent visual
RGB video
```

---

## D. Risk definition

```text
episode success
progress
drift
recoverability
combined
```

---

## E. Identity features

```text
with task text
without task text

with candidate prior
without candidate prior

with VLA ID
without VLA ID
```

VLA-ID 只作为 oracle/diagnostic，不进入正式方法。

---

## F. Selector

```text
absolute threshold
relative score
uncertainty-aware LCB
LCB + hysteresis
```

---

# 27. 统计协议

避免再次出现“看起来高，但不稳”。

---

## 27.1 Split unit

正式 split 至少做到：

```text
task-held-out
VLA-held-out
```

不能只做 random row split。

---

## 27.2 Bootstrap

以：

```text
task
```

或：

```text
task × seed
```

为 bootstrap unit。

不要按 transition 行 bootstrap。

---

## 27.3 Closed-loop

使用 paired evaluation：

```text
same task
matched episode seeds
same initial state distribution
```

---

## 27.4 报告

必须有：

- mean；
- 95% CI；
- absolute improvement；
- number of tasks；
- number of episodes；
- number of seeds。

---

# 28. 工程可行性优先策略

为了提高成功率，按“最小新增工程”推进。

---

## 28.1 最大化复用

直接复用：

- existing snapshot restore；
- immutable inference event；
- candidate capture；
- OFT checkpoints；
- π0-fast checkpoint；
- SmolVLA checkpoint；
- rollout evaluator；
- bootstrap pipeline；
- provenance/hash infrastructure。

---

## 28.2 不先做的东西

暂停：

- 大视频生成模型；
- 大规模 RL；
- 大量新 VLA 下载；
- 大规模 task expansion；
- 多个平台同时迁移；
- 复杂 learned selector。

---

## 28.3 每阶段只回答一个问题

```text
Phase 1:
current risk transfer?

Phase 2:
same-state candidates really diverge?

Phase 3:
true future contains enough selection signal?

Phase 4:
learned future preserves that signal?

Phase 5:
does it transfer to unseen VLA?

Phase 8:
does it improve closed-loop success?

Phase 9:
does it survive platform shift?
```

---

# 29. Kill Criteria：防止继续在错误路线烧算力

---

## Kill 1 — Current OPD

如果跨 architecture：

```text
AUROC ~ 0.5
```

停止调 v3。

---

## Kill 2 — Counterfactual data

如果同一 root：

```text
future divergence ≈ 0
```

换 candidate provider / boundary。

---

## Kill 3 — Oracle future

如果真实 future 无法有效 ranking：

> 停 World Model，重定义 target。

---

## Kill 4 — World Model novelty

如果：

```text
Predictive Risk
<=
Direct Risk
```

在 unseen-VLA 上无优势：

> 不把 World Model 作为主论文。

---

## Kill 5 — Selector utility

如果 oracle headroom <5pp：

> 不做完整 closed-loop selector。

---

## Kill 6 — Task router trap

如果：

```text
task-router ≈ oracle
```

且：

$$
H_{within}\approx0
$$

换 benchmark。

---

## Kill 7 — Platform integration

如果第二平台 VLA adapter 工程量过高、无法形成至少 2 candidate policy：

> 不为“跨平台”强行拖死核心论文。

先完成 strong cross-VLA paper，再扩平台。

---

# 30. 顶会 Submission-Ready Gate

只有满足下面 **Core Gate** 才认为主线达到投稿级别。

---

## Core Gate 1 — Scientific novelty

必须证明：

$$
PredictiveRisk_{unseenVLA}
>
DirectRisk_{unseenVLA}
$$

不是只在 IID 提升。

---

## Core Gate 2 — Zero-shot

至少一个：

```text
architecture-held-out VLA
```

完全 zero-shot 有稳定 risk ranking。

---

## Core Gate 3 — Selector utility

在 held-out VLA/task：

$$
Success(RASE)
>
Success(best-fixed)
$$

且 95% CI 支持。

---

## Core Gate 4 — Router control

RASE 必须在适合 real-time selector 的 benchmark 上：

```text
> task-router
```

或者至少证明收益来自 within-task candidate changes。

---

## Core Gate 5 — Mechanism evidence

必须有：

```text
micro deviation
-> future divergence
-> risk rise
-> earlier intervention
```

这种可解释机制证据。

---

# 31. Strong Top-Conference Gate

如果再满足以下任一项，竞争力明显提高：

### Option A

第二 simulator / physics stack zero-shot transfer。

### Option B

少量 real-robot validation。

### Option C

非常强的 held-out architecture generalization + large-scale benchmark。

### Option D

World Model selective imagination 带来显著 latency/compute advantage。

---

# 32. Venue Strategy

---

## CVPR 更适合的版本

如果核心贡献是：

- visual latent future；
- action-conditioned visual prediction；
- predictive representation；
- future visualization；
- cross-domain visual transfer；

优先 CVPR。

---

## CoRL / RSS 更适合的版本

如果最终 strongest contribution 是：

- arbitration；
- controller/selector；
- counterfactual execution protocol；
- online adaptation；
- robot deployment；

而视觉模型本身较简单，则 CoRL/RSS 可能更自然。

---

# 33. 推荐 Paper Story

## Title direction

**RASE: Predicting Action Consequences for Zero-Shot Arbitration Across Vision-Language-Action Policies**

副标题式核心句：

> **Learning Action Consequences, Not Policy Identities**

---

## Abstract story

### Problem

Frozen VLAs have heterogeneous failure modes; existing confidence/verifier/value methods can exploit policy-specific correlations and may not transfer to unseen policies.

### Observation

Current direct risk predictor is strong IID but fails to establish policy transfer; task lookup can achieve high success without learning action risk.

### Key idea

At the same physical state, the environment transition caused by an action is more policy-agnostic than the policy’s eventual task success.

### Method

Predict each candidate's future consequence with an action-conditioned world model, score deviation/recoverability risk, then conservatively arbitrate among heterogeneous frozen VLAs.

### Result

Predictive risk retains ranking under unseen-VLA shift and improves closed-loop success relative to best-fixed / direct verifier / task-router.

---

# 34. 推荐 Figure 1

一张图必须把 novelty 讲明白：

```text
                         SAME PHYSICAL STATE
                               |
         ------------------------------------------------
         |                     |                        |
     OpenVLA a_A            π0 a_B                SmolVLA a_C
         |                     |                        |
         v                     v                        v
   imagined future A     imagined future B       imagined future C
         |                     |                        |
   slight deviation        stable progress        direct failure
   compounds later             |                        |
         |                     |                        |
      risk 0.72             risk 0.14                risk 0.93
         \_____________________|________________________/
                               |
                         SELECT π0 candidate
```

旁边明确标：

```text
No policy ID
No task lookup
Unseen VLA
```

---

# 35. 推荐 Figure 2

最关键定量图：

```text
Risk ranking performance

                 Seen VLA      Unseen VLA
Direct Risk        high           ↓↓↓
Predictive Risk    high           stable
```

这张图决定 paper 的核心是否成立。

---

# 36. 推荐 Figure 3

微小偏移：

```text
epsilon action perturbation
        ↓
future divergence increases with horizon
        ↓
predictive risk rises before actual failure
```

展示：

- future frames / latent trajectory；
- drift curve；
- risk curve；
- intervention time。

---

# 37. 推荐主结果表

| Method                   | Seen VLA |  Unseen VLA | Task-held-out | Closed-loop | Policy ID? |
| ------------------------ | -------: | ----------: | ------------: | ----------: | ---------: |
| Best fixed               |        — |           — |             ✓ |    baseline |          — |
| Task router              |     high | low/unknown |             ✓ |           ✓ |   indirect |
| Direct OPD               |     high |           ? |             ✓ |           ✓ |         no |
| Value verifier           |     high |           ? |             ✓ |           ✓ |         no |
| **RASE Predictive Risk** |     high |    **high** |             ✓ |    **best** |     **no** |
| Oracle                   |    upper |       upper |             — |       upper |          — |

---

# 38. 8 个最重要的实验问题

最终论文必须能回答：

1. 当前 direct risk 是否跨 VLA？
2. 同一状态下不同 VLA candidate 是否真的产生不同 future？
3. ground-truth future 是否包含可排序的 risk signal？
4. learned World Model 是否保留这个 signal？
5. predictive future 是否比 direct risk 更能跨 unseen VLA？
6. predictive risk 是否能提前发现微小偏移累积？
7. risk-driven selector 是否超过 best-fixed 和 task-router？
8. 是否能在第二平台或更强 domain shift 下保持收益？

---

# 39. 推荐近期执行优先级

## P0 — 马上做

- Freeze OPD-v3；
- oft_goal zero-shot；
- π0-fast zero-shot；
- SmolVLA zero-shot（接口允许则做）；
- VLA-ID probe；
- current direct-risk transfer report。

---

## P1 — 紧接着

- same-root multi-VLA capture；
- future trajectory logging；
- future-divergence analysis；
- within-root comparative advantage analysis。

---

## P2

- Oracle Future Risk；
- recoverability label；
- drift label；
- ranking upper bound。

---

## P3

- state/object World Model MVP；
- Direct Risk vs Future-Bottleneck Risk；
- held-out VLA comparison。

---

## P4

只有 P3 PASS：

- latent visual world model；
- micro-deviation anticipation；
- uncertainty。

---

## P5

- zero-shot closed-loop selector；
- best-fixed / task-router / verifier baselines。

---

## P6

核心 paper PASS 后：

- distinct second simulator；
- OPD distillation；
- bandit；
- real robot。

---

# 40. 最终决策树

```text
Current direct OPD cross-VLA?
        |
   +----+----+
   |         |
  yes       no
   |         |
baseline   motivation
   |         |
   +----+----+
        |
Same-root future has candidate signal?
        |
   +----+----+
   |         |
  yes       no
   |         |
Oracle     change
future     boundary/provider
risk
   |
Oracle future can rank?
   |
 +--+--+
 |     |
yes    no
 |     |
WM     redefine risk
MVP
 |
Predictive Risk > Direct Risk on unseen VLA?
 |
 +-------+-------+
 |               |
YES             NO
 |               |
CORE NOVELTY    do not force
 |              World Model paper
 |
closed-loop > best fixed/task router?
 |
 +-------+-------+
 |               |
YES             NO
 |               |
TOP-CONF CORE   selector/risk redesign
 |
second platform / real robot
 |
STRONGER PAPER
```

---

# 41. 最终项目定位

RASE 不再定义成：

> “找出某个任务最适合哪个 VLA。”

也不定义成：

> “给单个 VLA 做 runtime safety verification。”

最终定义：

> **RASE is a policy-agnostic predictive arbitration layer for heterogeneous frozen VLAs. At the same physical state, it imagines the future consequences of candidate actions, estimates deviation and recoverability risk without using policy identity, and conservatively selects whether to continue, switch, fall back, or abort. Its central goal is to transfer risk across unseen VLA architectures rather than memorize which policy is strong on which task.**

中文：

> **RASE 是建立在异构冻结 VLA 之上的 policy-agnostic 预测式仲裁层。它在同一物理状态下想象不同 candidate action 的未来后果，判断微小偏移是否会持续累积、是否会进入不可恢复状态，并在不使用 VLA 身份和任务查询表的情况下保守地选择 continue / switch / fallback / abort。其核心不是记忆哪个模型在哪个任务更强，而是让 action-consequence risk 能够迁移到未见 VLA。**

---

# 42. 一句话顶会标准

> **真正值得冲顶会的结果不是“World Model 能预测未来”，也不是“selector 在 LIBERO 提高了成功率”，而是：一个在已知 VLA 上学到的 action-consequence model，在完全未见的异构 VLA 上仍然能正确预测候选动作未来风险，并依靠这种 transferable risk 在闭环中超过 best-fixed 与 task-router。**

---

# 43. 当前最重要的唯一 Milestone

$$
\boxed{
\textbf{Predictive future risk must outperform direct risk under unseen-VLA shift.}
}
$$

在这条成立之前：

- 不扩 RL；
- 不扩大视频模型；
- 不大规模迁平台；
- 不继续调 lookup；
- 不继续追 IID accuracy。

在这条成立之后：

> 项目才正式进入“顶会主线成立”的阶段。

---

# 44. Related Work Guardrail

后续每次改方法前，都需要重新检查是否只是复现以下已有方向：

- Nakamoto et al., **V-GPS**, arXiv:2410.13816
- Kwok et al., **RoboMonkey**, arXiv:2506.17811
- Dai et al., **RoVer**, arXiv:2510.10975
- Sun et al., **Pre-VLA**, arXiv:2605.22446
- Liu et al., **CheckVLA**, arXiv:2607.26789
- Lin et al., **World Pilot**, arXiv:2606.12403
- Yang et al., **RISE**, arXiv:2602.11075

RASE 必须始终能够回答：

> **“我们比 single-policy verifier / value reranker / action-conditioned runtime checker 多证明了什么？”**

标准答案必须来自实验：

> **Same-state, cross-policy counterfactual futures + held-out heterogeneous VLA zero-shot arbitration + policy-identity-free risk transfer。**

---

# 45. Benchmark Guardrail

近期 benchmark audit 工作已经强调 robot manipulation benchmark 容易出现：

- shortcut solvability；
- statistical significance 不足；
- benchmark overfitting；
- data-source dependence。

因此 RASE 主结果禁止只依赖一个 LIBERO average success number。

至少需要：

```text
task-held-out
VLA-held-out
multiple seeds
paired bootstrap
shortcut probe
task-router baseline
```

顶会增强版再增加：

```text
held-out simulator / real robot
```

---

# 46. 最终执行原则

每一个新增实验都必须回答以下至少一个问题：

```text
Does it prove transfer?
Does it prove action consequence?
Does it prove selector utility?
Does it rule out task/policy shortcut?
Does it strengthen cross-platform evidence?
```

如果答案全部是否：

> **后续项目不再以“实验数量”推进，而以“顶会核心 Claim 被逐个证实”推进。**
