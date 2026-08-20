# RASE + RL 与公开数据集可行性方案

**日期：** 2026-08-13  
**项目：** RASE — Risk-Aware Switching and Execution for Multi-VLA Policies  
**目标：** 判断强化学习是否适合 RASE、应该在哪一层使用、采用什么算法，以及现有公开机器人数据集能否有效降低数据采集成本并提升跨 VLA / 跨 benchmark 泛化。

---

# 0. 最终结论

## 0.1 RL 适合 RASE，但不应该替代风险模型

RASE 非常适合引入强化学习，但最合适的位置不是：

> 直接对整个 VLA 做端到端 PPO / GRPO。

最合适的位置是：

> **保持 source VLA 冻结，用监督学习训练实时风险 / urgency / recoverability 模型，再用 offline RL 学习“面对当前风险应该选择哪一种 correction operator”。**

最终推荐结构：

```text
Frozen source VLA
       |
       v
source action proposal
       |
       v
+----------------------------------+
|  RASE Stochastic Risk Model      |
|                                  |
|  - local source risk             |
|  - intervention urgency          |
|  - recoverability distribution   |
|  - uncertainty / OOD             |
+----------------------------------+
       |
       v
admissible correction operators
       |
       v
+----------------------------------+
| RASE Correction Policy           |
|                                  |
| Stage 1: supervised advantage    |
| Stage 2: contextual bandit       |
| Stage 3: offline RL              |
| Stage 4: optional online RL      |
+----------------------------------+
       |
       +------ continue
       +------ shorten + requery
       +------ resample
       +------ replan
       +------ fallback takeover
       +------ safe abort
```

因此推荐论文定义为：

> **RASE = calibrated stochastic risk model + conservative learned correction policy.**

RL 是决策层，而不是概率风险层。

---

# 1. 为什么 RASE 适合 RL

RASE 本质上已经是一个 sequential decision problem。

当前每个时刻需要决定：

```text
继续 source？
重新 query？
缩短 chunk？
resample？
replan？
切换 corrective policy？
停止？
```

一个操作不仅影响当前成功概率，还会改变未来状态。

例如：

```text
t = 12
   |
   +-- continue source
   |       ↓
   |    t = 20
   |    fallback 已经救不回来
   |
   +-- fallback now
           ↓
        task success
```

所以真正优化的不是：

\[
P(\mathrm{success}\mid s_t,c_t)
\]

而是长期价值：

\[
Q(s_t,c_t)
=
\mathbb E
\left[
\sum_{k=0}^{T-t}
\gamma^k r_{t+k}
\mid
s_t,c_t
\right].
\]

这正是 RL 有价值的地方。

---

# 2. 为什么不建议直接 RL 整个 VLA

## 2.1 会破坏主要 scientific attribution

如果同时：

- 修改 source VLA；
- 训练 risk controller；
- 使用 RL；
- 使用 stronger fallback；

最终 success 提升以后 reviewer 会问：

> 提升到底来自 RASE，还是 VLA 本身经过 RL 以后变强了？

这会削弱论文故事。

RASE 最干净的实验应该保持：

```text
VLA weights = frozen
```

RASE 只改变：

```text
何时继续
何时重新调用
何时 resample
何时 replan
何时 fallback
```

---

## 2.2 VLA online RL 的训练成本和稳定性更高

公开工作已经表明 VLA + RL 可行，但直接在线优化大型 VLA 会带来：

- rollout 计算量大；
- renderer / inference / training 三方资源竞争；
- policy update 不稳定；
- catastrophic forgetting；
- benchmark-specific reward engineering；
- 多 VLA 重复训练成本高。

例如 RLinf-VLA 专门构建了 VLA+RL 系统来解决 rendering、inference 和 training 的资源调度问题，并支持 OpenVLA/OpenVLA-OFT、PPO/GRPO、LIBERO/ManiSkill。

RASE 没必要首先承担这个复杂度。

---

# 3. 为什么风险预测本身不应该完全交给 RL

你仍然需要明确输出：

\[
P(\mathrm{failure})
\]

\[
P(\mathrm{recoverable})
\]

\[
P(T_{\mathrm{loss}}\le t+H)
\]

以及：

- calibration；
- epistemic uncertainty；
- OOD support；
- LCB/UCB。

RL 的：

\[
Q(s,a)
\]

不是 calibrated failure probability。

例如：

```text
Q(fallback) = 0.72
```

不能解释成：

```text
fallback success probability = 72%
```

因为 Q 还包含：

- future rewards；
- intervention cost；
- latency；
- downstream policy behavior；
- discount。

所以最好明确分工。

### Risk model

回答：

> 会不会出问题？  
> 还剩多少恢复机会？  
> prediction 有多可靠？

### RL correction policy

回答：

> 既然现在处于这个状态，采取什么干预的长期价值最大？

---

# 4. RASE 更接近 SMDP，而不是普通一步 MDP

不同 correction operator 持续时间不同：

```text
continue source       = 8 action steps
shorten + requery     = 2~4 steps
resample              = extra inference + H actions
replan                = planner latency + variable duration
fallback takeover     = potentially until episode end
```

所以 action 不是固定一步 primitive action，而是 temporal abstraction / option。

更严格地说，RASE 是一个 **Semi-Markov Decision Process (SMDP)**。

定义：

\[
\mathcal M
=
(\mathcal S,\mathcal C,P,R,\gamma)
\]

其中：

- \(s_t\)：当前历史状态；
- \(c_t\)：correction operator；
- \(\Delta t_t\)：operator 实际持续时间；
- \(s_{t+\Delta t}\)：operator 完成以后下一 decision state。

Bellman target 应使用：

\[
Q(s_t,c_t)
=
r_t+
\gamma^{\Delta t_t}
V(s_{t+\Delta t_t}).
\]

这一点很适合：

- action chunk；
- resample；
- requery；
- persistent fallback。

---

# 5. 推荐 RL State

不要直接把 raw simulator state 输入 RL。

RL policy 的输入应该严格 deployable：

\[
z_t =
f_\theta
(
o_{t-L:t},
q_{t-L:t},
a^{src}_{t:t+H},
task,
d_{policy},
d_{robot}
)
\]

再拼接 risk outputs：

```python
RLState = {
    "risk_embedding": z_t,
    "p_source_risk": ...,
    "risk_ucb": ...,
    "urgency": ...,
    "recoverability_lcb": ...,
    "ood_score": ...,
    "elapsed_time": ...,
    "controller_state": ...,
    "previous_operator": ...,
    "operator_cost_estimates": ...,
}
```

建议：

> **RL 不重新从 raw pixels 独立学一遍 representation。**

风险 backbone 和 RL critic 共享或者冻结同一 representation。

这能减少 sample complexity。

---

# 6. 推荐 RL Action Space

第一版必须保持小。

## v1

\[
\mathcal C=
\{
CONTINUE,
SHORTEN\_REQUERY,
RESAMPLE,
FALLBACK,
ABORT
\}
\]

也就是 5 个动作。

## v2

如果 v1 已证明 state-dependent selector 有价值，再增加：

```text
REPLAN
HAND_BACK
```

### 为什么不要第一版就 10–20 个 correction

Offline RL 最怕数据中很多 state-action pair 没 coverage。

如果每个状态只有：

```text
continue
fallback
```

有数据，却让 RL 输出：

```text
7 种 replan
5 种 resample
```

Q-function 很容易对 OOD action 产生虚假高价值。

因此 action space 扩张必须跟数据 coverage 同步。

---

# 7. Reward 怎么设计

建议将奖励拆成明确的可解释组件：

\[
r_t =
r_{\mathrm{task}}
-
\lambda_h c_{\mathrm{harm}}
-
\lambda_q c_{\mathrm{query}}
-
\lambda_f c_{\mathrm{fallback}}
-
\lambda_l c_{\mathrm{latency}}
-
\lambda_i c_{\mathrm{intervention}}.
\]

## 7.1 Task success

terminal：

```text
success = +1
failure = 0
```

或者：

```text
success = +10
failure = 0
```

只要所有 benchmark 最终 normalize 即可。

## 7.2 Harm

例如：

- collision；
- object drop；
- irreversible wrong placement；
- unsafe workspace excursion。

赋予强负奖励：

\[
-\lambda_h.
\]

## 7.3 Extra source query

每多调用一次 source：

\[
-\lambda_q.
\]

## 7.4 Resample cost

如果生成 \(M\) 个候选：

\[
C_{\mathrm{resample}}
\propto M.
\]

不能把这些 inference 当免费。

## 7.5 Fallback cost

按：

```text
fallback action-selection calls
fallback executed steps
teacher compute
```

计费。

## 7.6 Latency

\[
-\lambda_l T_{\mathrm{wallclock}}.
\]

这对“实时控制层”的 claim 很重要。

---

# 8. 不建议一开始使用 dense task reward

很多 simulator 可以访问：

- object distance；
- exact pose；
- task progress；
- simulator state。

这些可以用于训练 reward，但如果每个 benchmark 都写一套 highly engineered reward，reviewer 会质疑方法是否真正通用。

因此主版本推荐：

```text
task terminal success
+
generic intervention cost
+
generic safety event
```

privileged progress reward 可以作为 simulator-training ablation，而不是方法必要条件。

---

# 9. Safety：推荐 Shielded RL

RL policy 不应可以任意选择所有 operator。

先通过 calibrated risk model 做 admissibility mask：

```text
if OOD too high:
    disable handback

if fallback unsupported:
    disable fallback

if source risk UCB extremely high:
    disable long continue

if candidate action outside action support:
    disable candidate
```

于是 RL 只在：

\[
\mathcal C_{\mathrm{safe}}(s_t)
\subseteq
\mathcal C
\]

中选择。

最终：

\[
c_t^*
=
\arg\max_{c\in\mathcal C_{\mathrm{safe}}}
Q(s_t,c).
\]

这比完全 unconstrained RL 更符合 RASE 的风险控制定位。

---

# 10. 应该用什么 RL 算法？

推荐顺序如下。

---

# 11. Level 0：Supervised Advantage Regression

**第一优先级。**

严格来说它不是 RL，但必须作为 RL 第一 baseline。

你拥有同一 snapshot 的多个 counterfactual branches：

```text
same state
   |
   + continue -> success/failure
   + resample -> success/failure
   + fallback -> success/failure
```

可以直接监督：

\[
Q_{\mathrm{MC}}(s,c)
=
\text{empirical return of branch }c.
\]

训练：

\[
L_Q
=
(Q_\theta(s,c)-G_c)^2.
\]

如果有 stochastic repeats：

\[
G_c
=
E[R\mid s,c].
\]

### 为什么必须先做

如果连直接监督 counterfactual return 都学不出来：

> RL 不可能神奇地解决 observability 问题。

所以这是 RL 的信息 gate。

---

# 12. Level 1：Contextual Bandit

如果：

- 一个 intervention 决定 episode 后续；
- persistent fallback 常常直接执行到结束；
- handback 没开放；

那么很多 decision point 其实没有真正的多步 credit assignment。

这时问题更像：

\[
c^*=
\arg\max_c
E[R\mid s,c].
\]

使用 contextual bandit 比完整 RL 更简单、更稳定。

推荐做法：

```text
Q(s, continue)
Q(s, requery)
Q(s, resample)
Q(s, fallback)
```

直接选：

\[
\arg\max Q.
\]

再加 conservative uncertainty：

\[
c^*
=
\arg\max_c
\mu_Q(s,c)-\beta\sigma_Q(s,c).
\]

如果：

```text
Bandit ≈ Offline RL
```

就不要为了“强化学习”三个字硬上 RL。

---

# 13. Level 2：IQL —— 推荐作为正式 offline RL 主算法

**推荐级别：★★★★★**

我最推荐首先实现 **Implicit Q-Learning (IQL)**。

Offline RL 最大的问题之一是：

> Q-learning 会对 dataset 没出现过的 action 产生过高估计。

IQL 的重要特点是 policy improvement 时不需要显式评价 dataset 外动作。

这和你的数据非常匹配，因为不同 state 的 correction operator coverage 不完全一致。

## IQL 三个组成

### 1. Expectile value regression

\[
L_V
=
|\tau-\mathbb 1(Q-V<0)|
(Q-V)^2.
\]

可以从：

```text
tau = 0.7 / 0.8 / 0.9
```

做消融。

### 2. Q learning

\[
L_Q
=
\left(
Q(s,c)
-
[r+\gamma^{\Delta t}V(s')]
\right)^2.
\]

注意 RASE 使用：

\[
\gamma^{\Delta t}
\]

因为是 SMDP。

### 3. Advantage-weighted policy extraction

\[
A(s,c)=Q(s,c)-V(s)
\]

\[
w=\exp(\beta A)
\]

再做 weighted behavior cloning。

因为 correction action 是小型离散集合，可以直接输出：

```text
pi(c | z_t)
```

softmax over operator。

---

# 14. Level 2 baseline：CQL

**推荐级别：★★★★☆**

Conservative Q-Learning 的主要作用是：

> 显式压低 OOD action 的 Q 值。

它很适合作为 conservative baseline。

例如某 state 从未运行：

```text
REPLAN
```

普通 Q-learning 可能错误给出很高 Q。

CQL 会通过 conservative regularization 压制 dataset 外动作的估计。

对于小离散 action：

> CQL-DQN / discrete CQL

很自然。

### IQL vs CQL

**首选 IQL：**

- 数据 branch coverage 尚可；
- 希望从数据中学出比 behavior 更好的 policy。

**CQL 更重要：**

- operator coverage 很不均匀；
- OOD action overestimation 明显；
- safety 比 aggressive improvement 更重要。

最终论文最好两个都做。

---

# 15. Level 3：AWAC —— 推荐用于 Offline-to-Online

**推荐级别：★★★★☆，但不是第一阶段。**

AWAC 特别适合：

```text
已有 offline dataset
+
之后允许 simulator online interaction
```

RASE 使用方式：

## Offline initialization

先用已有：

- counterfactual branches；
- source trajectories；
- failure states；

训练 correction policy。

## Online simulator fine-tuning

然后在 LIBERO / RoboTwin：

```text
run RASE
collect new decision transitions
update policy
```

特别关注 policy 自己访问到的新 state distribution。

这能缓解纯 offline RL 的 distribution shift。

---

# 16. PPO / GRPO 是否适合？

## 对 RASE selector

**不作为首选。**

原因：

- 已经有大量离线数据；
- action space 小；
- simulator interaction 仍昂贵；
- PPO/GRPO 需要不断重新 rollout；
- 当前科学瓶颈不是 exploration，而是 risk observability 与 correction opportunity。

## 对完整 VLA post-training

可以作为未来 extension。

公开 RLinf-VLA 已经支持：

- OpenVLA；
- OpenVLA-OFT；
- PPO；
- GRPO；
- LIBERO；
- ManiSkill。

未来可以研究：

```text
RASE generates difficult / failure states
        ↓
RLinf-VLA
        ↓
fine-tune source VLA
```

但最好是 appendix / follow-up，而不是 RASE 主方法。

---

# 17. Decision Transformer 是否适合？

当前不推荐作为主算法。

Decision Transformer 把 RL 写成：

```text
return-to-go + state + action sequence
```

的 autoregressive sequence modeling。

它更适合：

- 长 trajectory；
- 长 correction sequence；
- 大规模 offline transitions。

如果以后出现：

```text
source
 -> requery
 -> source
 -> resample
 -> replan
 -> fallback
 -> handback
```

并且有数十万 decision transitions，那么 DT 更值得试。

第一版用 IQL/CQL 更简单、更容易解释。

---

# 18. 推荐算法总排名

| 方法 | 当前推荐 | 作用 |
|---|---:|---|
| Supervised Q/advantage | ★★★★★ | 信息 gate / baseline |
| Contextual bandit | ★★★★★ | 第一可用 selector |
| **IQL** | **★★★★★** | **主 offline RL** |
| **CQL** | **★★★★☆** | conservative offline RL baseline |
| **AWAC** | **★★★★☆** | offline → simulator online |
| Decision Transformer | ★★☆☆☆ | 长 correction sequence 后再考虑 |
| PPO | ★★☆☆☆ | online selector / full VLA extension |
| GRPO | ★★☆☆☆ | VLA reasoning/post-training extension |
| End-to-end VLA RL | ★☆☆☆☆ 当前 | 会混淆主贡献 |

---

# 19. 推荐最终算法：Risk-Shielded SMDP-IQL

如果要给 RL 版本一个具体实现，我建议：

> **Risk-Shielded SMDP-IQL**

结构：

```text
history
  |
  v
shared RASE encoder
  |
  +------- risk heads
  |
  +------- V(s)
  |
  +------- Q(s, correction)
  |
  +------- pi(c | s)
```

Inference：

1. risk model 输出 risk / urgency / recoverability / OOD；
2. 构造 action mask；
3. Q ensemble 计算每个 operator 的 LCB；
4. 在 safe actions 中选择 LCB 最大者。

\[
Q_{LCB}(s,c)
=
\mu_Q(s,c)-\beta\sigma_Q(s,c).
\]

---

# 20. 建议使用 Q Ensemble

不要只有一个 critic。

推荐：

```text
3–5 Q networks
```

task bootstrap。

输出：

\[
\mu_Q
\]

和：

\[
\sigma_Q.
\]

如果 Q disagreement 高：

- 不选择 aggressive handback；
- 不选择 unsupported replan；
- fallback / abort 更保守。

---

# 21. RL 数据需要什么格式？

```python
RASETransition = {
    "benchmark": ...,
    "task_group": ...,
    "episode_root_id": ...,
    "policy_id": ...,
    "policy_descriptor": ...,
    "robot_descriptor": ...,

    "state_history_ref": ...,
    "risk_embedding": ...,

    "operator": ...,
    "operator_available_mask": ...,

    "reward_task": ...,
    "reward_harm": ...,
    "cost_query": ...,
    "cost_fallback": ...,
    "cost_latency": ...,

    "delta_t": ...,

    "next_state_history_ref": ...,
    "next_risk_embedding": ...,

    "done": ...,
    "success": ...,

    "branch_root_hash": ...,
    "rollout_seed": ...,
    "replica_id": ...,
}
```

---

# 22. Counterfactual Branch 数据对 RL 特别有价值

普通 offline RL 只有：

```text
s -> behavior action -> outcome
```

你的 simulator restore 可以得到：

```text
                 continue -> outcome A
               /
same state s -- resample -> outcome B
               \
                 fallback -> outcome C
```

这比普通 offline RL dataset 更强，因为 action coverage 可以主动设计。

特别适合：

- Q regression；
- bandit；
- offline RL；
- counterfactual regret analysis。

---

# 23. 同一个 root 的 branches 不是独立样本

必须保持现有严格统计原则。

```text
same snapshot
   |- branch A
   |- branch B
   |- branch C
```

不能当成 3 个独立 task samples。

split：

> 所有 branch / replica / boundary 必须和 root task 在同一个 fold。

bootstrap：

> task / root episode level。

---

# 24. stochastic repeats 怎么进入 RL

对于：

```text
state s
operator c
```

做 K repeats：

\[
y_k\sim Bernoulli(p_{s,c}).
\]

两种处理：

## A. 保留每条 transition

每个 replica 都放进 dataset，但：

- group weight = 1/K；
- split 仍以 root 为单位。

## B. 聚合 distribution target

估计：

\[
\hat p_{succ}(s,c)
\]

以及 variance。

推荐：

> risk probability head 用聚合 count；RL transition 可保留 rollout-level dynamics。

---

# 25. RL 启动前必须满足的 Gate

## RL-1：Operator opportunity

state-dependent oracle 必须明显优于 best fixed operator。

否则 always fallback / always resample 已经够了。

## RL-2：Operator coverage

至少主要 operator：

```text
continue
requery/resample
fallback
```

都要在足够多 task/state 出现。

## RL-3：Observability

supervised Q / advantage regression 必须明显超过：

```text
task
policy
elapsed time
```

prior。

如果 advantage 从 input 根本预测不了：

> RL 也无法解决。

## RL-4：Sequential benefit

比较：

```text
contextual bandit
vs
offline RL
```

如果没有明显差异：

> 用 bandit。

---

# 26. 公开数据集：总体结论

公开数据非常有用，但是要分成两类。

## A 类：可以帮助 risk representation

- RoboMIND；
- RoboFAC；
- ViFailback；
- Open X-Embodiment；
- DROID；
- BridgeData V2。

## B 类：可以直接帮助最终 correction RL

数量明显更少。

最终 RL 需要：

\[
(s,\ correction,\ reward,\ s')
\]

最好还需要：

> 同一个 state 下多个 correction operator 的结果。

绝大多数公开 robot dataset 是 expert demonstration，并没有 continue vs resample vs fallback 的 counterfactual branch。

所以：

> **公开数据可以显著降低 representation learning 成本，但核心 selector / RL 数据仍应该主要由你自己的 simulator counterfactual collector 产生。**

---

# 27. RoboMIND

公开论文报告：

- 107k demonstration trajectories；
- 479 tasks；
- 96 object classes；
- 4 robot embodiments；
- multi-view observation；
- proprioception；
- language task description；
- 额外 5k real-world failure demonstrations；
- failure trajectories 附有详细 failure causes；
- 同时提供 Isaac Sim digital-twin environment。

## 对 RASE 的可行性

**★★★★★**

这是最值得优先使用的公开数据之一。

## 推荐怎么用

### 1. Failure-aware representation pretraining

训练：

\[
L_{failure-type}
\]

预测：

- grasp failure；
- collision；
- drop；
- wrong target；
- spatial mismatch；
- execution failure。

### 2. Temporal localization

如果 failure cause / temporal annotation 足够，训练模型识别 failure 前若干帧到 failure 发生的视觉变化。

但不能简单把“最终失败 trajectory”的所有早期帧都标成高风险，否则会重复你旧 t=0 / final-outcome target 的问题。

### 3. Cross-embodiment pretraining

训练：

```text
shared visual/state encoder
+
robot descriptor
```

## 不适合怎么用

不能直接把 RoboMIND 当 RASE correction RL dataset，因为没有保证包含同状态下多种 correction 的真实 counterfactual outcomes。

---

# 28. RoboMIND 2.0

论文报告：

- 310K+ real-world dual-arm trajectories；
- 6 robot embodiments；
- 739 tasks；
- 12K tactile-enhanced episodes；
- 20K mobile manipulation trajectories；
- 额外 20K simulated trajectories；
- MIND-2 本身采用 hierarchical system + offline RL。

## 对 RASE 的价值

**★★★★☆**

特别适合：

- bimanual representation；
- cross-embodiment；
- long-horizon；
- offline RL engineering reference。

建议不要第一阶段直接下载/训练全部 310K，优先选择 sim subset 和 action semantics 接近的 embodiment。

---

# 29. RoboFAC

论文报告：

- 9,440 erroneous manipulation trajectories；
- 78,623 QA pairs；
- 16 tasks；
- 53 scenes；
- simulation + real world；
- failure analysis + correction。

## 对 RASE 可行性

**★★★★☆**

适合：

> failure semantics。

推荐训练 auxiliary：

```text
failure type
failure cause
suggested correction category
```

可以把 correction category 对齐到：

```text
requery
resample
replan
fallback
```

但只作为 auxiliary target。

QA 中“应该重新抓取”不等于：

\[
Q(s,\mathrm{regrasp})
\]

因为没有真实执行 correction 后的 closed-loop return。

---

# 30. ViFailback

公开论文报告：

- 5,202 real-world manipulation trajectories；
- 58,126 VQA pairs；
- 11 fine-grained VQA tasks；
- failure diagnosis；
- textual / visual correction guidance。

## 对 RASE 的价值

**★★★★☆**

适合：

- visual failure diagnosis；
- correction localization；
- visual guidance。

可以训练：

```text
failure region / failure object
correction target
failure stage
```

但不要把 VQA answer 直接映射成 RL reward。

---

# 31. Open X-Embodiment

官方项目报告：

- 1M+ real robot trajectories；
- 22 robot embodiments；
- 21 institutions；
- 527 skills。

## 可行性

**★★★★☆ representation**  
**★☆☆☆☆ direct correction RL**

## 推荐用法

### 1. Action semantic encoder pretraining

把不同数据源 action 映射到：

```text
translation
rotation
gripper
joint
base
```

语义空间。

### 2. Visual-proprio temporal pretraining

训练：

```text
history -> future representation
```

或：

```text
observation + action -> next-state embedding
```

只学习动态一致性，不预测 success。

### 3. OOD / support model

学习 observation/action pair 是否落在机器人数据分布内。

## 主要问题

不同来源：

- observation modality 不统一；
- action semantics 不统一；
- frequency 不统一；
- robot geometry 不统一。

所以必须先做好 canonical action interface。

---

# 32. DROID

论文报告：

- 76k demonstrations；
- 350 hours；
- 564 scenes；
- 84 tasks；
- 50 data collectors；
- 多地区真实场景。

## 对 RASE 价值

**★★★★☆**

最大优势不是 failure，而是 scene diversity。

推荐：

- visual robustness pretraining；
- OOD encoder；
- multi-view temporal representation。

不适合直接学习何时 fallback。

---

# 33. BridgeData V2

公开论文报告约：

- 60k trajectories；
- 24 environments；
- language-conditioned manipulation；
- 多种 skills。

论文也直接评估了 imitation learning 与 offline reinforcement learning 方法。

## 对 RASE 价值

**★★★☆☆**

适合：

- language-action grounding；
- manipulation representation；
- offline RL data pipeline smoke。

不建议把其任务 reward 直接混入 RASE 主实验。

---

# 34. RoboTwin 2.0

官方项目和论文报告：

- 50 dual-arm tasks；
- 5 dual-arm embodiments；
- 100k+ pre-collected trajectories；
- structured domain randomization；
- open-source environment 和 dataset。

## 对 RASE 可行性

**★★★★★**

RoboMIND 更适合 failure representation；RoboTwin 更适合自己生成最终 counterfactual RL 数据。

---

# 35. 为什么 RoboTwin 特别适合 RL

因为它是 simulator。

你可以：

```text
restore same state
   |
   + continue VLA
   + resample
   + requery
   + expert fallback
   + replan
```

然后拿到真实：

```text
success
cost
next state
```

这正是 offline RL 需要的数据。

---

# 36. RoboTwin 官方 expert 数据的限制

官方 data collection 文档说明其流程会先搜索满足目标采集数量的 random seeds，再 replay seed 收集数据。

因此预采集数据更偏 expert successful demonstrations。

它不应该直接作为：

> failure-risk prevalence dataset。

正确方式：

### Stage A

官方 100k trajectories：

- action encoder；
- bimanual representation；
- task semantics。

### Stage B

运行你的 source VLA，自然产生：

- success；
- failure；
- near-failure；
- correction branches。

### Stage C

这些新 branch 才成为：

> RASE-RL dataset。

---

# 37. CALVIN

CALVIN 是公开 long-horizon language-conditioned manipulation benchmark。

## 价值

不是因为它提供 failure dataset，而是因为：

- long horizon；
- sequential task composition；
- language-conditioned；
- public simulator / dataset。

它适合验证：

> RL 是否真的比 contextual bandit 有 temporal credit assignment 优势。

如果 LIBERO：

```text
Bandit == IQL
```

而 CALVIN：

```text
IQL > Bandit
```

会形成很漂亮的结论：

> long-horizon correction requires sequential value learning.

---

# 38. 公开数据集优先级总表

| Dataset | Risk pretrain | Failure semantics | Cross-embodiment | Direct RL selector | Counterfactual generation | 推荐 |
|---|---:|---:|---:|---:|---:|---:|
| **RoboMIND** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ digital twin | **P0** |
| **RoboTwin 2.0** | ★★★★☆ | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★★ | **P0** |
| RoboFAC | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | P1 |
| ViFailback | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | P1 |
| Open X | ★★★★☆ | ★☆☆☆☆ | ★★★★★ | ★☆☆☆☆ | ★☆☆☆☆ | P1 |
| DROID | ★★★★☆ | ★☆☆☆☆ | ★★★☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | P2 |
| BridgeData V2 | ★★★☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | P2 |
| CALVIN | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ self-collect | ★★★★★ | P1 benchmark |

---

# 39. 最推荐的数据组合

不要一次使用全部公开数据。

## Core

```text
你的 R6–R10
+
新 LIBERO counterfactual
+
RoboTwin counterfactual
```

负责最终 risk + correction policy。

## External pretraining P0

```text
RoboMIND
```

负责 failure-aware representation。

## External pretraining P1

二选一：

```text
RoboFAC
or
ViFailback
```

负责 failure semantic auxiliary training。

## Cross-embodiment P1

```text
Open X subset
```

不要全量，负责 robot/action semantic representation。

---

# 40. 一个非常重要的数据使用原则

公开 failure trajectory 不能这样用：

```python
for every frame in failed_episode:
    risk_label = 1
```

这是错误的。

因为 episode 早期可能完全正常，会重新产生 t=0 final-outcome prediction 的问题。

## 正确方法

### 如果有 failure onset

只对：

```text
failure onset 前窗口
```

训练 local risk。

### 如果没有 onset annotation

只用：

- episode-level contrastive；
- failure-type auxiliary；
- final segment classification；
- temporal representation。

不要生成伪 early-risk label。

---

# 41. 公共数据与自己数据如何混合训练

建议三阶段。

## Stage A：General robot pretraining

数据：

```text
Open X subset
DROID subset
RoboMIND success
RoboTwin expert
```

loss：

```text
temporal consistency
action reconstruction
masked action semantics
future feature prediction
```

## Stage B：Failure-aware pretraining

数据：

```text
RoboMIND failures
RoboFAC
ViFailback
```

loss：

```text
failure type
failure semantics
temporal localization
correction category auxiliary
```

## Stage C：RASE task-specific training

只用：

```text
LIBERO / RoboTwin / CALVIN
source VLA rollout
+
counterfactual correction branches
```

训练：

```text
risk
urgency
recoverability
Q
policy
```

---

# 42. 是否应该联合训练所有数据？

第一版不建议。

更稳：

```text
pretrain
  ↓
freeze / low-LR
  ↓
RASE fine-tune
```

否则公开数据量远大于 counterfactual data，会让模型主要优化 expert behavior representation，而不是 failure / intervention decisions。

---

# 43. 推荐 RASE-RL 训练流水线

```text
Step 1
Public robot data
    ↓
general temporal encoder

Step 2
Public failure data
    ↓
failure-aware encoder

Step 3
Your counterfactual dataset
    ↓
calibrated risk + urgency

Step 4
Counterfactual Monte-Carlo returns
    ↓
supervised Q / bandit

Step 5
same transitions
    ↓
IQL

Step 6
CQL baseline

Step 7
closed-loop simulator evaluation

Step 8
if IQL works
    ↓
AWAC online tuning

Step 9
held-out VLA zero-shot

Step 10
tiny RASE post-training
```

---

# 44. 代码架构建议

```text
rase/
├── encoders/
│   ├── visual_temporal.py
│   ├── proprio_temporal.py
│   ├── action_semantic.py
│   └── embodiment.py
│
├── risk/
│   ├── risk_core.py
│   ├── source_risk.py
│   ├── urgency.py
│   ├── recoverability.py
│   └── uncertainty.py
│
├── rl/
│   ├── transition.py
│   ├── reward.py
│   ├── q_supervised.py
│   ├── bandit.py
│   ├── iql.py
│   ├── cql.py
│   ├── awac.py
│   └── shield.py
│
├── operators/
│   ├── continue_source.py
│   ├── requery.py
│   ├── resample.py
│   ├── replan.py
│   ├── fallback.py
│   └── abort.py
│
└── datasets/
    ├── robomind_adapter.py
    ├── robotwin_adapter.py
    ├── libero_adapter.py
    └── rase_transition_builder.py
```

---

# 45. RL 实验必须有哪些 baseline

## Non-RL

1. source only
2. always fallback
3. fixed early fallback
4. fixed short chunk
5. always resample
6. risk threshold
7. supervised advantage
8. contextual bandit

## Offline RL

9. IQL
10. CQL
11. optional Decision Transformer

## Offline-to-online

12. IQL → online fine-tune
13. AWAC

## Full-VLA RL

只在 appendix：

14. PPO/GRPO VLA post-training

---

# 46. RL 是否真的带来贡献，看什么指标

不只看 success。

## 46.1 Success

\[
P(\mathrm{success})
\]

## 46.2 Correction cost

```text
extra source calls
resample calls
fallback steps
planner calls
```

## 46.3 Harm

\[
P(\text{RASE action harms source-success episode})
\]

## 46.4 Regret to privileged oracle

\[
Regret
=
Q(s,c^{oracle})
-
Q(s,c^{RASE}).
\]

## 46.5 Early Rescue Recall

本来存在 rescue opportunity 的 source failure 中，有多少在 window 关闭前被救。

## 46.6 Action coverage

每个 operator：

- train frequency；
- evaluation frequency；
- OOD selection rate。

---

# 47. Zero-shot RL policy 怎么做

RL policy 不应该只输入 VLA ID，而应使用：

```text
policy behavior descriptor
robot embodiment descriptor
action semantics
risk representation
```

训练：

```text
VLA A + VLA B
```

测试：

```text
held-out VLA C
```

## Zero-shot

不更新任何参数。

## Unlabeled adaptation

使用：

- action distribution；
- score normalization；
- descriptor statistics。

## Labeled RASE-PT

只更新：

- Q calibration；
- policy adapter；
- FiLM；
- temperature。

保持 VLA frozen。

---

# 48. RL 版本的论文主张

如果 RL 有明显收益，可以写：

> RASE formulates proactive VLA correction as a stochastic semi-Markov decision process over reusable correction operators. A calibrated risk model shields unsupported actions, while a conservative offline RL policy optimizes long-horizon success, intervention cost, and recovery timing.

---

# 49. 如果 RL 最后没有赢怎么办？

这不会让项目失败。

如果：

```text
supervised advantage = 84.1
IQL = 84.3
CQL = 83.9
```

说明当前 correction problem 的长期 credit assignment 不强。

主论文用 supervised / bandit selector 即可，RL 放 appendix。

---

# 50. 什么结果才值得把 RL 放进标题/主要 contribution

至少满足：

## G-RL-1

IQL 相对 supervised advantage：

\[
+2pp
\]

以上 success，或者同 success 下明显降低 intervention cost。

## G-RL-2

至少：

```text
2 VLA
2 benchmark
```

同方向。

## G-RL-3

改善来自 sequential decisions，而不是 threshold 巧合。

## G-RL-4

online fine-tuning 如果使用，不能导致 zero-shot/generalization 崩溃。

否则 RL 只是 implementation option，不是 main novelty。

---

# 51. 推荐实际实施顺序

## 第一周

不要真正跑 RL。

完成：

1. counterfactual transition schema；
2. reward decomposition；
3. SMDP delta_t；
4. operator availability mask；
5. supervised Q regression。

## 第二周

做：

```text
supervised Q
vs
contextual bandit
```

如果 supervised Q 都没有 signal：

> STOP RL。

## 第三周

实现：

```text
IQL
CQL
```

只在 LIBERO。

## 第四周

闭环比较：

```text
bandit
IQL
CQL
```

如果 IQL 有明显收益，再解锁 cross-VLA / cross-benchmark。

---

# 52. 预计额外耗时

如果已有新版 counterfactual dataset：

### Supervised Q / bandit

约 2–4 天。

### IQL + CQL

约 4–7 天。

### closed-loop RL evaluation

约 3–5 天。

### AWAC simulator fine-tuning

额外约 4–7 天。

所以：

> **把一个可靠的 offline-RL selector 加进 RASE，大约增加 2–3 周。**

如果做完整 VLA PPO/GRPO，通常还会再增加数周，而且研究风险明显更高。

---

# 53. 最终推荐

## 现在应该做

```text
Risk model
+
Supervised Q
+
Bandit
+
IQL
+
CQL baseline
```

## 核心 RL 主算法

> **Risk-Shielded SMDP-IQL**

## Offline-to-online extension

> **AWAC**

## 现在不应该做

```text
end-to-end PPO/GRPO on all VLA parameters
Decision Transformer as first RL method
world-model-based RL
```

---

# 54. 公开数据最终选择

如果只允许下载两个：

## 第一名：RoboMIND

用途：

> failure-aware representation + cross-embodiment。

## 第二名：RoboTwin 2.0

用途：

> bimanual representation + 自己生成 counterfactual RL branches。

如果再加一个：

> RoboFAC 或 ViFailback

用于：

> failure semantics auxiliary training。

---

# 55. 最终完整方案

```text
                  PUBLIC DATA
                       |
       +---------------+----------------+
       |                                |
       v                                v
RoboMIND / RoboFAC             Open X / RoboTwin
failure semantics              robot/action diversity
       |                                |
       +---------------+----------------+
                       |
                       v
              shared temporal encoder
                       |
                       v
            OWN COUNTERFACTUAL DATA
       LIBERO / RoboTwin / CALVIN
                       |
                       v
       stochastic risk + urgency model
                       |
                       v
             supervised Q baseline
                       |
                       v
             contextual bandit
                       |
                       v
       Risk-Shielded SMDP-IQL
                       |
                       v
              CQL comparison
                       |
                       v
         optional AWAC online tuning
                       |
                       v
      held-out VLA / benchmark zero-shot
                       |
                       v
          tiny RASE post-training
```

---

# 56. 最终判断

### RL 是否适合 RASE？

**适合，而且逻辑非常自然。**

但 RL 应学 correction decision，而不是替代 calibrated risk estimation。

### Offline RL 还是 Online RL？

**Offline RL 优先。**

因为你本来就在 simulator 中通过 snapshot / restore 采集大量 counterfactual branches，这是一种非常适合 offline RL 的数据形态。

### 什么算法首选？

1. supervised Q；
2. contextual bandit；
3. **IQL 主算法**；
4. **CQL conservative baseline**；
5. AWAC offline-to-online；
6. PPO/GRPO 只作为 VLA post-training extension。

### 公开数据能不能解决数据问题？

**能解决很大一部分 representation / failure semantics 问题，但不能完全解决 selector 数据问题。**

公开数据适合：

- visual representation；
- temporal representation；
- failure semantics；
- embodiment/action generalization；
- OOD support。

真正决定 RASE 能否学会：

> “现在应该 continue、resample、replan 还是 fallback”

的数据，仍然需要：

> **你自己的 same-state counterfactual correction branches。**

这部分反而可能成为 RASE 独特的数据资产和论文贡献之一。

---

# 57. 参考资料与公开来源

以下信息于 **2026-08-13** 核验；正式投稿前建议再次检查版本、license 和下载接口。

## RL algorithms

- Implicit Q-Learning (IQL):  
  https://arxiv.org/abs/2110.06169

- Conservative Q-Learning (CQL):  
  https://arxiv.org/abs/2006.04779

- AWAC:  
  https://arxiv.org/abs/2006.09359

- Decision Transformer:  
  https://arxiv.org/abs/2106.01345

- RLinf-VLA:  
  https://arxiv.org/abs/2510.06710

- Improving VLA with Online Reinforcement Learning / iRe-VLA:  
  https://arxiv.org/abs/2501.16664

## Public robot datasets / environments

- RoboMIND:  
  https://arxiv.org/abs/2412.13877  
  https://x-humanoid-robomind.github.io/

- RoboMIND 2.0:  
  https://arxiv.org/abs/2512.24653

- RoboFAC:  
  https://arxiv.org/abs/2505.12224  
  https://mint-sjtu.github.io/RoboFAC.io/

- ViFailback:  
  https://arxiv.org/abs/2512.02787

- Open X-Embodiment:  
  https://robotics-transformer-x.github.io/  
  https://arxiv.org/abs/2310.08864

- DROID:  
  https://droid-dataset.github.io/  
  https://arxiv.org/abs/2403.12945

- BridgeData V2:  
  https://rail-berkeley.github.io/bridgedata/  
  https://arxiv.org/abs/2308.12952

- RoboTwin 2.0:  
  https://robotwin-platform.github.io/  
  https://arxiv.org/abs/2506.18088

- CALVIN:  
  https://calvin.cs.uni-freiburg.de/
