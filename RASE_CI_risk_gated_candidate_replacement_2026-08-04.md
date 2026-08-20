# RASE-CI：风险门控的候选动作替换与同策略纠正式推理

**版本日期：** 2026-08-04  
**工作名称：** RASE-CI  
**英文全称：** Risk-Aware Selection and Execution for Corrective Inference  
**中文定位：** 面向冻结 VLA 的风险门控、候选动作替换与执行中纠正系统

---

# 0. 一句话结论

本方法不再把恢复问题定义为：

> 先切换到 OFT，再学习什么时候交还给 SmolVLA。

新的核心问题是：

> **在冻结 VLA 生成动作之后，系统能否在动作执行前判断其风险；当当前动作风险过高时，能否通过同策略重采样、重新条件化或受约束纠正生成更可靠的替代动作；当动作执行过程中环境发生变化时，能否及时截断失效 suffix 并重新规划；当所有当前状态候选都不可靠时，能否安全停止、恢复物理状态或请求人工。**

核心原则：

> **先判断当前动作是否值得执行，再决定是否替换；执行后持续验证，动作失效时立即截断；没有可信候选时不强行执行。**

---

# 1. 方法总览

```text
Frozen VLA generates current action chunk
        ↓
Pre-execution candidate-conditioned risk evaluation
        ↓
Current action safe and useful?
   ├── Yes
   │    ↓
   │ Execute only a short safe prefix
   │    ↓
   │ Observe actual transition
   │    ↓
   │ Transition consistent with expectation?
   │       ├── Yes → Continue / re-evaluate next prefix
   │       └── No  → Truncate stale suffix and fresh replan
   │
   └── No
        ↓
Generate alternative candidates
        ↓
Strict resample / fresh replan / guided refinement
        ↓
Evaluate all candidates
        ↓
Any candidate reliably better than current?
   ├── Yes → Replace current action and execute short prefix
   └── No
        ↓
Safe hold / re-observe / local state restoration / abstain
        ↓
If state changes, invalidate all old candidates
        ↓
Regenerate and re-evaluate from the new state
```

---

# 2. 核心研究问题

## Q1：当前动作是否值得执行？

给定当前状态历史 \(h_t\) 和当前候选动作 \(c_0\)，预测：

\[
R_\theta(h_t,c_0)
\]

系统需要判断：

- 当前动作是否会导致碰撞；
- 是否会丢失抓取；
- 是否会让任务进度回退；
- 是否会进入不可逆状态；
- 是否处于策略支持范围之外；
- 是否值得执行至少一个短前缀。

这里评价的是：

\[
R(h_t,c)
\]

而不是仅评价：

\[
R(h_t).
\]

同一个状态下，不同候选动作可能具有完全不同的结果。

## Q2：如果当前动作风险高，是否存在更好的替代动作？

替代动作可能来自：

1. strict same-distribution resample；
2. fresh same-policy replan；
3. shorter execution horizon；
4. privileged 或 learned guided refinement；
5. safe hold；
6. local corrective primitive。

如果所有候选都失败，则问题不是 selector，而是 candidate support failure 或 state-support failure。

## Q3：执行前安全的动作，执行过程中是否会变得失效？

即使执行前风险很低，也可能因为目标移动、contact/slippage、相机变化、gripper 状态变化、动力学误差或长 action chunk 过时而失效。

因此需要第二层机制：

> **Post-dispatch deviation monitor**

执行中持续比较：

\[
\text{expected transition}
\quad \text{vs} \quad
\text{observed transition}.
\]

若持续不一致：

```text
cancel unexecuted suffix
→ acquire latest observation
→ rebuild history/cache
→ fresh replan
```

## Q4：什么时候应该停止继续挑候选？

如果：

\[
\max_{c\in\mathcal C_t}
LCB(U(h_t,c))
<
\tau,
\]

不能执行“相对最优但绝对不可靠”的候选。

此时应进入：

```text
SAFE_HOLD
REOBSERVE
RETREAT
STABILIZE
REGRASP
ABSTAIN
REQUEST_HUMAN
```

---

# 3. 与 VLA-Corrector 的区别

## 本方法：执行前候选风险选择

```text
VLA generates candidate
→ predict candidate risk before execution
→ reject risky candidate
→ sample or replan alternatives
→ select best reliable candidate
```

主要解决：

- 单次坏采样；
- 动作方向错误；
- 碰撞风险；
- 抓取不稳定；
- 任务语义不一致；
- 当前候选差但同策略分布中存在更好候选。

## VLA-Corrector 类方法：执行中偏差纠正

```text
action chunk already dispatched
→ observe actual transition
→ detect mismatch
→ truncate remaining suffix
→ corrective replan
```

主要解决：

- stale action chunk；
- 执行误差；
- 环境突然变化；
- contact/slippage；
- 原计划在执行中失效。

## 推荐组合

```text
Layer 1: Pre-execution risk selection
Layer 2: Post-dispatch deviation correction
Layer 3: State restoration or abstention
```

第一层阻止“从一开始就不该执行”的动作。  
第二层纠正“开始时合理、后来失效”的动作。  
第三层处理“当前状态下所有候选都无解”的情况。

---

# 4. Candidate 的正式定义

一个候选不应只是 action tensor。

\[
c=
(
u,
e,
a_{t:t+L},
H_g,
H_e,
B,
M
)
\]

其中：

- \(u\)：candidate family；
- \(e\)：executor；
- \(a_{t:t+L}\)：具体 action chunk；
- \(H_g\)：generation horizon；
- \(H_e\)：execution horizon；
- \(B\)：计算与安全预算；
- \(M\)：seed、采样参数、cache 状态等 metadata。

风险模型和价值模型评价：

\[
F_\theta(h_t,c)
\]

而不是只输入状态输出固定 operator label。

---

# 5. 候选动作来源

## C0：Current suffix

当前 SmolVLA 已生成但尚未执行完的 suffix：

\[
c_t^{current}
\]

必须记录：

- 生成时间；
- 已执行前缀长度；
- 剩余 suffix；
- generation horizon；
- execution horizon；
- policy seed；
- action latency；
- cache 初始化方式。

## C1：Strict same-distribution resample

\[
c_t^{(k)}
\sim
\pi_{\text{SmolVLA}}
(c\mid h_t,l;\xi_k)
\]

严格保持：

- 同一 simulator snapshot；
- 同一 RGB / proprioception；
- 同一 history；
- 同一语言指令；
- 同一 cache 初始化；
- 同一 temperature；
- 同一 top-p / top-k；
- 同一 flow / diffusion schedule；
- 同一 generation horizon；
- 同一 execution horizon；
- 只改变 seed。

它回答：

> 当前失败是否只是一次坏采样？

\[
H_{sampling}
=
S_{oracle\ resample}
-
S_{current}
\]

## C2：Fresh same-policy replan

丢弃旧 suffix，获取最新状态后重新调用同一个 SmolVLA：

\[
c_t^{replan}
\sim
\pi_{\text{SmolVLA}}
(c\mid h_t^{fresh},l)
\]

它回答：

> 当前 suffix 是否因为信息陈旧而失效？

\[
H_{reconditioning}
=
S_{oracle\ resample+replan}
-
S_{oracle\ resample}
\]

## C3：Short-horizon receding execution

模型可以生成长 chunk，但只执行前：

\[
H_e\in\{1,2,4\}
\]

个 control steps，然后重新观察和重新规划。

注意：这不是单次 action candidate，而是闭环执行协议，必须与 candidate generation 分开审计。

## C4：Guided candidate refinement

第一阶段可使用：

```text
privileged trust-region action refinement
```

形式：

\[
a'
=
G(a,h_t,R_{priv})
\]

其中：

- \(a\) 来自 SmolVLA；
- \(R_{priv}\) 使用 simulator privileged information；
- \(G\) 在 trust region 中优化动作；
- 当前阶段不能声称已经修改 SmolVLA flow API。

## C5：Safe hold

正式候选：

```text
SAFE_HOLD
STOP_AND_REOBSERVE
```

## C6：Local correction

固定候选库：

```text
RETREAT_SHORT
LIFT_AND_REOBSERVE
STABILIZE_HOLD
REALIGN
OPEN_RESET_GRIPPER
REGRASP
```

## C7：Abstain / request help

```text
ABSTAIN
REQUEST_HUMAN
RESET
```

必须作为正式决策结果，而不是隐式 failure。

---

# 6. 风险模型设计

## 输入

\[
F_\theta(
I_{t-K:t},
q_{t-K:t},
a_{t-K:t},
l,
c
)
\]

包括：

- 多帧 RGB 或 RGB-D；
- proprioception；
- recent action history；
- 语言任务；
- 具体 candidate action chunk；
- generation horizon；
- execution horizon；
- candidate family；
- sampling metadata。

## 输出

\[
F_\theta(h_t,c)
\rightarrow
\left\{
\begin{aligned}
&\hat p_c^{success},\\
&\hat p_c^{collision},\\
&\hat p_c^{object\ loss},\\
&\hat p_c^{grasp\ instability},\\
&\hat p_c^{progress\ regression},\\
&\hat p_c^{irreversible},\\
&\hat p_c^{unsupported},\\
&\Delta\widehat{progress}_c,\\
&\hat\sigma_c
\end{aligned}
\right\}
\]

不建议只预测 terminal failure，因为它会混合局部风险、可恢复失败和最终任务结果。

## Utility

\[
\hat U_c
=
\hat p_c^{success}
+
\alpha_p\Delta\widehat{progress}_c
-
\alpha_h\hat p_c^{harm}
-
\lambda_l C_c^{latency}
-
\lambda_g C_c^{compute}
\]

主论文仍需分别报告 terminal success、collision、object drop、clean-success harm、latency、compute、intervention rate 和 abstention rate。

---

# 7. 动作替换规则

## 当前动作批准

当前动作可以执行，当：

\[
LCB(\hat U_{current})
\geq
\tau_{execute}
\]

且：

\[
UCB(\hat p_{current}^{harm})
<
\eta_h
\]

即使批准，也只执行安全前缀：

\[
H_e\in\{1,2,4\}
\]

## 风险触发候选扩展

若：

\[
LCB(\hat U_{current})
<
\tau_{execute}
\]

或：

\[
UCB(\hat p_{current}^{harm})
\geq
\eta_h,
\]

则生成候选池：

```text
current
+ strict resample × K
+ fresh replan × J
+ safe hold
+ optional guided candidates
```

## 替换条件

令：

\[
c^*
=
\arg\max_{c\in\mathcal C_t}
LCB(\hat U_c)
\]

只有同时满足：

\[
LCB(
\hat U_{c^*}
-
\hat U_{current}
)
>
\delta
\]

和：

\[
UCB(
\hat p_{c^*}^{harm}
)
<
\eta_h
\]

和：

\[
\hat p_{c^*}^{feasible}
>
\eta_f
\]

才替换当前动作。

禁止：

```text
current risk high
→ always choose lowest-risk alternative
```

相对最优不等于可执行。

---

# 8. 典型动作替换流程

## 情况 A：当前动作低风险

```text
Generate current c0
→ evaluate c0
→ risk low
→ execute short prefix
```

## 情况 B：当前动作高风险，但 resample 有解

```text
Generate current c0
→ high risk
→ strict resample c1...cK
→ evaluate all candidates
→ c3 reliably better than c0
→ replace c0 with c3
→ execute short prefix of c3
```

## 情况 C：Strict resample 无解，但 fresh replan 有解

```text
Current suffix high risk
→ strict resamples all low value
→ discard stale suffix
→ acquire latest observation
→ rebuild history and cache
→ fresh replan
→ evaluate new candidate
→ execute if reliable
```

## 情况 D：Natural candidates 无解，但 guided refinement 有解

```text
Natural current/resample/replan all fail
→ privileged trust-region refinement
→ evaluate refined actions
→ compare against matched-compute Best-of-K
→ execute only if independent gain is reliable
```

## 情况 E：所有 same-state candidates 都无解

```text
all same-state candidates unreliable
→ do not hard-select
→ safe hold / local correction / state restoration
```

状态改变后：

```text
observe new real state
→ invalidate all old candidates
→ invalidate all old predictions
→ rebuild history/cache
→ regenerate all candidates
```

---

# 9. 执行中纠正

执行前安全的动作也可能在执行中失效，因此需要 deviation monitor：

\[
D_t
=
D(
\hat z_{t+1},
z_{t+1}^{obs}
)
\]

可定义：

\[
D_t
=
w_vD_{visual}
+
w_pD_{progress}
+
w_cD_{contact}
+
w_aD_{action-consistency}
\]

第一阶段可用：

- feature motion residual；
- object relation regression；
- grasp/contact event；
- no-progress count；
- commanded motion 与 observed displacement；
- gripper command 与物体状态矛盾。

触发规则：

\[
D_t>\tau_D
\quad
\text{for }m\text{ consecutive steps}
\]

触发后：

```text
cancel unexecuted suffix
→ save trigger snapshot
→ acquire latest RGB/proprioception
→ rebuild public history
→ rebuild cache
→ fresh replan
→ return to pre-execution risk evaluation
```

风险模型回答：

> 这个动作是否值得尝试？

Deviation monitor 回答：

> 这个动作是否仍然有效？

---

# 10. 状态改变后的严格规则

一旦执行：

```text
RETREAT
STABILIZE
REGRASP
REWIND
```

就有：

\[
h_t\neq h_{t'}
\]

因此必须：

```text
invalidate all old action candidates
invalidate all old risk predictions
invalidate all old world-model predictions
rebuild public history
rebuild policy cache
regenerate all candidates
rerank at the new state
```

---

# 11. 系统伪代码

```python
def corrective_step(history, instruction):
    current = smolvla.generate(history, instruction)
    current_score = risk_model.evaluate(history, current)

    if is_safe_and_useful(current_score):
        chosen = current
    else:
        candidates = [current]

        candidates += strict_resample(
            history=history,
            instruction=instruction,
            num_samples=K,
        )

        candidates += fresh_replan(
            history=history,
            instruction=instruction,
            num_samples=J,
        )

        candidates.append(safe_hold_candidate())

        if natural_gate_failed_and_guidance_enabled():
            candidates += guided_refinement(history, candidates)

        scores = [
            risk_model.evaluate(history, candidate)
            for candidate in candidates
        ]

        chosen = conservative_select(
            candidates=candidates,
            scores=scores,
            reference=current,
        )

        if chosen is None:
            return safe_hold_or_restore(history)

    result = execute_short_prefix(
        chosen,
        execution_horizon=H_EXEC,
    )

    if deviation_monitor.triggered(result):
        cancel_remaining_suffix(chosen)
        new_history = rebuild_history_from_real_observation(result)
        return corrective_step(new_history, instruction)

    return result
```

---

# 12. PRE-C0 机会空间审计

## Natural candidate oracle

\[
S_0=S_{current}
\]

\[
S_1=S_{current+strict\ resample\ oracle}
\]

\[
S_2=S_{current+resample+fresh\ replan\ oracle}
\]

\[
S_3=S_{natural+fixed\ short-horizon}
\]

对应：

\[
H_{sampling}=S_1-S_0
\]

\[
H_{reconditioning}=S_2-S_1
\]

\[
H_{closed-loop}=S_3-S_2
\]

\[
H_{natural}=S_3-S_0
\]

## Risk-trigger oracle

假设系统完美知道 current 是否会失败，只在失败动作上启动替换：

\[
S_{risk-trigger\ oracle}
\]

它回答：

> 如果风险检测完全正确，风险门控接口最多能带来多少收益？

## Candidate selector headroom

\[
H_{selector}
=
S_{candidate\ oracle}
-
S_{learned\ selector}
\]

## Adaptive horizon headroom

\[
H_{adaptive-horizon}
=
S_{per-state\ horizon\ oracle}
-
S_{best\ fixed\ horizon}
\]

只有该值足够大，才训练 adaptive horizon selector。

## Guided headroom

仅 Natural Gate A FAIL 时运行：

\[
H_{guidance}
=
S_{privileged\ refinement}
-
S_{matched-compute\ Best-of-K}
\]

---

# 13. Gate 设计

## Gate A：Natural Same-Policy Headroom

```yaml
natural_headroom_point_estimate: ">= 5 pp"
trajectory_cluster_bootstrap_lower_bound: "> 0"
positive_in_at_least_two_suites: true
not_driven_by_single_task: true
clean_success_harm_controlled: true
```

通过后：

```yaml
natural_same_policy_gate: open
candidate_critic_gate: eligible
```

## Gate B：Privileged Guided Headroom

仅 Gate A FAIL 时运行。

```yaml
gain_over_matched_compute_best_of_k: positive
guidance_gain_point_estimate: ">= 8-10 pp"
trajectory_cluster_bootstrap_lower_bound: "> 0"
clean_success_harm_controlled: true
trust_region_violations: below_limit
irreversible_harm_not_increased: true
```

通过后：

```yaml
privileged_guidance_gate: open
learned_recovery_critic_gate: eligible
```

失败后：

```yaml
frozen_same_policy_guidance: nogo
next_step:
  - recovery_adapter
  - recovery_distillation
  - abstention
```

---

# 14. 风险模型训练数据

从 exact snapshot 分叉：

```text
current suffix
strict resamples
fresh replans
short-horizon protocols
guided candidates
safe hold
local corrections
```

每个候选执行真实 closed-loop continuation，记录：

- terminal success；
- short-horizon progress；
- collision；
- object drop；
- grasp loss；
- irreversible event；
- recovery to stable state；
- latency；
- compute；
- harmful replacement。

数据隔离：

```text
Risk Model Train
Calibration
Validation
Hidden Test
```

禁止：

- 使用 hidden test 调 threshold；
- 使用 imagined outcome 作为最终真值；
- 使用 candidate test outcome 更新模型；
- 用 privileged state 作为部署时输入，除非明确作为 upper bound。

---

# 15. 第一版最小方法

第一版不使用 world model。

```text
Frozen SmolVLA
+ candidate-conditioned risk critic
+ strict resample
+ fresh replan
+ fixed short execution horizon
+ conservative replacement
+ safe hold
```

第一版不加入：

- alternate VLA；
- 复杂 local planner；
- video world model；
- learned adaptive horizon；
- learned state restoration；
- flow API guidance。

原因：

> 必须先证明自然候选空间和简单风险选择有 headroom。

---

# 16. 推荐实验顺序

## Phase 1：Natural opportunity audit

```text
current
vs strict resample
vs fresh replan
vs fixed H=1/2/4
```

按 T0–T4 deviation stage 分开报告。

## Phase 2：Risk-trigger oracle

验证“只在 current 会失败时干预”是否相对 always-intervene 明显降低误伤。

## Phase 3：Candidate critic

前提：

\[
S_{oracle@K}
-
S_{current}
\]

足够大。

比较：

```text
random
VLA likelihood
entropy
MBR / medoid
history-only critic
candidate-conditioned critic
oracle
```

## Phase 4：Execution-time deviation monitor

比较：

```text
full chunk
fixed H=1
fixed H=2
fixed H=4
heuristic trigger
learned trigger
oracle trigger
```

只有 learned trigger 明显超过 best fixed horizon 时才保留。

## Phase 5：Privileged guidance

仅在 natural candidate headroom 不足时运行。

## Phase 6：Adapter / distillation

如果 privileged guidance 也无效，则说明冻结 SmolVLA 缺少恢复动作支持。

---

# 17. 最终方法定位

建议标题方向：

> **Risk-Gated Candidate Replacement and Corrective Execution for Frozen Vision-Language-Action Policies**

或：

> **Look Before and While You Act: Risk-Gated Candidate Selection and Online Correction for Frozen VLAs**

核心贡献：

1. 执行前 candidate-conditioned risk verification；
2. 风险触发的同策略候选扩展与保守替换；
3. 执行中 stale-suffix detection 和 fresh replan；
4. proposal–evaluation–support failure decomposition；
5. 所有候选无解时的 hold、state restoration 和 abstention；
6. clean-success harm、calibration 和 compute-normalized audit。

---

# 18. 最终结论

本方法的核心不是：

```text
风险高就换动作
```

而是：

```text
先评价当前具体动作
→ 风险高时扩展候选池
→ 对所有候选做绝对风险和相对价值判断
→ 只有替代候选显著更可靠时才替换
→ 执行短前缀
→ 执行中持续验证
→ suffix 失效时立即截断并重新规划
→ 无可信候选时停止、恢复或拒绝
```

方法成立需要同时证明：

\[
\boxed{
\text{风险可区分}
+
\text{候选有 headroom}
+
\text{selector 能捕获 headroom}
+
\text{误伤可控}
}
\]

推荐最终系统：

```text
Pre-execution:
candidate-conditioned risk
→ resample / replan / conservative replacement

During execution:
deviation monitor
→ stale suffix truncation
→ fresh replan

No reliable candidate:
safe hold / state restoration / abstain
```
