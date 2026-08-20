# RASE：从有限 Handback 失败转向同策略纠正式推理

**文档日期：** 2026-08-04  
**阶段状态：** PRE-A3 confirmatory gate 已完成  

```yaml
confirmatory_gate: FAIL
validation_decision: NOGO
finite_handback_method_gate: closed
termination_selector_gate: closed
world_model_gate: closed
candidate_critic_gate: closed_for_finite_handback
hidden_test: not_unblinded
recommended_pivot: same_policy_corrective_inference
```

---

# 0. 一句话结论

PRE-A3 已经较为明确地证明：

> **有限时长 OFT 接管后的 adaptive handback 没有足够的可学习空间。相对于“固定执行到 h=128 再交还”，逐状态选择有限恢复时长几乎没有额外收益。**

但这不等于原始 SmolVLA 无法通过任何方法自我纠错。

它真正说明的是：

> **“先换成 OFT，执行一段时间，再学习什么时候交还给 SmolVLA”不是正确的方法接口。**

下一步若仍坚持“不依赖运行时策略更换”，应把问题从“何时结束 OFT 并 handback”改成：

```text
如何在 SmolVLA 刚开始偏离时，
截断失效动作，
重新条件化，
并直接引导 SmolVLA 自身生成纠正动作？
```

新的核心路线应是：

> **Early deviation detection + stale-chunk truncation + same-policy corrective generation + short-horizon closed-loop execution**

OFT 不再作为运行时恢复策略，而只作为 recovery teacher、反事实标签生成器、恢复方向监督和 recoverability upper bound。

---

# 1. Confirmatory 结果总结

## 1.1 门控结果

Val24 的 7 个冻结门控条件中，6 个通过，唯一失败项为：

```yaml
adaptive_headroom_ge_5pp: FAIL
```

具体结果：

- Validation adaptive headroom：0.0 percentage points；
- Train adaptive headroom：2.78 percentage points；
- 两者均小于预注册阈值 5 percentage points。

定义：

\[
H_{adaptive}
=
S_{finite\ oracle}
-
S_{best\ fixed\ duration}.
\]

Validation 中：

\[
S_{finite\ oracle}=S_{h=128}=17/24.
\]

Train 中：

\[
S_{finite\ oracle}=53/72,\qquad S_{h=128}=51/72.
\]

仅有 2 个 train 状态满足“某个有限时长成功但 h=128 失败”，因此训练逐状态 duration selector 缺少足够 oracle headroom。

## 1.2 主结果

| 指标 | Train72 | Val24 |
|---|---:|---:|
| Base，h=0 | 25.0% | 20.8% |
| Best fixed，h=128 | 70.8% | 70.8% |
| Finite oracle | 73.6% | 70.8% |
| Persistent OFT | 91.7% | 95.8% |
| Oracle − Base | 48.6 pp | 50.0 pp |
| Rescue 数量 | 35 | 12 |
| h=128 对 base-success 误伤 | 0% | 0% |

两边都呈现相同结构：

1. h=8 到 h=32 几乎不额外解锁恢复；
2. 成功率在 h=64→96 区间明显上升；
3. 大量状态的最短成功恢复长度位于 h=96 或 h=128；
4. h=128 已经接近或完全覆盖 finite oracle；
5. persistent OFT 仍比 finite oracle 高约 18–25 percentage points。

---

# 2. 这个 FAIL 实际关闭了什么

## 2.1 被关闭的研究假设

被关闭的是：

> 不同失败状态需要不同的有限 OFT 恢复时长，因此可以通过学习逐状态 termination 或 handback selector，显著优于最佳固定时长。

Validation 上：

\[
S_{adaptive\ finite\ oracle}=S_{fixed\ h=128}.
\]

即使拥有 oracle，也无法超越固定 h=128。因此任何 learned selector 都不可能获得比 oracle 更大的收益。

正式结论：

```yaml
finite_duration_selector: NOGO
adaptive_finite_handback: NOGO
termination_predictor_for_finite_OFT_prefix: NOGO
```

## 2.2 没有被关闭的研究假设

PRE-A3 没有直接测试：

1. SmolVLA 在同一个 snapshot 上是否存在成功 resample；
2. 使用最新观察重新条件化是否能产生成功 replan；
3. 缩短 execution horizon 是否能防止错误累积；
4. 在首次偏离时干预是否比在最终失败状态恢复更有效；
5. critic-guided SmolVLA sampling 是否能进入自然采样难以访问的成功动作区域；
6. 将 OFT 恢复行为蒸馏进 SmolVLA 是否能够增加其恢复能力；
7. 同一个 SmolVLA backbone 加轻量 recovery adapter 是否能实现自我纠错。

因此应重新定义独立门控：

```yaml
same_policy_resample_gate: untested
fresh_replan_gate: untested
adaptive_execution_horizon_gate: untested
guided_generation_gate: untested
recovery_adapter_gate: untested
recovery_distillation_gate: untested
```

---

# 3. 为什么 Persistent OFT 高，而 Finite Handback 饱和

Train：

\[
S_{persistent\ OFT}-S_{finite\ oracle}=91.7\%-73.6\%=18.1\text{ pp}.
\]

Validation：

\[
95.8\%-70.8\%=25.0\text{ pp}.
\]

这说明存在一批状态：

```text
OFT 可以继续完成任务
但任何有限 OFT prefix 后交还 SmolVLA 都失败
```

最可能的解释包括：

## 3.1 SmolVLA 没有重新进入 competence region

OFT 沿着自身熟悉的闭环轨迹继续完成任务，但这些中间状态对 SmolVLA 仍可能是视觉分布外、接触分布外、任务阶段不明确或历史上下文不一致。

## 3.2 OFT 不只是恢复状态，而是在完成剩余任务

对于 persistent-only 状态，OFT 可能继续完成重新定位、重新抓取、运输、放置或剩余子任务。这已经不是局部恢复，而是持续承担任务。

## 3.3 恢复发生得太晚

一旦 SmolVLA 错误动作已把系统推到 unsupported state，普通 resample、fresh replan 和 handback 都可能失败。

因此更合理的方向不是“恢复后更聪明地交还”，而是：

> **在 competence collapse 发生之前，阻止错误继续累积。**

---

# 4. 新的核心研究问题

建议正式改写为：

> **冻结 SmolVLA 在运行时开始偏离时，能否通过同策略重采样、重新条件化、短闭环执行和引导式动作生成产生纠正动作，从而避免进入不可恢复状态？**

英文版本：

> **Can a frozen VLA correct its own emerging failures through early replanning and guided action generation before it leaves its competence region?**

对应四个子问题：

1. 什么时候旧 action chunk 已经不应继续执行？
2. 同一个 SmolVLA 的自然候选空间中是否存在恢复动作？
3. 如果自然采样难以找到恢复动作，能否引导其生成过程？
4. 如果冻结生成空间仍然无解，是否需要 recovery adapter 或蒸馏？

---

# 5. 新方法总览：Same-Policy Corrective Inference

旧流程：

```text
SmolVLA
→ 检测失败
→ OFT 接管
→ 学习何时交还
```

新流程：

```text
SmolVLA normal execution
        ↓
Detect early deviation or stale chunk
        ↓
Cancel unexecuted action suffix
        ↓
Acquire latest observation and history
        ↓
Generate corrective candidates using the same SmolVLA
        ↓
Resample / fresh replan / guided flow sampling
        ↓
Conservative candidate evaluation
        ↓
Execute only a short prefix
        ↓
Re-observe and repeat until stable
```

整个过程中 task policy 始终是 SmolVLA。OFT 不在 runtime 执行动作。

---

# 6. 解决组件一：提前检测偏离

## 6.1 为什么必须提前

从 terminal failure snapshot 开始恢复时，系统可能已经发生：

- 物体掉落；
- 相机视角完全失配；
- gripper 与物体异常接触；
- 机械臂进入不熟悉构型；
- 子任务进度与模型历史不一致；
- 不可逆错误。

新的干预点应前移到：

```text
last stable state
→ first observable deviation
→ first sustained deviation
→ irreversible failure
```

目标是找到：

\[
t_{deviation}<t_{competence\ collapse}.
\]

## 6.2 检测目标

优先检测：

- predicted motion 与 observed motion 不一致；
- object relation 回退；
- grasp stability 下降；
- contact 状态异常；
- progress 连续停滞；
- 当前 suffix 与最新 observation 不一致；
- gripper command 与物体状态矛盾；
- 机器人运动方向偏离任务目标。

可定义：

\[
D_t=
w_1D_{visual}
+w_2D_{progress}
+w_3D_{contact}
+w_4D_{action\ consistency}.
\]

第一阶段可先使用 heuristic，而不是复杂模型：

- task progress；
- object relation；
- feature motion residual；
- action 与 observed displacement；
- grasp/contact event；
- consecutive no-progress count。

## 6.3 触发规则

\[
D_t>\tau_D
\quad\text{for }m\text{ consecutive steps}.
\]

触发后：

1. 取消尚未执行的 action suffix；
2. 保存触发 snapshot；
3. 获取最新 RGB、proprioception 和 action history；
4. 重建 SmolVLA cache；
5. 进入 corrective mode。

---

# 7. 解决组件二：Fresh Replan

长 action chunk 在生成时只看到了较早状态。执行中可能出现视觉、接触、物体、相机和 gripper 状态变化，因此旧 suffix 可能失效。

解决流程：

```text
detect deviation
→ truncate suffix
→ re-observe
→ rebuild public history
→ fresh SmolVLA call
```

Fresh replan 改变：

- observation timestamp；
- proprioception；
- recent action history；
- cache；
- chunk boundary。

保持：

- 同一个 SmolVLA checkpoint；
- 同一个任务指令；
- 同一 action representation；
- 同一控制接口。

它回答的是：

> 使用新证据重新条件化，能否让原策略产生正确动作？

---

# 8. 解决组件三：严格同策略 Resample

定义：

\[
a^{(k)}
\sim
\pi_{SmolVLA}(a\mid h_t,l;\xi_k),
\]

其中仅改变随机 seed。

严格保持：

- snapshot；
- RGB；
- proprioception；
- history；
- cache 初始化；
- sampling temperature；
- top-p / top-k；
- flow schedule；
- action horizon。

关键指标：

\[
S_{resample\ oracle@K}
=
P(\exists k,\ Y(a^{(k)})=1).
\]

\[
H_{resample}
=
S_{resample\ oracle@K}
-
S_{current}.
\]

若明显为正，说明 SmolVLA 分布中已有恢复能力，只是单次采样不稳定；若接近零，仅增加 seed 不能解决问题。

---

# 9. 解决组件四：Adaptive Execution Horizon

纠正模式中，不应完整执行长 chunk。

建议比较：

\[
H_{exec}\in\{1,2,4,8,\text{full chunk}\}.
\]

流程：

```text
generate action chunk
→ execute first 1/2/4 steps
→ re-observe
→ regenerate
```

短 execution horizon 可以更快利用新观测、限制坏动作累积伤害，并形成 receding-horizon closed loop。

它与 finite handback 完全不同：

```text
Finite handback:
OFT 执行 h 步 → 切回 SmolVLA

Adaptive execution horizon:
始终由 SmolVLA 生成动作 → 只改变重新观察频率
```

因此 PRE-A3 的 NOGO 不影响该方向。

---

# 10. 解决组件五：Guided SmolVLA Generation

如果当前状态下 SmolVLA 的自然分布把大部分概率质量放在错误动作上，多次随机采样仍可能得到相似错误。

目标分布：

\[
q_\beta(a\mid h_t)
\propto
\pi_{SmolVLA}(a\mid h_t)
\exp(\beta Q_{recovery}(h_t,a)).
\]

其中：

- SmolVLA 提供动作先验；
- \(Q_{recovery}\) 评价动作是否促进恢复；
- \(eta\) 控制 guidance 强度。

模型参数可以保持冻结，只在 flow / diffusion inference 中修改采样轨迹。

## 10.1 必须先做 Privileged Guidance Upper Bound

在训练 learned critic 前，先使用 simulator privileged information 构造近似 oracle recovery score：

- object-to-goal distance；
- end-effector-to-object alignment；
- grasp attachment；
- object height；
- collision；
- task progress；
- irreversible state indicator；
- contact stability。

定义诊断性 reward：

\[
R_{priv}(h_t,a)
=
\alpha_p\Delta progress
-
\alpha_h harm
+
\alpha_g\Delta grasp\ quality
-
\alpha_c collision.
\]

它回答：

> **如果系统知道正确的恢复方向，SmolVLA 的生成空间里是否存在可执行恢复动作？**

## 10.2 三种结果

### A. Privileged guidance 明显有效

说明 SmolVLA 具备产生恢复动作的运动能力，但自然采样难以访问；可继续训练 recovery critic。

### B. 只在 early-deviation snapshot 有效

说明存在有限 recoverability window。方法重点应是：

```text
detect early
→ correct early
→ avoid competence collapse
```

### C. Privileged guidance 也无效

说明恢复动作不在冻结 SmolVLA 的生成支持内，应关闭：

```yaml
frozen_same_policy_guided_recovery: NOGO
```

然后进入 recovery adapter、OFT-to-SmolVLA distillation 或 abstention。

---

# 11. OFT 的新角色

OFT 不再用于 runtime policy switching。

## 11.1 Recovery Teacher

为 SmolVLA 开始偏离的 snapshot 提供：

- 成功恢复动作前缀；
- 有益状态变化；
- grasp / alignment / progress 修复方向；
- 恢复阶段标签。

## 11.2 Counterfactual Label Generator

从同一个 snapshot 分叉：

```text
SmolVLA current
SmolVLA resamples
SmolVLA fresh replans
guided SmolVLA candidates
OFT teacher continuation
```

真实 continuation outcome 用于训练 candidate critic。

## 11.3 Recovery Direction，而不是动作复制

学习：

\[
\Delta z_t^{recovery}
=
z_{t+k}^{OFT}-z_t.
\]

例如：

- 物体应向目标靠近；
- gripper 应恢复稳定抓取；
-相机—物体关系应回到可识别区域；
- end-effector 应重新对齐；
- 当前状态应离开危险接触。

原则：

> OFT 提供“恢复方向”，SmolVLA 生成“具体动作”。

---

# 12. 新实验阶段：PRE-C0

```yaml
phase: PRE-C0
name: same_policy_corrective_headroom_audit
```

## 12.1 目标

回答：

1. SmolVLA 的自然候选空间中是否存在恢复动作？
2. 使用最新观测和短闭环是否有独立增量？
3. 使用近似 oracle recovery guidance 能否显著扩大成功候选空间？

## 12.2 Snapshot 采样

每条失败 episode 采集：

```text
T0: last stable snapshot
T1: first measurable deviation
T2: first sustained deviation
T3: clear failure-in-progress
T4: terminal failure snapshot
```

同时加入：

- clean success snapshots；
- benign high-risk snapshots；
- camera perturbation；
- robot perturbation；
- contact / non-contact；
- 四个 LIBERO suite；
- 不同任务阶段。

## 12.3 Candidate Arms

每个 exact snapshot：

```text
A0: Current SmolVLA suffix
A1: 8 × strict same-distribution resamples
A2: 4 × fresh SmolVLA replans
A3: Fresh replan with execution horizon 1
A4: Fresh replan with execution horizon 2
A5: Fresh replan with execution horizon 4
A6: 4 × privileged-guided SmolVLA samples
A7: 4 × OFT-target-guided SmolVLA samples
A8: Safe hold / abstain
```

第一阶段不执行 runtime OFT。

## 12.4 Continuation 评测

每个 candidate 从 exact snapshot 执行真实 closed-loop continuation，记录：

- terminal success；
- short-horizon progress；
- object drop；
- grasp loss；
- collision；
- irreversible event；
- first deviation recovery；
- time to return to stable state；
- GPU time；
- wall-clock latency；
- candidate generation cost。

---

# 13. PRE-C0 的嵌套 Oracle

定义：

\[
S_0=S_{current},
\]

\[
S_1=S_{current+strict\ resample\ oracle},
\]

\[
S_2=S_{current+resample+fresh\ replan\ oracle},
\]

\[
S_3=S_{natural+adaptive\ execution\ horizon\ oracle},
\]

\[
S_4=S_{privileged-guided\ oracle},
\]

\[
S_5=S_{OFT-target-guided\ oracle}.
\]

对应增量：

\[
H_{sampling}=S_1-S_0,
\]

\[
H_{reconditioning}=S_2-S_1,
\]

\[
H_{closed-loop}=S_3-S_2,
\]

\[
H_{guided\ support}=S_4-S_3,
\]

\[
H_{teacher\ target}=S_5-S_4.
\]

---

# 14. 新门控标准

## 14.1 Gate A：Natural Same-Policy Headroom

要求：

\[
S_3-S_0\ge 5\text{ pp}.
\]

并且：

- 至少两个 suite 方向为正；
- 收益不只集中于一个 task；
- clean-success harm 不显著；
- held-out episode 方向稳定。

通过后：

```yaml
natural_same_policy_gate: open
candidate_critic_gate: eligible
```

## 14.2 Gate B：Guided Generation Headroom

要求：

\[
S_4-S_3\ge 8\text{–}10\text{ pp}.
\]

并且：

- 至少两个 suite 有独立增益；
- early-deviation 状态收益稳定；
- matched compute 下超过增加随机 seed；
- base-success harm 可控。

通过后：

```yaml
guided_generation_gate: open
learned_recovery_critic_gate: eligible
```

## 14.3 Gate C：Learned Recovery Critic

learned critic 应：

- 捕获 30%–50% privileged guidance gain；
- 显著超过 random candidate；
- 超过 VLA confidence；
- task-held-out 方向稳定；
- harmful replacement rate 低；
- uncertainty 高时关闭 guidance；
- matched compute 下超过 Best-of-K。

定义：

\[
\rho_{guided}
=
\frac{
S_{learned\ guidance}-S_{natural}
}{
S_{privileged\ guidance}-S_{natural}
}.
\]

建议最低要求：

\[
\rho_{guided}\ge0.3.
\]

## 14.4 Gate D：Frozen-Policy Limit

如果在 first-deviation snapshot 上：

\[
S_{privileged\ guidance}
-
S_{natural}
<
5\text{ pp},
\]

则：

```yaml
frozen_same_policy_recovery: NOGO
recovery_adapter_or_distillation: required
```

---

# 15. Learned Recovery Critic 设计

输入：

\[
Q_\phi(h_t,a_{t:t+L})
\]

包括：

- 最近多帧 RGB；
- proprioception；
- recent action history；
- 语言任务；
- candidate action chunk；
- generation seed/profile；
- deviation stage；
- execution horizon。

输出：

\[
\hat p^{success},
\hat p^{recovery},
\hat p^{harm},
\Delta\widehat{progress},
\hat p^{support},
\hat\sigma.
\]

监督使用真实 continuation：

- success / failure；
- 是否恢复 grasp；
- 是否降低 deviation；
- progress delta；
- first irreversible event；
- harmful replacement；
- latency 和 compute。

损失：

\[
\mathcal L
=
\mathcal L_{success}
+
\lambda_r\mathcal L_{recovery}
+
\lambda_h\mathcal L_{harm}
+
\lambda_p\mathcal L_{progress}
+
\lambda_{rank}\mathcal L_{pairwise}.
\]

---

# 16. 保守执行规则

定义：

\[
\hat U_c
=
\hat p_c^{success}
+
\alpha_r\hat p_c^{recovery}
+
\alpha_p\Delta\widehat{progress}_c
-
\alpha_h\hat p_c^{harm}
-
\lambda_g C_{compute}.
\]

选择：

\[
c^*
=
\arg\max_c LCB(\hat U_c).
\]

只有满足：

\[
LCB(\hat U_{c^*}-\hat U_{current})>\delta,
\]

且：

\[
UCB(\hat p_{c^*}^{harm})<\eta_h,
\]

才替换当前候选。

否则执行 current 的短安全前缀、继续观察、safe hold 或 abstain。

---

# 17. 如果冻结 SmolVLA 仍然无解

如果 privileged guidance 都不能显著提升，则必须接受：

> 冻结 SmolVLA 在这些状态上没有足够的恢复动作支持。

此时仍可坚持“不是换成另一个运行时策略”，但需要让同一个模型获得恢复能力。

---

# 18. 方案 A：Recovery LoRA / Adapter

结构：

```text
SmolVLA backbone
+ small recovery LoRA or adapter
```

正常状态 adapter off，检测到 deviation 时 adapter on。

训练数据：

- SmolVLA failure prefixes；
- OFT recovery prefixes；
- successful SmolVLA continuations；
- clean base-success replay；
- perturbation-balanced snapshots。

训练目标：

- imitate recovery-relevant OFT action；
- match useful state transition；
- retain SmolVLA clean behavior；
- penalize harmful deviations。

在 clean states 上加入：

\[
\mathcal L_{retain}
=
D_{KL}(\pi_{adapted}\|\pi_{base}).
\]

必须报告：

- base-success retention；
- adapter activation rate；
- false activation harm；
- recovery-only success；
- held-out suite generalization。

---

# 19. 方案 B：OFT-to-SmolVLA Recovery Distillation

目标不是 runtime 使用 OFT，而是将 OFT 的恢复能力离线蒸馏进 SmolVLA。

每个失败状态构造：

```text
state before deviation
SmolVLA failed action
OFT successful recovery prefix
post-recovery continuation
```

## 19.1 Action Distillation

直接监督 SmolVLA 预测 OFT action chunk。

优点：简单、可直接执行。  
风险：两个策略动作分布可能不兼容。

## 19.2 State-Transition Distillation

监督动作产生与 OFT 相似的状态变化：

\[
z_{t+k}^{student}
\approx
z_{t+k}^{OFT}.
\]

优点：不要求动作完全相同。  
风险：需要稳定状态表征，接触监督较难。

必须加入：

- clean success replay；
- behavior retention；
- low-rank update；
- recovery-only gating；
- hidden confirmation split。

---

# 20. 方案 C：Test-Time Adaptation

可能形式：

- 更新小型 fast weights；
- 更新 LoRA；
- 更新 action head；
- 更新 recovery prompt / latent；
- 使用 progress signal 做少步优化。

风险：

- 在线计算成本；
- 不稳定更新；
- catastrophic drift；
- 安全难保证；
- hidden test 污染风险。

因此优先级低于：

1. frozen guided generation；
2. offline recovery adapter；
3. recovery distillation。

---

# 21. World Model 何时重新打开

当前保持：

```yaml
world_model_gate: closed
```

只有满足以下条件才考虑：

1. natural 或 guided candidate oracle 已有明显 headroom；
2. learned no-WM critic 不能充分捕获；
3. 剩余错误主要来自 action-conditioned future；
4. privileged future evidence 能改善 ranking；
5. imagined ranking 与真实 continuation ranking 相关；
6. held-out perturbation 上仍有增量；
7. compute 和 latency 可接受。

World model 只能帮助评价已有候选，不能替代候选生成。

---

# 22. 推荐代码实现

## 22.1 Deviation Snapshot Mining

```text
scripts/mine_deviation_snapshots.py
scripts/analyze_deviation_timeline.py
tests/test_deviation_snapshot_mining.py
```

输出：

```yaml
last_stable_step:
first_deviation_step:
sustained_deviation_step:
irreversible_failure_step:
terminal_step:
```

## 22.2 Same-Policy Candidate Runner

```text
scripts/generate_smolvla_corrective_candidates.py
scripts/run_pre_c0_same_policy_audit.sh
scripts/analyze_same_policy_headroom.py
tests/test_same_policy_candidate_fidelity.py
```

必须记录：

- exact snapshot hash；
- candidate seed；
- candidate type；
- cache initialization；
- generation horizon；
- execution horizon；
- action tensor；
- continuation outcome；
- compute；
- latency。

## 22.3 Guided Sampling

```text
rase/guidance/privileged_recovery_score.py
rase/guidance/flow_guidance.py
scripts/run_privileged_guidance_audit.py
tests/test_guidance_trust_region.py
tests/test_guidance_determinism.py
```

安全约束：

- action bounds；
- workspace bounds；
- velocity limits；
- gripper limits；
- trust-region penalty；
- guidance norm clipping；
- NaN / divergence fallback。

## 22.4 Audit 输出

```yaml
current_success:
strict_resample_oracle:
fresh_replan_oracle:
adaptive_horizon_oracle:
privileged_guidance_oracle:
oft_target_guidance_oracle:
early_snapshot_gain:
late_snapshot_gain:
base_success_harm:
same_policy_gate:
guided_generation_gate:
adapter_gate:
world_model_gate:
```

---

# 23. 推荐执行顺序

## Step 1：冻结并归档 PRE-A3

正式记录：

```yaml
finite_handback_method_gate: NOGO
reason: no adaptive finite-duration oracle headroom
hidden_test: sealed
```

不再继续 duration selector、termination critic、finite-handback world model 或更多 duration threshold tuning。

## Step 2：挖掘早期偏离 Snapshot

标记：

- last stable；
- first deviation；
- sustained deviation；
- terminal failure。

先人工审计 20–30 条，验证定义可靠。

## Step 3：运行 Natural Same-Policy Audit

比较：

- current；
- strict resample；
- fresh replan；
- adaptive execution horizon。

暂不训练模型。

## Step 4：运行 Privileged Guidance Upper Bound

回答：

> 只要知道恢复方向，冻结 SmolVLA 能不能生成恢复动作？

这是决定 frozen method 是否继续的关键 gate。

## Step 5：训练 Learned Recovery Critic

仅在 Gate B 通过后执行。

第一版：

- no world model；
- structured state/action input；
- pairwise ranking；
- uncertainty calibration；
- held-out task。

## Step 6：闭环 Corrective Mode

实现：

```text
deviation trigger
→ suffix truncation
→ guided SmolVLA generation
→ short execution horizon
→ repeated re-observation
```

## Step 7：决定是否训练 Recovery Adapter

若 privileged guidance 无效或 learned guidance headroom 太小，则训练 LoRA / adapter 或做 OFT-to-SmolVLA distillation。

---

# 24. 论文叙事建议

## 24.1 诊断叙事

> Long recovery prefixes consistently rescue frozen VLA failures, but adaptive finite handback provides no gain over a fixed long prefix. Persistent expert control remains substantially stronger, showing that the missing capability lies not in termination timing but in the base policy's ability to generate corrective actions after deviation.

## 24.2 新方法叙事

> We therefore move correction before handoff. Rather than switching policies after failure, we detect emerging deviations, invalidate stale action chunks, and guide the frozen VLA's own action generation under short-horizon feedback.

## 24.3 推荐标题

- **Correct Before Collapse: Guided Self-Repair for Frozen Vision-Language-Action Policies**
- **Why Handoff Fails: From Recovery Duration to Same-Policy Corrective Generation**
- **Can a Frozen VLA Correct Itself? Early Deviation Detection and Guided Action Generation**
- **Beyond Policy Handoff: Harm-Constrained Self-Correction for Frozen VLAs**

---

# 25. 可提出与不能提出的 Claim

## 25.1 当前可以提出

- Adaptive finite handback lacks oracle headroom beyond a fixed long recovery prefix；
- long recovery duration structure is reproducible；
- persistent OFT exposes substantial residual recoverability；
- the residual gap is not explained by finite termination timing；
- future methods should target corrective action generation or recovery competence rather than duration ranking；
- hidden test remained sealed according to protocol。

## 25.2 新方法通过前不能提出

- SmolVLA natural resampling can reliably recover failures；
- fresh replan is sufficient；
- guided flow sampling will work；
- OFT recovery can be distilled without clean-performance loss；
- early deviation is always detectable；
- frozen same-policy recovery can close the persistent-OFT gap；
- world model evidence is necessary。

---

# 26. 最终决策树

```text
PRE-C0 Natural Same-Policy Audit
        ↓
Do resample/replan/short-horizon candidates add ≥5pp?
  ├── Yes
  │    → train candidate critic
  │    → conservative selection
  │    → no world model initially
  │
  └── No
       ↓
Does privileged guidance add ≥8–10pp?
  ├── Yes
  │    → train recovery critic
  │    → guide frozen SmolVLA generation
  │
  └── No
       ↓
Frozen SmolVLA lacks recovery support
       ↓
Train recovery adapter or distill OFT recovery
       ↓
Does adapted SmolVLA retain clean success?
  ├── Yes
  │    → same-backbone recovery method
  └── No
       → abstain / accept policy capability boundary
```

---

# 27. 最终建议

当前问题不能再通过以下方式解决：

```text
训练更好的 finite duration selector
训练 termination model
加入 world model 预测 handback 时机
继续扩大 h 扫描
```

这些方向已经被 confirmatory gate 合理关闭。

如果仍坚持“原策略通过方法进行纠正”，最合理的解决路径是：

1. 将干预点前移到首次偏离，而不是最终失败；
2. 立即截断已经失效的 SmolVLA action suffix；
3. 使用最新观测进行 fresh SmolVLA replan；
4. 在 corrective mode 中使用短 execution horizon；
5. 先测严格 resample 和 replan 的真实 oracle headroom；
6. 使用 privileged recovery score 测试 SmolVLA 生成空间的上限；
7. 若有上限，训练 recovery critic 引导 SmolVLA flow；
8. OFT 只做离线 teacher，不参与 runtime 策略切换；
9. 若冻结生成空间无解，训练同 backbone recovery adapter 或做 recovery distillation；
10. 继续用 clean-success harm、held-out suite 和 matched compute 作为硬门控。

最终方法的核心不再是：

> 什么时候把控制权从 OFT 交还给 SmolVLA？

而是：

> **如何在 SmolVLA 失去纠正能力之前，直接改变它下一次动作的生成过程，使同一个策略产生恢复动作。**

这条路线既尊重 PRE-A3 的 NOGO 结果，也保留了最初“不是依赖策略更换，而是让原策略能够纠错”的研究目标。
