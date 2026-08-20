# RASE 下一阶段详细执行规划：从 R10-B 到 CVPR 2027 Submission-Ready

**日期：** 2026-08-13  
**目标：** 最大化形成 CVPR 2027 级别完整论文证据链的概率  
**当前起点：** R10-B reproducibility FAIL；full OFT action-trace root-cause diagnostic 正在运行；learned stages 仍锁定

---

# 0. 关于“保证成功”

没有任何研究方案可以保证：

- 实验一定得到正结果；
- zero-shot 一定提升；
- CVPR 一定接收。

能做的是把项目改造成：

> **每一步都有明确的信息价值、预注册 gate、失败即止损、成功则逐层扩大，而且任何一条失败路径都不会让你浪费数周继续堆模型。**

下面这份计划是“成功概率最大化方案”，不是“无条件保证正结果”。

---

# 1. 当前最重要的决定

现在**不要**立刻：

- 训练更大的 risk model；
- 引入 world model；
- 做第三个 VLA；
- 做 CALVIN/RoboTwin；
- 调 selector 阈值；
- 做 handback；
- 解封 validation/test。

因为当前最根本的问题仍是：

> 你到底在测一种可观测的、可预测的闭环风险，还是 simulator / policy stochasticity 导致的不可约随机结果？

先把这一点确定。

---

# 2. Phase 0：完成 R10-B root-cause diagnostic

**优先级：P0**  
**时间：立即，1–2 天**

当前 frozen pilot：

- 9 unstable groups
- 9 matched stable controls
- 每组 K3
- 54 trajectories
- full OFT action-trace hash

---

## 2.1 Branch A：完整 fallback trace 相同，但 outcome 翻转

说明：

- 不是 action-policy branching；
- 可能是 simulator / termination / contact solver randomness；
- 这类部分属于 aleatoric environment risk。

### 下一步

做 fixed-action replay：

```text
same snapshot
same exact full action trace
different repeated execution
```

记录：

- simulator seed；
- contact state；
- object pose；
- termination flags；
- numerical solver；
- success checker。

### Gate

如果 fixed-action replay 仍出现明显 outcome flip：

> STOP deterministic recoverability learning.

之后所有 correction outcome：

- 使用 probability/count supervision；
- 必须报告 posterior uncertainty；
- unstable state 保留。

---

## 2.2 Branch B：首动作相同，但后续 fallback trace 分叉

这反而是比较好的结果。

说明：

> 后续 observation/state 的微小变化导致 closed-loop policy trajectory 分叉。

### 下一步

逐 chunk 记录：

```text
obs hash
proprio hash
source proposal hash
fallback proposal hash
contact/progress
sim state diagnostic hash
```

找最早 divergence step。

### Gate

如果 divergence 之前 deployable observation 已有差异：

> temporal observable signal EXISTS candidate.

这时才允许进入新的 temporal risk data design。

---

## 2.3 Branch C：新的 K3 不再复现 outcome flip

说明小 K selection bias 很严重。

### 下一步

使用 sequential sampling 估计：

\[
p_{\mathrm{succ}}(s,c)
\]

不要再筛选：

```text
2/2 success -> 0/2 failure
```

这种 extreme case。

---

# 3. Phase 1：冻结 vNext research question

**时间：1 天，与 Phase 0 后半并行**

写一个 1 页 protocol，不训练模型。

必须只保留三个主要目标：

### T1 — Local source risk

> 未来 H 步继续执行是否会进入 critical precursor？

### T2 — Intervention urgency

> 如果现在不纠正，未来 Δ 步后最佳 rescue probability 是否明显下降？

### T3 — Correction operator value

> resample/requery/replan/fallback 中哪个最值得？

如果某个实验不服务这三个问题，先不做。

---

# 4. Phase 2：实现 Multi-Benchmark Canonical Interface

**时间：3–5 天**  
**和后续实验并行开发**

这是整个项目能不能从“LIBERO paper”变成“cross-platform paper”的基础。

---

## 4.1 Robot/Observation adapter

实现：

```text
BenchmarkAdapter
  - reset
  - observation_to_canonical
  - success
  - snapshot
  - restore_for_offline_labeling
  - execute
```

Canonical observation 使用：

- variable camera views；
- camera role；
- semantic proprio tokens；
- task text；
- time。

---

## 4.2 Action adapter

不要再把核心写死成 7D。

实现：

```text
PolicyAdapter
  - propose
  - raw_to_canonical_action_tokens
  - canonical_to_raw
  - policy_descriptor
  - supports_resample
  - supports_replan
```

必须有单元测试：

- rotation conversion；
- coordinate frame；
- gripper；
- frequency；
- scaling；
- padding/mask；
- round-trip parity。

这是高优先级，因为公开的 VLA benchmark 复现中，一个 action/proprio conversion 就可能造成非常大的 success 差异。

---

## 4.3 Correction operator API

```python
class CorrectionOperator:
    def available(ctx): ...
    def estimate_call_cost(ctx): ...
    def propose(ctx): ...
    def execute(candidate): ...
```

首批只实现：

1. continue；
2. shorten+requery；
3. resample；
4. persistent fallback；
5. abort。

`replan` 放第二批。

原因：

> 先用最少 operator 验证“多候选价值学习”是否成立。

---

# 5. Phase 3：新的 Decision-Point Dataset

这是整个项目最关键的新实验。

**时间：4–7 天收集首批**

先只做 LIBERO：

- Pi0Fast
- Pi0.5

因为已有 opportunity。

不要一开始加 SmolVLA。

---

## 5.1 自然 cohort

建议：

- 48 tasks；
- 4 suites；
- 每 task 4 independent reset states；
- 每 VLA 至少 2 independent source seeds。

目标：

> 不是重复很多 snapshot，而是增加 independent episode diversity。

---

## 5.2 Decision points

不要全轨迹每一步都 branch。

每条 source trajectory 选 2–4 个 outcome-independent points。

推荐组合：

### 固定点

- normalized progress 20%
- 40%
- 60%
- 80%

### 事件点

仅由当前/过去信息触发：

- gripper transition；
- contact transition；
- action curvature jump；
- velocity/jerk change；
- visual motion phase transition。

不允许使用 future success/failure 来选点。

---

# 6. Phase 3A：先做 Model-Free Correction Opportunity Audit

在训练任何 risk model 前，先回答：

> 多 operator 真的存在不同最优选择吗？

如果最优永远是 persistent OFT，那和 R3 一样，learning selector 没有意义。

---

## 6.1 Branch set v1

对每个 decision point：

- continue source；
- shorten+requery；
- resample-1；
- resample-2；
- persistent OFT。

初始 K=2。

对于结果不一致的 state/operator：

- 增到 K=4；
- 必要时 K=8。

---

## 6.2 必须计算

对于每个 state：

\[
p_{continue}
\]

\[
p_{requery}
\]

\[
p_{resample}
\]

\[
p_{fallback}
\]

以及：

\[
c^*=\arg\max[p_c-\lambda Cost_c]
\]

---

## 6.3 Opportunity Gate

至少满足：

### G-O1

best fixed operator 与 state-dependent oracle 存在真实 gap。

建议：

\[
OracleUtility-BestFixedUtility > 5\%
\]

且 task-bootstrap CI 不完全跨 0。

### G-O2

至少两个 operator 各自在 ≥10% decision points 上成为有效最优。

如果 95% 状态都是 fallback：

> 不训练 operator selector。

### G-O3

两个 VLA 都存在 opportunity。

否则只能做单 VLA 方法。

---

# 7. Phase 3B：重新定义 positive density

R9 失败的重要原因是 positive 太少。

因此新的 risk target 不能再次产生 2–3% positive。

---

## 7.1 Urgent state

定义：

\[
U_t
=
V_{rescue}(t)-V_{rescue}(t+\Delta)
\]

如果：

\[
U_t>\delta
\]

则是 urgent。

---

## 7.2 预训练前 Gate

自然 development cohort 中：

- 每 VLA 至少 50 个 urgent groups；
- 每 suite 至少有 positive；
- positive prevalence 最好 ≥10%；
- 不能只由 elapsed time 分离。

如果做不到：

> 重新设计 target / decision point sampling，而不是训练大模型。

---

# 8. Phase 4：Low-Capacity Information Gate

**时间：1–2 天**

只训练轻量 probe。

---

## 8.1 Baselines

必须先跑：

1. task text only；
2. elapsed time only；
3. policy only；
4. action stats only；
5. task+policy+time；
6. state history；
7. state+source action history。

---

## 8.2 Gate

新 causal features 必须相对最强 prior：

- AUROC +0.05；
- AP 有明显正增益；
- 5 个 task-held-out seeds ≥4 seed 同方向；
- task-bootstrap lower bound 不低于 prior；
- 至少 3/4 suite 有 signal。

如果失败：

> STOP neural model escalation。

不要上 transformer/world model。

---

# 9. Phase 5：建立 RASE-vNext MVP

只有 Phase 3/4 通过才开始。

**时间：4–6 天**

---

## 9.1 MVP 只做四个 head

1. local risk；
2. urgency；
3. operator prior advantage；
4. OOD/confidence。

Candidate verifier 可以 phase 5B 再加。

---

## 9.2 网络

推荐第一版：

```text
tiny visual encoder
+
proprio temporal encoder
+
action semantic temporal encoder
+
task embedding
+
policy descriptor
-> 4-layer temporal transformer
-> heads
```

不要超过 15M。

---

## 9.3 Loss

\[
L
=
L_{risk}
+
\lambda_uL_{urgency}
+
\lambda_aL_{advantage}
+
\lambda_cL_{calibration}
+
\lambda_{ood}L_{selective}
\]

fallback success 使用 count likelihood。

---

# 10. Phase 6：Closed-Loop LIBERO Gate

**时间：2–3 天**

不要先做第三 benchmark。

先证明 controller 真能救。

---

## 10.1 必测 baseline

- source；
- always fallback；
- fixed t0 fallback；
- fixed early fallback；
- always resample；
- fixed short chunk；
- risk-only trigger；
- RASE operator-prior；
- RASE full candidate verifier。

---

## 10.2 第一闭环 Gate

每 VLA：

### Safety

\[
absolute\ paired\ harm \le 5\%
\]

### Success

至少满足一个：

A.

\[
SuccessGain \ge 3pp
\]

或

B.

\[
SuccessGap \ge -1pp
\]

同时：

\[
ExtraCalls/FallbackCostSaving \ge 20\%
\]

### Early rescue

真实 recoverable source failures 中：

\[
EarlyRescueRecall \ge 70\%
\]

作为目标值；如果样本量不足，先报告 CI。

### Stability

5 seeds ≥4 seeds 通过主要方向。

---

# 11. Phase 7：Zero-Shot vs Post-Training 核心实验

这是论文的中心图之一。

---

## 11.1 Split 设计

把一个完整 VLA family held out。

例如：

```text
Train:
Pi0Fast + Pi0.5

Held-out:
VLA-X
```

新 VLA 的 test tasks 与 descriptor collection tasks 分离。

---

## 11.2 RASE-ZS

允许：

- policy behavior descriptor；
- no outcome labels；
- no parameter update。

目标不是大幅提升。

目标：

> macro success gain > 0，并且不显著增加 harm。

---

## 11.3 RASE-UCal

用：

- 8 / 16 / 32 unlabeled episodes。

只能做：

- feature normalization；
- score scale；
- OOD statistics；
- action distribution descriptor。

---

## 11.4 RASE-PT

用：

- 8 / 16 / 32 / 64 labeled trajectories。

训练：

- affine calibration；
- 10K–100K policy adapter；
- 最多小型 last-layer/FiLM。

不要更新 VLA。

---

## 11.5 需要的结果形态

理想曲线：

```text
Source       --------------
RASE-ZS      + small gain
RASE-UCal    + slightly more
RASE-PT-16   ++
RASE-PT-32   +++
RASE-PT-64   ++++
```

如果 zero-shot 只有 +0.5pp：

> 完全可以接受，只要 across cells 一致、CI 不显示明显伤害。

---

# 12. Phase 8：第三 VLA

只有：

- 两个当前 VLA 闭环 gate 通过；
- shared core 不是 collapse；

才加第三 source。

---

## 候选原则

不要因为“模型有名”就选。

选择：

1. 有公开 checkpoint；
2. benchmark reproduction 稳；
3. source success 不能太高到几乎无 failure；
4. 也不能太低到 correction 无意义；
5. 能产生独立 failure modes；
6. correction pair 有 model-free opportunity。

---

## LIBERO 优先候选

- base OpenVLA / OpenVLA-family source；
- 与更强 corrective arm 配对。

当前 OFT checkpoint 不能直接被当作一个新 source claim，先做 source-only parity。

---

# 13. Phase 9：第二 benchmark

优先推荐：

> **CALVIN 或 SimplerEnv，哪个能在 48 小时内完成两个 source checkpoint 的 reproduction，先做哪个。**

不要按“名气”决定。

---

## 13.1 48h Reproduction Gate

一个 benchmark 只有同时满足：

- 官方/可信 baseline 可以复现；
- action conversion parity；
- 2 个 source policy 可以运行；
- 失败数量足够；
- snapshot/restore 可做离线 counterfactual；

才进入主实验。

否则换另一个 benchmark。

---

# 14. Phase 10：第三 benchmark

最低建议：

- LIBERO
- CALVIN
- SimplerEnv

Stretch：

- RoboTwin 2.0

RoboTwin 必须等 variable-effector action semantics 已经稳定。

不要为了 3 benchmark 硬写三套 controller。

必须同一个 shared core。

---

# 15. Phase 11：Cross-Benchmark Zero-Shot

这是高风险但高价值实验。

分三个难度。

---

## ZS-A：Held-out VLA / seen benchmark

必须做。

---

## ZS-B：Seen VLA family / held-out benchmark

强烈建议。

允许：

- robot/benchmark descriptor；
- outcome-free normalization。

---

## ZS-C：Held-out VLA + held-out benchmark

Stretch。

不要把它设成 paper acceptance 的 single point of failure。

---

# 16. Phase 12：加入 Replan Operator

只有前三个 correction operator 已经验证 selector 有价值再加。

---

## 16.1 为什么晚加

`replan` 会引入：

- planner architecture；
- prompt engineering；
- 额外 latency；
- failure mode。

很容易让 paper 复杂度失控。

---

## 16.2 Replan 实现原则

定义统一接口：

```text
replan(task, history) -> new subgoal/context
```

RASE 只评价：

- 是否值得调用；
- 哪个 plan candidate 风险更低。

不要让 RASE 自己变成 planner。

---

# 17. Phase 13：Visual OOD Evaluation

为了 CVPR 强烈建议。

---

## 条件

至少用一种：

- camera shift；
- object appearance shift；
- occlusion；
- lighting；
- distractor；
- language counterfactual。

---

## 主要问题

不是：

> RASE 是否让 OOD success 变 SOTA？

而是：

> **风险预测能否在 OOD 时保持 calibration，或者正确 abstain？**

报告：

- AUROC；
- ECE；
- selective risk；
- OOD coverage；
- closed-loop harm。

---

# 18. Phase 14：World Model 最后再决定

如果 no-WM model 已经成功，开一个 48 小时 budget。

---

## 18.1 只尝试三个 feature

- multi-step action-conditioned residual；
- ensemble disagreement；
- predicted progress residual。

---

## 18.2 Keep Gate

必须同时：

1. +0.03 以上 held-out risk AUROC 或明显 calibration gain；
2. 至少 2 VLA 同方向；
3. closed-loop utility 提升；
4. online latency 可接受或可 distill。

否则：

> 写成 negative ablation。

不要继续。

---

# 19. 12 周冲刺时间表

当前是 2026-08-13。

CVPR 2027 官方 paper deadline 在本计划生成时尚未确认；CVPR 2026 的 paper deadline 是 2025-11-13。为了不赌官方日期，建议把**11 月初作为内部实验冻结线**。

---

## Week 1：Aug 13–20

目标：

- R10 root cause 完结；
- vNext target freeze；
- canonical multi-embodiment action interface；
- correction operator API；
- LIBERO vNext collector smoke。

**必须输出：**

- root-cause verdict；
- protocol v1；
- 20-state branch smoke；
- no-leakage audit。

---

## Week 2：Aug 21–27

目标：

- LIBERO decision-point opportunity dataset v1；
- source/resample/requery/fallback counterfactual。

**Gate：**

- operator oracle gap；
- positive density；
- stochasticity profile。

失败就改 target，不训练模型。

---

## Week 3：Aug 28–Sep 3

目标：

- low-capacity information gate；
- temporal/action features；
- urgency target。

**Gate：**

- causal model > prior +0.05。

---

## Week 4：Sep 4–10

目标：

- RASE MVP；
- task-held-out OOF；
- calibration；
- 5 seeds。

---

## Week 5：Sep 11–17

目标：

- LIBERO closed-loop；
- Pi0Fast + Pi0.5；
- resample/requery/fallback portfolio。

**必须决定：**

> 有没有真正的 controller paper？

---

## Week 6：Sep 18–24

目标：

- shared core；
- held-out VLA；
- zero-shot；
- unlabeled calibration；
- 8/16/32/64 labeled adaptation curve。

---

## Week 7：Sep 25–Oct 1

目标：

- 第三 VLA；
- benchmark-2 reproduction。

---

## Week 8：Oct 2–8

目标：

- benchmark-2 data + controller；
- cross-benchmark adapter。

---

## Week 9：Oct 9–15

目标：

- benchmark-3；
- 或 RoboTwin stretch feasibility。

---

## Week 10：Oct 16–22

目标：

- visual OOD；
- main ablations；
- latency/compute；
- multi-benchmark table。

---

## Week 11：Oct 23–29

目标：

- independent validation；
- 100+ paired closed-loop；
- confidence intervals；
- final seeds。

**此后冻结核心方法。**

---

## Week 12：Oct 30–Nov 5

目标：

- main paper tables；
- figures；
- failure cases；
- video；
- supplementary；
- reproducibility package。

world model 只有此前已经自然通过才允许存在。

---

# 20. Go/No-Go 总表

| Gate | 问题 | PASS | FAIL 后动作 |
|---|---|---|---|
| G0 | stochastic root cause | 明确来源 | 扩大 diagnostic，不训练 |
| G1 | multi-operator opportunity | oracle > best fixed | 简化 correction story |
| G2 | positive density | ≥10% 且跨 suite | 改 target / sampling |
| G3 | observability | causal > prior +0.05 | 停止模型升级 |
| G4 | per-VLA OOF | ≥4/5 seeds | 不做 shared |
| G5 | closed-loop | success/Pareto 改善 | 重新检查 trigger/operator |
| G6 | shared core | 接近 per-VLA | 改 adapter |
| G7 | zero-shot | macro gain >0 | 降为 challenge metric |
| G8 | few-shot PT | 明显 > ZS | 主张 calibration-based transfer |
| G9 | benchmark-2 | 同方向改善 | 不扩 benchmark-3 |
| G10 | benchmark-3 | 不显著伤害 | 完成 multi-benchmark claim |
| G11 | OOD | calibrate/abstain | 作为 failure analysis |
| G12 | WM | state/control gain | 删除 WM |

---

# 21. “成功率最大化”的实验优先级

## P0：必须

- root cause
- new decision target
- model-free operator opportunity
- low-capacity info gate
- closed-loop two-VLA LIBERO
- zero-shot vs few-shot

---

## P1：非常重要

- third VLA
- second simulator benchmark
- third benchmark
- visual OOD
- latency

---

## P2：有时间再做

- replan
- handback
- world model
- RoboTwin
- source VLA LoRA

---

# 22. 明确禁止的时间黑洞

未来三个月尽量不做：

### 1. 大规模 world-model engineering

当前证据不支持。

### 2. 继续搜神奇 threshold

R6 已经说明问题主要不是 threshold。

### 3. handback 微调

不是核心成立条件。

### 4. 低价值超大 K

用 sequential sampling。

### 5. 所有 benchmark 全 Cartesian product

用 connected evaluation graph。

### 6. 在 test set 上调 controller

strict sealed test。

---

# 23. 计算预算控制

Counterfactual branching 会很贵。

推荐三级预算。

---

## Budget A：Discovery

- K=2
- 2 decision points / episode
- 2 VLA
- LIBERO

目标：

> 判断是否值得继续。

---

## Budget B：Confirmation

仅对：

- ambiguous probability；
- informative states；
- validation cohort；

提高 K。

---

## Budget C：Final evaluation

不依赖 offline K。

直接：

- fresh closed-loop episodes；
- paired seeds；
- natural task distribution。

---

# 24. 数据 split 规则

必须：

- task-held-out；
- same task 的所有 state/branch/repeat 同 fold；
- enrichment 不进 natural OOF；
- descriptor collection 与 test task 分开；
- zero-shot held-out VLA 不能使用成功率统计；
- benchmark held-out 时不能用其 outcome 做 threshold。

---

# 25. Reviewer 最容易攻击的点及预防

## Attack 1

> 只是一个 failure detector。

### 防御

必须展示：

- correction operator value；
- urgency；
- multi-operator selection。

---

## Attack 2

> 只是 threshold/resampling。

### 防御

比较：

- fixed resample；
- uncertainty resample；
- risk-only；
- full counterfactual operator selector。

---

## Attack 3

> 只在 LIBERO 有用。

### 防御

至少 3 benchmark connected graph。

---

## Attack 4

> shared model 其实在记 policy ID。

### 防御

- behavior descriptor；
- remove-ID zero-shot；
- leave-one-VLA-out。

---

## Attack 5

> improvement 来自更强 fallback。

### 防御

比较：

- always fallback；
- fixed early fallback；
- same fallback with RASE selective calls。

主指标用：

> success–cost Pareto。

---

## Attack 6

> failure 发生以后才检测。

### 防御

报告：

- intervention lead time；
- recovery-window remaining；
- early rescue recall。

---

## Attack 7

> stochastic label 不可信。

### 防御

- repeated counterfactual；
- Beta-binomial；
- task-level bootstrap；
- unstable groups 不删除。

---

## Attack 8

> multi-benchmark 接口不公平。

### 防御

- canonical semantic action contract；
- round-trip adapter tests；
- benchmark config hashes；
- exact control frequency/report。

---

# 26. 最低可投稿版本 vs 理想版本

## Minimum viable strong paper

- 3 VLA
- LIBERO + 1 different simulator
- 1 visual OOD benchmark
- proactive local risk
- urgency
- 3 correction operators
- zero-shot positive
- 32-shot adaptation stronger
- latency
- >100 fresh closed-loop eval

---

## Ideal CVPR version

- 3+ VLA
- LIBERO + CALVIN + SimplerEnv
- RoboTwin stretch
- 4–5 correction operators
- cross-VLA ZS
- cross-benchmark ZS
- lightweight post-training curve
- vision perturbation robustness
- stochastic calibration
- full success–compute Pareto

---

# 27. 结果表模板

## Table 1：Main benchmark

| Method | LIBERO | CALVIN | SimplerEnv | Avg | Extra Calls |
|---|---:|---:|---:|---:|---:|
| Source |  |  |  |  | 0 |
| Always resample |  |  |  |  |  |
| Always fallback |  |  |  |  |  |
| Fixed replan |  |  |  |  |  |
| Risk-only |  |  |  |  |  |
| **RASE-ZS** |  |  |  |  |  |
| **RASE-PT** |  |  |  |  |  |

---

## Table 2：Cross-VLA

| Train VLA | Test VLA | Zero-shot | +16 | +32 | +64 |
|---|---|---:|---:|---:|---:|
| A+B | C |  |  |  |  |
| A+C | B |  |  |  |  |
| B+C | A |  |  |  |  |

---

## Table 3：Risk quality

| Method | AUROC | AP | ECE | Lead Time | Early Rescue Recall |
|---|---:|---:|---:|---:|---:|
| Time prior | | | | | |
| Action stats | | | | | |
| Temporal state | | | | | |
| Risk-only | | | | | |
| **RASE** | | | | | |

---

## Table 4：Correction policy

| Selector | Success | Harm | Unnecessary Intervention | Fallback Steps | Extra Calls |
|---|---:|---:|---:|---:|---:|
| Always fallback | | | | | |
| Fixed trigger | | | | | |
| Risk-only | | | | | |
| **RASE** | | | | | |

---

# 28. 必须画的主图

### Fig. 1

系统图：

```text
VLA -> source action -> RASE risk/urgency -> correction portfolio -> robot
```

---

### Fig. 2

一条真实失败 trajectory：

```text
risk
urgency
recoverability
trigger
correction
success
```

要清楚展示：

> RASE 在 failure 之前触发。

---

### Fig. 3

Zero-shot / few-shot adaptation curve。

---

### Fig. 4

Success–compute Pareto。

---

### Fig. 5

Cross-VLA / benchmark heatmap。

---

# 29. 如果 zero-shot 不提升怎么办

不要继续调 test。

按顺序检查：

1. policy descriptor 是否覆盖 action statistics；
2. normalization 是否 policy-specific；
3. risk ranking 是否 transfer，但 calibration 失效；
4. architecture shift 是否主要体现在 action encoder；
5. embodiment descriptor 是否缺失。

如果：

- ranking transfer；
- calibration 不 transfer；

这是好消息。

论文可以强调：

> tiny calibration is necessary and sufficient.

如果 ranking 本身不 transfer：

> zero-shot 降为负结果 / challenge metric，主张 small-adapter transfer。

---

# 30. 如果第二 benchmark 失败怎么办

先区分：

### A. Adapter bug

action/proprio/success contract。

修。

### B. No correction opportunity

换 source/corrective pair。

### C. Risk unobservable

只把 benchmark 当 generalization failure analysis，不要硬调。

### D. Source already near ceiling

换更中等难度 checkpoint。

---

# 31. 如果最终只有 LIBERO 很强怎么办

不要把 paper 写成 universal。

可收缩为：

> proactive stochastic risk control across multiple VLAs under diverse visual perturbations.

然后：

- 3 VLA；
- LIBERO 4 suites；
- LIBERO-X/Plus；
- strong zero/few-shot；
- detailed closed-loop analysis。

这仍然可以形成有力论文，但 multi-simulator claim 必须删掉。

---

# 32. 接下来 72 小时具体做什么

## 今天

1. 等 R10 full-trace diagnostic 完成；
2. 生成 root-cause verdict；
3. 冻结 `RASE_VNEXT_PROTOCOL_v1.md`；
4. 禁止其他 training 自动启动。

---

## 明天

实现：

- `CanonicalRobotSpec`
- variable-view observation
- semantic action token
- `CorrectionOperator`
- `continue`
- `requery`
- `resample`
- `persistent_fallback`

并写 round-trip tests。

---

## 第 3 天

做：

> 20–40 state × 2 VLA × 4 correction branches 的 counterfactual smoke。

只看：

- branch correctness；
- success/cost variability；
- state-dependent oracle gap。

**不要训练模型。**

---

# 33. 一周以后你应该得到的唯一关键答案

不是：

> “AUC 有没有 0.8？”

而是：

> **“在自然、outcome-independent 的实时决策点上，是否存在足够多的状态，使得不同纠错操作具有不同最优价值，而且这些状态的 urgency 可以从当前可部署 history/action 中预测？”**

如果答案是 YES：

> 项目真正进入一个强方法论文轨道。

如果答案是 NO：

> 立刻停止 selector escalation，换 target/纠错操作，而不是再堆模型。

---

# 34. 最终论文成功条件

我建议把 submission-ready gate 固定成：

### Method

- proactive risk
- urgency
- operator portfolio
- stochastic uncertainty
- abstention

### Generalization

- 3 VLA
- ≥2 simulator，目标 3
- zero-shot positive macro trend
- few-shot significant improvement

### Control

- success improvement or strong success-cost Pareto
- paired harm ≤5%
- early rescue before failure
- unnecessary intervention controlled

### Reproducibility

- task-level split
- fresh validation
- paired closed-loop
- adapter parity
- config/checkpoint hashes

### CVPR relevance

- visual OOD / camera/object perturbation analysis
- multi-view temporal visual risk
- strong qualitative trajectories/videos

---

# 35. 最后一个战略建议

从现在开始，项目不要再以：

> “risk model accuracy”

作为最高层目标。

真正的最高层目标应该固定为：

\[
\boxed{
\text{Prevent avoidable failures early}
+
\text{choose the cheapest effective correction}
+
\text{transfer across frozen VLAs}
}
\]

所有 AUROC、hazard、world-model、adapter 都只是服务于这个目标。

如果你能做到：

> **一个冻结的 shared RASE 在未见 VLA 上 zero-shot 带来小但一致的 improvement；只用几十条 trajectories 校准后获得明显更大的 improvement；同时在至少三个 benchmark 上展示提前于失败的 intervention 和成功—成本 Pareto 改善，**

这会比“又一个更准的 failure predictor”强很多，也最贴合你当前已有的所有正负实验。
