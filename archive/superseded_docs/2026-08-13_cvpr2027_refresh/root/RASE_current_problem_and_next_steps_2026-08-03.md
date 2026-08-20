# RASE 当前问题诊断与下一阶段执行方案

**日期：** 2026-08-03  
**当前阶段：** PRE-A2 已完成，准备进入 PRE-A2.5 / PRE-A3  
**当前审计状态：**

```yaml
status: duration_structure_signal
critic_gate: replication_required
world_model_gate: closed
```

---

## 0. 执行摘要

当前结果说明，RASE 仍然值得继续推进，但研究问题需要进一步收紧。

PRE-A2 已经证明：

1. 短时间 OFT 接管后再交还 SmolVLA，不能稳定恢复任务；
2. 恢复存在明显的状态依赖和时间结构；
3. 固定接管时长会产生非单调结果和对原成功状态的误伤；
4. persistent OFT 明显强于有限时长 handback，说明主要瓶颈很可能位于恢复后的重新进入和交还阶段。

但 PRE-A2 **没有证明**：

- 同一 SmolVLA 的严格重采样无效；
- 使用最新观测重新规划无效；
- 候选集中不存在成功动作；
- 成功候选存在但无法被评价器识别；
- SmolVLA 在经过局部状态修复后仍然无法继续；
- 世界模型或 candidate critic 必然必要。

因此，当前最重要的问题不是立即训练 termination model、world model 或新的 action replanner，而是通过严格的同状态反事实实验，区分以下几类瓶颈：

\[
\text{坏采样}
\neq
\text{旧动作块失效}
\neq
\text{候选评价失败}
\neq
\text{策略支持集之外}
\neq
\text{恢复后交还失败}.
\]

下一阶段应分成两条互补但独立的实验线：

- **PRE-A3-R：恢复持续时间与重新进入结构复现；**
- **PRE-A3-S：同策略 resample / replan / guided generation 的机会空间审计。**

只有在同策略候选空间被证明存在足够大的 oracle headroom 后，才训练 candidate-conditioned critic；只有 critic 在 held-out 条件下仍不能充分评价候选，且世界模型证据提供稳定独立增量时，才重新打开 world model gate。

---

# 1. 当前实验状态

## 1.1 PRE-A2 已完成内容

当前已完成：

- 12 条 persistent OFT 闭环恢复轨迹；
- 60 个配对 handback rollout；
- 恢复长度：

\[
h \in \{0,8,16,32,64\};
\]

- 所有模型冻结；
- 没有训练 world model、critic 或 replanner。

实验结果：

| OFT 接管长度 | 成功率 | 对原成功状态的误伤 |
|---:|---:|---:|
| 0 | 3/12 | — |
| 8 | 3/12 | 0 |
| 16 | 3/12 | 0 |
| 32 | 2/12 | 1 |
| 64 | 5/12 | 0 |
| Persistent OFT | 11/12 | — |

已完成的工程工作包括：

- `scripts/generate_oft_recovery_trajectories.py`
- `scripts/run_recovery_duration12.sh`
- `scripts/analyze_recovery_duration.py`
- `scripts/run_oft_verify_suites.sh`
- `tests/test_recovery_duration.py`
- `tests/test_runner_hardening.py`
- `progress/2026-08-03_rase_pre_a2_recovery_duration.md`
- `progress/README.md`

验证结果：

```text
ruff: passed
pytest: 10 passed
git diff --check: passed
```

服务器本地提交：

```text
8d82126 Measure closed-loop recovery duration
```

---

# 2. PRE-A2 实际证明了什么

## 2.1 短时 OFT 接管不足以恢复 SmolVLA 的后续闭环能力

当 \(h=8\) 或 \(h=16\) 时，成功率仍然为 \(3/12\)，与 \(h=0\) 完全一致。

这说明短暂使用 OFT 并没有可靠地把系统带回到一个 SmolVLA 可以继续完成任务的状态。

但这里必须使用准确表述：

> PRE-A2 证明的是“短时 OFT 接管后交还 SmolVLA不能稳定恢复”，而不是“SmolVLA 的 replan 或 resample 无效”。

当前实验中的关键变量是：

\[
\text{OFT 控制物理系统的持续时间},
\]

不是：

\[
\text{SmolVLA 在同一个 snapshot 上的重新采样或重新条件化}.
\]

因此，不能从 PRE-A2 直接推出同策略 self-repair 没有机会空间。

---

## 2.2 恢复具有真实的状态和时间结构

Spatial-camera 和 Goal-camera 状态只有在 OFT 接管到 64 步时才恢复。

这说明恢复不是单个 action chunk 的局部修正，而可能需要完成某个物理或语义阶段，例如：

- 重新进入可视区域；
- 恢复合理的相机—物体几何关系；
- 完成重新对齐；
- 形成稳定抓取；
- 完成一个子任务；
- 到达 SmolVLA 熟悉的状态分布；
- 消除由此前错误动作导致的接触或任务进度偏差。

因此，“恢复持续时间”确实是一个机制变量，但它更可能是隐藏恢复进度的代理，而不是最终应学习的变量。

真正需要建模的可能是：

\[
P(
\text{base policy 可成功继续}
\mid
h_t,\ \text{recovery progress}
).
\]

---

## 2.3 固定持续时间不是安全的恢复机制

Goal-clean 原本成功，但在 \(h=32\) 时失败，在 \(h=64\) 时又恢复成功。

该非单调结果说明：

1. 恢复过程可能经过暂时不适合交还的中间状态；
2. 交还过早会破坏 OFT 已经建立但尚未完成的恢复动作；
3. 固定时间不能判断恢复是否完成；
4. 某些原本成功的状态会因不必要干预而受损；
5. “执行更久”并不等价于“始终更安全”。

因此，不能部署如下规则：

```text
检测失败
→ 固定执行 OFT 32 或 64 步
→ 无条件交还
```

更合理的机制应是：

```text
恢复策略执行
→ 连续观察恢复状态
→ 判断 base policy 是否重新具备成功能力
→ 仅在满足保守交还条件时 handback
```

---

## 2.4 Persistent OFT 与有限 handback 的差距是当前最重要信号

Persistent OFT 达到 \(11/12\)，而 \(h=64\) 只有 \(5/12\)。

这意味着剩余的 6 个状态至少存在三种可能：

### 情况 A：SmolVLA 从未重新进入可用区域

OFT 可以一直完成任务，但沿着 OFT 的成功轨迹，SmolVLA 在任何中间状态都无法可靠接管。

此时 termination predictor 无论多准确都无法解决问题，因为不存在合适的 handback 点。

### 情况 B：存在 handback 点，但固定 \(h\) 没有扫描到

SmolVLA 可能在 \(h=72\)、\(96\)、\(128\) 或某个短暂阶段可以成功接管。

此时需要更密集的 handback tomography，而不是立即训练模型。

### 情况 C：同一个 handback 状态具有采样随机性

单次 SmolVLA continuation 失败，但多个 seed 中可能存在成功 continuation。

此时当前的单 rollout 评估会低估真实 competence。

因此，下一步必须估计：

\[
g(\tau)
=
P_{\xi}
\left(
\text{SmolVLA 成功}
\mid
s_\tau^{OFT}
\right),
\]

而不是只记录某个 \(\tau\) 上一次 rollout 的二元结果。

---

# 3. 当前真正卡在哪里

当前不能简单总结为：

> “策略遇到错误后无法有效 replan 或 resample。”

更准确的诊断是：目前尚未区分五个不同层次的问题。

## 3.1 单次坏采样：Sampling Failure

当前 SmolVLA 条件分布中存在成功动作，但当前 noise seed 产生了失败动作。

形式上：

\[
\exists a^\star \sim \pi(a\mid h_t,l)
\quad
\text{s.t.}
\quad
Y(a^\star)=1,
\]

但当前采样 \(a^0\) 失败。

识别方法：

- 从完全相同的 snapshot、观测、历史、cache 初始化和采样配置出发；
- 只改变随机 seed；
- 运行多个严格同分布 resample；
- 计算 strict-resample oracle。

对应解决方法：

- Best-of-\(K\)；
- candidate-conditioned critic；
- conservative candidate replacement；
- critic-guided sampling。

---

## 3.2 旧 action chunk 失效：Stale-Chunk Failure

当前 action suffix 是基于较早观测生成的。环境已经因为执行、接触、相机变化或物体移动发生变化，但策略仍在执行旧 chunk。

此时简单重采样旧条件分布可能没有意义，真正需要的是：

- 截断未执行 suffix；
- 获取最新观测；
- 更新 proprioception；
- 重建 history；
- 重置或重建 policy cache；
- 使用最新条件重新调用同一个 SmolVLA。

该操作是 fresh replan，而不是 strict resample。

识别方法：

\[
H_{\text{recondition}}
=
S_{\text{oracle-resample+replan}}
-
S_{\text{oracle-resample}}.
\]

如果该差值明显为正，说明新观测和重新条件化比增加随机 seed 更重要。

---

## 3.3 候选存在但系统不会选择：Evaluation Failure

可能存在多个候选，其中至少一个可以成功，但系统无法提前识别。

此时：

\[
S_{\text{oracle@K}}
\gg
S_{\text{selected@K}}.
\]

这是 candidate evaluation 问题，而不是 candidate generation 问题。

对应解决方法：

- 使用真实 continuation outcome 训练 candidate-conditioned critic；
- 输入必须包含具体 candidate action；
- 使用 pairwise ranking；
- 独立做 calibration；
- 报告 harmful replacement；
- 使用 lower confidence bound；
- 当所有候选都不可靠时 abstain，而不是硬选相对最优候选。

---

## 3.4 当前状态不在 SmolVLA 支持集：State-Support Failure

如果所有同状态 SmolVLA resample 和 replan 都失败，说明问题可能不在采样，而在当前状态本身。

例如：

- 摄像机视角严重偏移；
- 物体已经进入训练分布外的位置；
- 抓取关系异常；
- gripper 与物体发生非典型接触；
- 已完成进度和视觉状态不一致；
- 机器人处于策略未见过的局部构型。

此时从相同状态继续采样，可能只会反复生成同类错误。

需要先执行低层状态修复：

```text
RETREAT
LIFT_AND_REOBSERVE
STABILIZE
REALIGN
REGRASP
OPEN_RESET_GRIPPER
```

然后从新的真实状态重新调用 **同一个 SmolVLA**。

这仍然属于同策略 self-repair，不等于更换任务策略。低层 primitive 的作用是恢复物理状态，而不是替代 SmolVLA 完成任务。

---

## 3.5 恢复完成但交还失败：Re-entry / Handback Failure

即使 OFT 或局部修复已经产生进展，SmolVLA 也不一定立即能接管。

交还条件不能只依赖：

- 已执行步数；
- recovery policy confidence；
- 当前动作是否结束；
- 是否达到固定 horizon。

需要直接建模：

\[
P(
\text{handback 后 base 成功}
\mid
h_t,\ d_t,\ \pi_{\text{base}},\ \pi_{\text{recovery}}
).
\]

但在训练该模型前，必须先证明：

1. 确实存在可复现的成功 handback 状态；
2. 不同状态的最小充分恢复程度不同；
3. 可以提前 handback 而不明显误伤 base success；
4. oracle handback 相对最佳固定持续时间有足够 headroom。

---

# 4. 研究问题应如何重新定义

不建议继续使用宽泛叙事：

> VLA 失败后切换到更强策略，再决定何时切回。

建议把主问题收紧为：

> **冻结 VLA 的失败中，哪些可以通过同策略重采样、重新条件化或引导生成完成自我修复，哪些必须先恢复物理状态，以及系统如何在不造成额外伤害的前提下区分这些情况？**

英文可表述为：

> **When can a frozen VLA repair itself through resampling, reconditioning, or guided action generation, and when must the physical state be restored before replanning becomes effective?**

建议保留三个核心贡献方向：

1. **Exact-snapshot self-repair benchmark**  
   在同一 simulator snapshot 上严格比较 current、resample、replan、guided sample 和 post-restoration replan。

2. **Proposal–evaluation–support decomposition**  
   区分候选生成不足、候选评价失败和策略支持集失败。

3. **Harm-constrained self-repair**  
   只有在替代候选具有可信绝对增益时才替换当前动作；当候选无解时进行状态修复或 abstain。

---

# 5. 下一阶段总体路线

建议把下一阶段拆成两个相互独立的实验协议。

---

# 6. PRE-A2.5：Dense Handback Tomography

## 6.1 目标

回答：

> 沿着 persistent OFT 成功恢复轨迹，SmolVLA 在什么时候重新获得闭环成功能力？

## 6.2 实验对象

优先使用：

- 当前 11 条 persistent OFT 成功轨迹；
- 尤其关注 h=64 仍不能 handback 成功的 6 个状态；
- 包含 clean、camera、robot 扰动；
- 对 Spatial-camera、Goal-camera 和 Goal-clean 做重点分析。

## 6.3 时间点

建议扫描：

\[
\tau \in
\{0,8,16,24,32,40,48,56,64,72,80,96,112,128\},
\]

并加入：

- OFT 子任务结束点；
- 接触状态变化点；
- grasp 建立或丢失点；
- object relation 明显变化点；
- task progress milestone；
- OFT 动作 chunk 边界。

## 6.4 每个时间点的 continuation

对每个 \(\tau\)：

- restore 完全相同 simulator snapshot；
- 重建公开 history；
- 重新初始化 SmolVLA cache；
- 使用 \(K=4\) 或 \(K=8\) 个 seed；
- 运行完整 closed-loop continuation；
- 记录成功率而不是单个二元结果。

定义：

\[
g_i(\tau)
=
\frac{1}{K}
\sum_{k=1}^{K}
Y_{i,\tau,k}.
\]

## 6.5 需要分类的轨迹类型

### 类型 1：Monotonic Re-entry

\[
g(\tau)
\]

随恢复进展总体提高，并在某个阶段稳定超过阈值。

说明可以学习 termination/competence predictor。

### 类型 2：Transient Re-entry

存在短暂可 handback 区间，之后又下降。

说明交还机会是阶段性的，不能只使用累计时长。

### 类型 3：Non-monotonic Re-entry

成功概率上下波动。

说明需要状态语义、接触、进度或视觉支持特征，而不是单变量 duration。

### 类型 4：No Re-entry

沿 persistent OFT 成功轨迹始终：

\[
g(\tau)\approx 0.
\]

说明 SmolVLA 无法从这些中间状态继续，termination model 没有意义；需要同策略 guided generation、更强状态恢复或承认 base competence boundary。

## 6.6 PRE-A2.5 输出

- 每条轨迹的 \(g_i(\tau)\) 曲线；
- 最早稳定 handback 点；
- 可 handback 区间长度；
- 非单调状态数量；
- no-re-entry 状态数量；
- 单 seed 与多 seed competence 判断差异；
- duration 与真实恢复事件之间的对应关系；
- 是否存在足够大的 oracle handback headroom。

---

# 7. PRE-A3-R：恢复持续时间独立复现

## 7.1 目标

验证 PRE-A2 的 duration structure 是否在独立任务和 episode 上复现。

## 7.2 数据规模

建议：

- 96–120 个 task/episode-disjoint 状态；
- 四个 LIBERO suite 平衡；
- clean / camera / robot 扰动平衡；
- natural failure 与 controlled perturbation 分开报告；
- 任务阶段平衡；
- 保留 hidden confirmation split。

## 7.3 接管长度

\[
h \in \{0,8,16,32,64,96,128\}.
\]

## 7.4 执行要求

必须改为 live closed-loop OFT 接管：

- 每一步重新观察；
- OFT 实时生成动作；
- 不使用先捕获后 deterministic replay 作为主结果；
- handback 时重建 SmolVLA public history；
- 记录 policy warm-up 和 first-action latency；
- 所有模型继续冻结。

deterministic replay 可以保留为诊断对照，但不能作为主要机制证据。

## 7.5 核心指标

- terminal rescue rate；
- false-handback harm；
- base-success preservation；
- persistent-OFT gap；
- recovery action cost；
- intervention latency；
- non-monotonic duration count；
- persistent-OFT-only count；
- earliest successful handback；
- task/episode cluster bootstrap confidence interval。

## 7.6 Hidden confirmation split

开发阶段可以：

- 选择指标；
- 确定统计代码；
- 固定阈值；
- 定义稳定 handback；
- 调试运行器。

但不能根据 hidden split：

- 修改 threshold；
- 修改 horizon；
- 修改状态筛选；
- 删除不利任务；
- 重定义 success；
- 调整模型。

## 7.7 打开 termination model gate 的条件

必须同时满足：

1. 至少两个任务族存在稳定 rescue；
2. 不同状态的最小充分恢复程度确实不同；
3. oracle adaptive handback 明显优于最佳 fixed duration；
4. 可提前 handback 并降低 OFT action cost；
5. 对 base-success 的误伤可被控制；
6. hidden split 上方向保持。

否则：

```yaml
termination_model_gate: closed
```

---

# 8. PRE-A3-S：同策略 Self-Repair Opportunity Screen

## 8.1 目标

直接回答：

> 当 SmolVLA 当前动作可能失败时，同一个 SmolVLA 是否能够通过 resample、fresh replan 或 guided generation 产生成功替代候选？

这是决定主 idea 是否真正成立的关键实验。

## 8.2 Snapshot 类型

必须包含：

- clean successes；
- natural failures；
- controlled camera perturbation；
- controlled robot perturbation；
- near-failure states；
- silent failure states；
- benign high-risk states；
- contact 和 non-contact 状态；
- 不同任务阶段。

不能只采失败状态，否则无法测量：

- false intervention；
- harmful replacement；
- 当前动作本来成功时是否被错误替换；
- 不必要 replan 的成本。

## 8.3 第一阶段候选池

每个相同 snapshot 建议生成：

```text
1 × current suffix
8 × strict same-distribution resamples
4 × fresh same-policy replans
4 × adaptive-horizon replans
1 × safe hold / abstain
```

第一阶段不加入 fallback VLA。

原因是当前研究目标是判断 **SmolVLA 是否可以自我纠错**，而不是通过更换策略获得收益。

## 8.4 Strict Resample 定义

严格保持：

- 相同物理状态；
- 相同 RGB / proprioception；
- 相同历史窗口；
- 相同语言指令；
- 相同 policy cache 初始化；
- 相同 temperature；
- 相同 top-p / top-k；
- 相同 flow / diffusion 配置；
- 相同 action horizon；
- 只改变 noise seed。

否则不能称为同分布 resample。

## 8.5 Fresh Replan 定义

Fresh replan 应：

- 丢弃旧 suffix；
- 使用当前最新 observation；
- 使用当前 proprioception；
- 重建最近历史；
- 重建 cache；
- 在相同 SmolVLA 上生成新 chunk。

它测量的是新信息和重新条件化的价值，而不是随机 seed 的价值。

## 8.6 Adaptive-Horizon Replan

固定长 action chunk 可能造成：

- 观测利用不足；
- 接触后旧动作仍继续执行；
- 小误差累积；
- 相机扰动后无法快速重新条件化。

因此建议比较执行 horizon：

\[
e \in \{1,2,4,8\}
\]

或比较：

- full chunk；
- receding horizon；
- contact-triggered truncation；
- deviation-triggered truncation。

注意要区分：

- generation horizon；
- execution horizon。

模型可以生成长 chunk，但只执行前若干步后重新观察。

## 8.7 第二阶段候选池：Critic-Guided SmolVLA Sampling

只有普通候选 oracle 存在明显 headroom 后，再加入：

```text
4 × critic-guided SmolVLA flow candidates
```

目标分布可以概念化为：

\[
q_\beta(a\mid h)
\propto
\pi_{\text{SmolVLA}}(a\mid h)
\exp(\beta \hat Q(h,a)).
\]

必须加入：

- policy prior trust region；
- action smoothness constraint；
- workspace constraint；
- uncertainty-aware guidance；
- diversity regularization；
- guidance strength sweep；
- matched compute baseline。

这仍然属于冻结 SmolVLA 的 inference-time action steering，不是替换任务策略。

## 8.8 第三阶段候选：状态修复后同策略 replan

当所有 same-state SmolVLA 候选失败时，加入少量固定 local corrections：

```text
RETREAT_SHORT
LIFT_AND_REOBSERVE
STABILIZE_HOLD
REALIGN
OPEN_RESET_GRIPPER
```

执行后：

1. 观察新的真实状态；
2. 使所有旧候选失效；
3. 使所有旧价值预测失效；
4. 重建 history；
5. 重新调用 SmolVLA；
6. 再生成 resample / replan 候选。

这一步用于测量：

\[
H_{\text{state-restoration}}
=
S_{\text{post-restoration}}
-
S_{\text{same-state oracle}}.
\]

---

# 9. PRE-A3-S 的核心分解指标

定义嵌套成功率：

\[
S_0
=
S_{\text{current}},
\]

\[
S_1
=
S_{\text{current+resample oracle}},
\]

\[
S_2
=
S_{\text{current+resample+replan oracle}},
\]

\[
S_3
=
S_{\text{current+resample+replan+adaptive horizon oracle}},
\]

\[
S_4
=
S_{\text{guided candidate oracle}},
\]

\[
S_5
=
S_{\text{post-restoration oracle}}.
\]

对应增量：

\[
H_{\text{sampling}}
=
S_1-S_0,
\]

\[
H_{\text{reconditioning}}
=
S_2-S_1,
\]

\[
H_{\text{horizon}}
=
S_3-S_2,
\]

\[
H_{\text{guidance}}
=
S_4-S_3,
\]

\[
H_{\text{restoration}}
=
S_5-S_4.
\]

这些量分别回答：

1. 是否只是坏 seed；
2. 新观测是否能纠正旧 chunk；
3. 更高频闭环是否有价值；
4. 是否需要主动改变 proposal distribution；
5. 当前状态是否必须先被物理修复。

---

# 10. Candidate Generator Go / No-Go 门槛

在训练任何复杂 selector、world model 或 escalation policy 前，建议要求：

## 10.1 Candidate Headroom

\[
S_{\text{oracle candidate}}
-
S_{\text{current}}
\ge 8\text{–}10\text{ percentage points}.
\]

## 10.2 Failure Rescue Coverage

在 base failure 中，至少约 20% 存在成功同策略替代候选。

## 10.3 Generator Diversity

至少两个 generator family 具有稳定独有成功，例如：

- strict resample 独有成功；
- fresh replan 独有成功；
- adaptive horizon 独有成功；
- guided sample 独有成功。

## 10.4 跨任务稳定性

收益不能只集中于：

- 一个 task；
- 一个 perturbation；
- 一个 suite；
- 一个特定阶段；
- 一个 seed cluster。

## 10.5 Held-Out 保持

task-held-out 或 episode-held-out pilot 上，收益方向必须保持。

如果不满足：

```yaml
candidate_critic_gate: closed
world_model_gate: closed
```

此时应优先改进 proposal generator，而不是训练评价器。

---

# 11. Candidate Critic 应如何训练

只有 candidate oracle headroom 足够大时才训练。

## 11.1 第一版模型

第一版不要使用 world model。

输入：

\[
(h_t,c)
\]

其中：

- 当前多帧观测；
- proprioception；
- action history；
- 语言指令；
- candidate action chunk；
- generation metadata；
- execution horizon。

输出：

\[
\hat p_c^{success},
\quad
\hat p_c^{harm},
\quad
\hat p_c^{feasible},
\quad
\hat r_c^{progress},
\quad
\hat\sigma_c.
\]

## 11.2 训练监督

使用真实同状态 continuation outcome：

- terminal success；
- object drop；
- grasp loss；
- collision；
- progress regression；
- irreversible transition；
- continuation cost；
- latency；
- candidate completion。

## 11.3 损失

建议包括：

- success BCE；
- harm BCE；
- feasibility BCE；
- pairwise ranking；
- progress regression；
- calibration loss 或独立后校准。

## 11.4 部署规则

不能只选择预测值最大的候选。

应使用保守规则：

\[
c^\star
=
\arg\max_c
LCB(\hat U_c).
\]

只有满足：

\[
LCB(
\hat U_{c^\star}
-
\hat U_{\text{current}}
)
>
\delta,
\]

并且：

\[
UCB(\hat p_{c^\star}^{harm})<\eta_h,
\]

才替换当前动作。

否则：

- 继续 current；
- 收集更多候选；
- 进入状态修复；
- 或安全 abstain。

## 11.5 关键指标

- selected success；
- captured oracle gap；
- ranking regret；
- beneficial replacement precision；
- harmful intervention rate；
- false replacement on base success；
- expected calibration error；
- risk–coverage；
- success per extra GPU second。

---

# 12. World Model 为什么现在仍应关闭

当前阶段 world model 不是最需要解决的问题。

其局限包括：

1. 如果候选空间中没有成功动作，world model 无法凭空创造动作；
2. 接触、抓取和物体动力学是最难预测的区域；
3. imagined ranking 可能对 OOD candidate 错误自信；
4. 训练数据、评测数据和 candidate-value 数据隔离复杂；
5. 会增加延迟和论文复杂度；
6. 容易稀释当前最清晰的 self-repair / competence-boundary 贡献。

因此应保持：

```yaml
world_model_gate: closed
```

重新打开 gate 的前提：

1. candidate oracle headroom 明显；
2. no-WM critic 能捕获一部分但不是全部 oracle gain；
3. 主要残余错误与短时未来结果有关；
4. action-conditioned WM 超过 history-only critic；
5. imagined ranking 与真实 ranking 稳定相关；
6. held-out perturbation 或 VLA 上仍有增量；
7. 增益能够覆盖额外计算和延迟。

如果不满足，world model 只能作为 optional evidence 或 baseline。

---

# 13. OFT 在新路线中的角色

为了坚持“不通过换策略提升”，建议把 OFT 从主部署方法降为以下角色：

## 13.1 Recovery Oracle

用于判断：

- 当前状态是否存在更强闭环控制能力；
- persistent recovery 上限；
- 最小状态恢复程度；
- base policy competence boundary。

## 13.2 Counterfactual State Generator

OFT 可以生成：

- 从失败状态到成功状态的恢复轨迹；
- 多个恢复中间 snapshot；
- handback tomography 数据；
- re-entry 正负样本。

## 13.3 Teacher for Local Recovery Semantics

分析 OFT 成功恢复轨迹中的关键事件：

- retreat；
- re-align；
- regrasp；
- stabilize；
- reposition；
- recover task progress。

这些事件可以帮助设计固定 local primitives，但不直接把 OFT 作为最终任务策略。

## 13.4 Strong Upper Bound

报告：

```text
Base-only
Same-policy resample/replan
Same-policy guided generation
State-restoration + same-policy replan
Persistent OFT upper bound
```

这样可以清楚展示：

- 同策略纠错能够捕获多少能力；
- 剩余 gap 是否来自 SmolVLA 的根本能力不足。

---

# 14. 建议的系统主流程

```text
SmolVLA current suffix
        ↓
Risk / deviation / stale-chunk trigger
        ↓
Is current candidate still reliable?
  ├── Yes → execute conservatively
  └── No
        ↓
Generate same-policy candidates
  - strict resamples
  - fresh replans
  - adaptive-horizon replans
  - critic-guided SmolVLA samples
        ↓
Candidate-conditioned conservative ranking
        ↓
Reliable improvement exists?
  ├── Yes → execute selected SmolVLA candidate
  └── No
        ↓
Is state restoration safe and feasible?
  ├── Yes → local repair / lift / retreat / realign
  │         ↓
  │       observe new state
  │         ↓
  │       invalidate old candidates
  │         ↓
  │       rerun SmolVLA candidate generation
  └── No → safe hold / abstain / human
```

该系统的核心原则是：

> 优先在同一个任务策略中寻找可行纠正；只有当前物理状态不被策略支持时，才使用低层状态修复。状态修复完成后仍由原 SmolVLA 重新规划任务动作。

---

# 15. 立即需要实现的代码工作

## 15.1 Handback Tomography Runner

新增建议：

```text
scripts/run_handback_tomography.py
scripts/analyze_handback_tomography.py
tests/test_handback_tomography.py
```

功能：

- 从 persistent OFT 轨迹选择时间点；
- restore snapshot；
- 多 seed SmolVLA continuation；
- 计算 \(g_i(\tau)\)；
- 输出 re-entry 类型；
- 检测非单调 competence；
- 生成 task/episode bootstrap。

## 15.2 Same-State Candidate Runner

新增建议：

```text
scripts/generate_same_state_candidates.py
scripts/run_same_state_candidate_audit.py
scripts/analyze_candidate_headroom.py
tests/test_same_state_candidate_fidelity.py
```

必须保存：

- snapshot hash；
- policy seed；
- cache initialization；
- candidate generator type；
- candidate action；
- generation horizon；
- execution horizon；
- sampling profile；
- continuation outcome；
- wall-clock 和 GPU time。

## 15.3 Snapshot Fidelity Audit

需要验证：

- physics state hash；
- controller state；
- RGB parity；
- proprioception parity；
- contact parity；
- task-state parity；
- current suffix parity；
- RNG state；
- sensor latency；
- restore 后 base replay parity。

## 15.4 Candidate Decomposition Audit

审计器输出：

```yaml
current_success:
strict_resample_oracle:
fresh_replan_oracle:
adaptive_horizon_oracle:
guided_sampling_oracle:
post_restoration_oracle:
resample_unique_success:
replan_unique_success:
guided_unique_success:
all_same_state_fail:
harmful_replacement_opportunities:
candidate_oracle_headroom:
critic_gate:
world_model_gate:
```

---

# 16. 推荐执行顺序

## 阶段 1：完成 PRE-A2.5

优先回答：

- persistent OFT 成功轨迹上是否存在可 handback 点；
- fixed duration 是否只是漏掉了正确阶段；
- 多 seed 是否改变 handback 判断；
- 哪些状态完全 no-re-entry。

## 阶段 2：运行 PRE-A3-S 小规模 pilot

规模建议：

- 12–16 个 tasks；
- 100–200 个 independent snapshots；
- 每个状态 8 strict resamples；
- 4 fresh replans；
- 多 execution horizons；
- 暂不训练模型。

首先只计算 oracle headroom。

## 阶段 3：决定是否训练 candidate critic

只有 candidate oracle 足够大时：

- 训练 observation/history + action critic；
- 做 pairwise ranking；
- 做 task-held-out；
- 独立 calibration；
- closed-loop conservative selection。

## 阶段 4：加入 critic-guided SmolVLA sampling

当普通 resample/replan 有一定 headroom但覆盖不足时：

- 使用 critic 引导 flow sampling；
- matched compute 比较；
- 加入 trust region 和 uncertainty gate。

## 阶段 5：完成 PRE-A3-R 独立复现

验证 duration / re-entry 结构是否跨任务稳定。

## 阶段 6：决定 termination predictor 和 world model gate

- termination predictor 只在 re-entry oracle 有 headroom 时训练；
- world model 只在 no-WM critic 的剩余误差具有可预测短时动力学结构时训练。

---

# 17. 关键 Go / No-Go 决策树

```text
PRE-A2.5:
沿 OFT 轨迹是否存在稳定 re-entry 点？
  ├── No
  │    → 不训练 termination predictor
  │    → 研究 base competence boundary
  │    → 测试 guided same-policy generation / state restoration
  └── Yes
       ↓
oracle adaptive handback 是否明显优于 fixed duration？
  ├── No
  │    → 使用简单 fixed / event-based handback
  └── Yes
       → 打开 termination predictor gate

PRE-A3-S:
same-policy candidate oracle 是否明显高于 current？
  ├── No
  │    → 不训练 candidate critic
  │    → 改善 proposal generator
  │    → 检测 unsupported state
  └── Yes
       ↓
简单 critic 是否捕获足够 oracle gain？
  ├── Yes
  │    → 不需要 world model
  └── No
       ↓
WM evidence 是否有 held-out 独立增量？
  ├── No
  │    → 保持 no-WM 主线
  └── Yes
       → 打开 world model gate
```

---

# 18. 论文叙事建议

## 18.1 推荐标题方向

### 方向一

> **Can a Frozen VLA Repair Itself? Decomposing Resampling, Replanning, and State Restoration**

### 方向二

> **Where Does VLA Self-Recovery Come From? Exact-Snapshot Analysis of Proposal, Evaluation, and Competence Boundaries**

### 方向三

> **Try Again or Restore the State? Harm-Constrained Self-Repair for Frozen VLA Policies**

## 18.2 推荐主 Claim

可以提出：

- 冻结 VLA 的运行时失败应分解为 sampling、reconditioning、evaluation 和 state-support failure；
- strict resample 与 fresh replan 测量不同机制；
- exact-snapshot continuation 可以直接测量 self-repair oracle headroom；
- 固定恢复时长不是可靠的 handback 机制；
- 状态改变后必须废弃旧候选并从新的真实状态重新规划；
- 同策略 guided generation 与低层 state restoration 可以在不替换任务策略的情况下实现纠错；
- candidate replacement 必须显式约束对 base-success 的误伤。

暂时不能提出：

- resampling 一定有效；
- fresh replan 一定有效；
- world model 一定必要；
- adaptive termination 一定优于最佳 fixed duration；
- SmolVLA 可以恢复所有 OFT 可恢复状态；
- guided sampling 一定能弥补策略能力不足；
- 仿真 exact-snapshot oracle 等价于真实机器人反事实。

---

# 19. 当前最大的研究风险

## 风险一：同策略候选空间没有 headroom

如果：

```text
current fails
all strict resamples fail
all fresh replans fail
all adaptive horizons fail
guided samples fail
```

则 SmolVLA 的问题不是选择器，而是策略支持不足。

应转向：

- unsupported-state detection；
- state-restoration benchmark；
- competence boundary；
- 何时应 abstain；
- 更强 base policy 作为独立实验，而不是恢复算法。

## 风险二：Oracle 很高但 critic 选不出来

说明问题在 candidate evaluation。

应：

- 增强 candidate action representation；
- 使用 pairwise labels；
- 做 action-shuffle probe；
- 检查 task ID / generator ID shortcut；
- 加强校准和 uncertainty；
- 将论文贡献收紧为 proposal–evaluation gap。

## 风险三：Fresh replan 有效但 resample 无效

这是合理且有价值的结论：

> 失败主要来自 stale chunk 和闭环频率不足，而不是随机采样偶然性。

此时方法重点应变为：

- deviation-triggered truncation；
- adaptive execution horizon；
- event-driven replanning；
- cache reset / history re-anchoring。

## 风险四：State restoration 有效，同状态候选无效

这说明真正问题是 competence region，而不是 candidate ranking。

论文可以转向：

- 如何识别 unsupported state；
- 如何用最小物理修复重新进入支持集；
- 如何证明状态修复比换策略更低侵入。

## 风险五：Persistent OFT 从头到尾明显支配

必须回答为什么不直接使用 OFT：

- OFT 成本是否更高；
- latency 是否更大；
- 是否需要云端；
- 是否有部署约束；
- 是否存在安全或资源理由；
- SmolVLA 是否在大部分 clean 状态更高效。

如果 OFT 全面更好且成本无差异，则 recovery framing 会被削弱。

---

# 20. 最终建议

当前不应立即训练：

- termination model；
- world model；
- end-to-end escalation policy；
- 新的 action replanner。

当前最优先的工作顺序是：

1. **对 persistent OFT 轨迹做 dense multi-seed handback tomography；**
2. **在 exact same snapshot 上测 strict SmolVLA resample oracle；**
3. **测 fresh SmolVLA replan 的独立增量；**
4. **测更短 execution horizon 和事件触发重规划；**
5. **确认同策略候选 headroom 后再训练 no-WM candidate critic；**
6. **普通候选覆盖不足时再做 critic-guided SmolVLA sampling；**
7. **所有同状态候选失败时，再测试局部状态修复后重新调用 SmolVLA；**
8. **PRE-A3 独立复现通过后，才打开 termination predictor；**
9. **world model 最后作为增量审计，而不是预设主线。**

最终研究目标应保持为：

> **不是在 VLA 失败后简单切换到更强策略，而是确定冻结 VLA 的错误是否可以通过同策略重采样、重新条件化、引导生成或最小状态修复完成纠正；当无法纠正时，系统应可靠地识别其 competence boundary，并避免无效或有害的动作替换。**

这条路线既保留了“有效 replan / resample / rollback 纠错”的最初目标，也与 PRE-A2 已经观察到的恢复时间结构、非单调 handback 和 persistent-OFT gap 保持一致。
