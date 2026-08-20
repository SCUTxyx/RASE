# RASE vNext：面向多 VLA、多仿真平台的实时风险预测与主动纠错控制层

**日期：** 2026-08-13  
**目标：** CVPR 2027 级别的方法论文  
**项目主线：** 多 VLA / 多 benchmark / 实时预失败风险预测 / 多纠错操作选择 / zero-shot + lightweight post-training  
**基于现有材料：**
- `RASE_CANONICAL_IDEA_stochastic_multi_vla_risk_control_2026-08-13.md`
- `2026-08-13_rase_canonical_idea_rewrite.md`
- `CURRENT.md`

---

## 0. 先给结论

RASE 不应该再被定义成“一个 failure classifier + 一个 fallback selector”。

更有竞争力、也最符合现有实验事实的定义是：

> **RASE 是部署在冻结 VLA 与机器人执行器之间的轻量、实时、模型无关的风险控制层。它在动作真正造成不可恢复错误之前，预测继续执行的短期风险和恢复窗口收缩，并在一个可扩展的纠错操作集合中选择成本最低、成功下界最高的动作：继续安全前缀、重新查询、resample、replan、切换 corrective policy、保持 corrective takeover，或安全停止。**

核心科学问题从：

> “现在会不会失败？”

升级为：

> **“如果现在继续执行，未来短时间内是否会进入更难恢复的区域；若需要干预，哪一种纠错操作在当前状态下最值得执行？”**

这比单纯预测最终 episode success 更适合你的目标，也直接回应当前 R7/R9/R10 的负结果。

---

# 1. 现有实验到底告诉了我们什么

这些负结果不是需要隐藏的东西，而是新版方法设计的依据。

## 1.1 已经被排除的路线

### 1. Success-only operator selector 不成立

你已有 PRE-C0-R3 结果：

- Student：18/72 = 25.0%
- persistent OFT：66/72 = 91.7%
- success oracle：66/72 = 91.7%
- oracle minus best fixed：0pp

如果只优化成功率，persistent OFT 已经等于 oracle。

因此不能把论文写成：

> “学习哪个 policy 更成功。”

必须优化：

> **成功率 + 干预成本 + 延迟 + 风险 + 自动纠错次数。**

---

### 2. “短动作纠正后立即交还 source”不是主要恢复机制

PRE-A1/A2 说明：

- source 的错误往往是策略级错误；
- 短 OFT prefix 能暂时拉回轨迹，但交还以后错误模式会重新出现；
- handback 不能作为默认主方法。

因此：

- persistent corrective takeover 是可靠 baseline；
- handback 只做 gated extension；
- 新版论文不能依赖频繁 source↔fallback 抖动。

---

### 3. 任意时间统一 selector 不成立

R6-C/R6-C.1 已表明：

- Pi0Fast 的 fallback 可恢复性随进入时间快速下降；
- 晚切换会错过机会窗口；
- “高风险以后再 dwell 一段时间确认”会把动作推到不可救区域。

这直接支持新版 RASE 必须建模：

> **urgency / intervention window，而不是只建模 risk magnitude。**

---

### 4. t=0 单帧最终失败预测不够

R7-A：

- 191 states
- 48 tasks
- 5-seed task-held-out
- mean AUROC 0.631
- 0/5 gate pass

这说明不能再假设：

> “任务开始的一帧图像可以稳定预测最终 episode failure。”

新版应该预测更局部、更可观测、直接对应干预价值的目标。

---

### 5. 无条件 one-step hazard 太稀疏

R9：

- 480 boundaries
- 只有 12 hazard positives
- all-causal AUROC 0.829
- 但 policy+horizon prior 0.842

所以问题不是“模型不够大”，而是：

- target 定义不够好；
- positive density 太低；
- 时间 prior 已经解释了大部分信号。

---

### 6. Recoverability 是随机变量

R10-B：

- 66 groups × K3
- 9/66 outcome unstable
- 只有 32/66 保持旧 K2 label
- 旧 33 cases 只有 8 个仍是 stable case

因此：

> **“这个状态一定可救 / 一定不可救”不应作为二元 ground truth。**

必须训练概率：

\[
p_{\mathrm{succ}}(s,c)
\]

以及其置信区间，而不是把小 K rollout 硬化成标签。

---

### 7. 世界模型不能当救火工具

目前已有结果显示：

- pooled V-JEPA latent 不能改善 state-level risk；
- single-step latent delta 也没有产生可靠控制增益。

所以 world model 只能是：

> optional action-conditioned evidence

而不是主方法依赖。

---

# 2. CVPR 论文最应该讲的故事

2025–2026 已经出现多条邻近路线：

- **SAFE**：多任务 VLA failure detection，可在 unseen tasks 上检测失败；
- **FPC-VLA**：failure prediction + correction，并在 LIBERO/SIMPLER 上评估；
- **CycleVLA**：proactive self-correction、subtask backtracking、retry；
- **Pre-VLA**：执行前 action verification + adaptive resampling；
- **DVAC**：利用 flow denoising variance 决定何时提前 replan，覆盖 LIBERO/RoboTwin/CALVIN；
- **StarVLA / vla-eval**：已经把 multi-benchmark VLA evaluation 做成系统化问题。

因此论文不能只声称：

> “我们提前预测失败并纠错。”

这已经不够。

## 2.1 RASE 应该占据的独特位置

建议把 novelty 压在五点组合上：

1. **Model-agnostic shared runtime controller**
   - 不修改 source VLA；
   - 不依赖某种 flow/diffusion 内部量；
   - OpenVLA、π 系列、SmolVLA、StarVLA 类模型都可以接。

2. **Cross-VLA + cross-benchmark**
   - 一个 shared risk core；
   - 只通过 policy/embodiment descriptors 与小型 adapter 适配。

3. **Counterfactual correction portfolio**
   - 不只是“检测失败”；
   - 不只是“超过阈值就 resample”；
   - 而是比较多个可执行纠错操作的未来价值。

4. **Stochastic recoverability + intervention urgency**
   - 直接建模“现在不干预会不会错过恢复窗口”；
   - 不把 recoverability 当 deterministic label。

5. **Zero-shot positive transfer + lightweight post-training gains**
   - zero-shot 作为强挑战指标；
   - few-shot/lightweight controller adaptation 作为主结果；
   - source VLA 本身保持冻结。

这五点联合起来，比单独任何一点都更容易构成顶会方法故事。

---

# 3. 新的核心问题定义

## 3.1 不再主要预测最终 episode failure

最终成功：

\[
Y_{\mathrm{final}}\in\{0,1\}
\]

距离当前动作可能非常远，且在部分可观测环境中有大量 aleatoric randomness。

更合理的是预测多个短期、决策相关变量。

---

## 3.2 Source short-horizon risk

定义：

\[
r_t^{H}
=
P(
\text{continue source causes a critical failure precursor within } H
\mid \mathcal H_t, a_t^{src}, d_{src}
)
\]

其中 `critical failure precursor` 不一定已经 task failed，而可以是：

- 即将碰撞；
- 即将掉落；
- grasp 逐渐丢失；
- 已进入错误 object/goal；
- action 与可行方向持续冲突；
- progress 停滞；
- 继续执行后 corrective success 明显下降。

这是“要失败”的实时定义。

---

## 3.3 Intervention urgency

这是新版最关键的 head。

先定义当前时刻最佳可恢复概率：

\[
V_{\mathrm{rescue}}(t)
=
\max_{c\in\mathcal C}
P(\mathrm{success}\mid s_t,c)
\]

让 source 再执行 \(\Delta\) 步：

\[
V_{\mathrm{rescue}}^{+\Delta}(t)
=
E[
V_{\mathrm{rescue}}(t+\Delta)
\mid \mathrm{continue\ source}
]
\]

定义：

\[
U_t
=
V_{\mathrm{rescue}}(t)
-
V_{\mathrm{rescue}}^{+\Delta}(t)
\]

如果 \(U_t\) 很高：

> 现在不是“风险稍微高”，而是**再等就会错过纠错窗口**。

这正好对应你原始 idea 中“不能等到失败以后才纠正”。

---

# 4. Correction Portfolio：把 fallback 从一个 policy 扩展为一组操作

这是我建议对 canonical idea 做的最大但仍保持主线不变的升级。

风险模型本身仍然**不生成机器人动作**。

它只选择已有 operator。

---

## 4.1 标准 operator 集

\[
\mathcal C_t=
\{
C_{\mathrm{continue}},
C_{\mathrm{short}},
C_{\mathrm{requery}},
C_{\mathrm{resample}},
C_{\mathrm{replan}},
C_{\mathrm{fallback}},
C_{\mathrm{hold/abort}}
\}
\]

### C0 — CONTINUE_SOURCE

执行当前 source chunk 的安全前缀。

---

### C1 — SHORTEN_AND_REQUERY

只执行预测安全的前 \(h'<H\) 步，然后立刻获取新 observation 并重新调用 source。

这是最轻量的纠错。

它和 DVAC 类 adaptive chunking 相关，但 RASE 的选择依据是：

- external visual/state/action risk；
- recoverability urgency；
- correction utility；

而不是只能访问 flow denoising variance。

---

### C2 — SOURCE_REQUERY

直接拒绝尚未执行的 chunk，使用当前 observation 重新 query source。

对于 deterministic model：

- observation 更新以后 requery 才有意义。

---

### C3 — SOURCE_RESAMPLE

若 source 有 stochastic decoding / flow initialization：

- 用不同 seed / sampling noise 生成 \(M\) 个候选；
- RASE candidate verifier 选择风险最低的 chunk。

注意：

> 所有额外 source calls 必须计入 inference cost。

---

### C4 — REPLAN

语义级重规划。

可能实现为：

- source 自带 subtask/planning interface；
- 轻量 planner 产生新的 subgoal；
- VLM 产生重新描述后的 instruction；
- benchmark-specific planner adapter。

为了保持“通用、轻量”：

> **REPLAN 应该是可选 operator，而不是 RASE core 的必要依赖。**

---

### C5 — SWITCH_CORRECTIVE_POLICY

切换到：

- OFT；
- stronger VLA；
- specialized recovery policy；
- benchmark expert。

进入以后默认 persistent takeover。

只有 handback gate 独立成立时才交回。

---

### C6 — SAFE_HOLD / SAFE_ABORT

当：

- source 高风险；
- 所有 correction LCB 都不可靠；
- OOD 太高；

不要强制选择一个“最不差”的错误动作。

这是系统可信度的重要部分。

---

# 5. 两阶段纠错选择：避免每步把所有模型都跑一遍

如果每个 timestep 都：

- query source 4 次；
- query fallback；
- query planner；

方法会非常慢，也无法叫 lightweight。

所以使用两阶段 selector。

---

## Stage A：Operator Prior

只使用：

- history；
- source proposal；
- source descriptor；
- correction operator descriptors；

预测：

\[
\widehat{\Delta}^{prior}_c
\]

不真正调用昂贵 correction。

输出：

- 哪些 operator 值得 query；
- 预计成本；
- 预计收益；
- 置信区间。

---

## Stage B：Candidate Verifier

只有 Stage A 认为值得时才实际生成：

- resample chunk；
- fallback chunk；
- replanned chunk。

然后执行 candidate-conditioned scoring：

\[
\widehat{Q}(s_t,a^{candidate},c)
\]

最终：

\[
c^*
=
\arg\max_c
LCB(Q_c)
-\lambda_{call}C_{call}
-\lambda_{lat}C_{latency}
-\lambda_{harm}C_{harm}
\]

---

# 6. Multi-benchmark 的真正关键：接口必须从 H×7 升级

当前 canonical action：

```text
float[H,7]
```

可以支持 LIBERO/Franka，但不适合作为最终 cross-benchmark interface。

特别是：

- RoboTwin 双臂；
- 不同 robot embodiment；
- joint control / EEF delta control；
- 不同 gripper conventions。

建议换成：

```python
CanonicalRobotSpec = {
    "effectors": [...],
    "control_mode": ...,
    "control_dt": ...,
    "state_semantics": [...],
    "action_semantics": [...],
    "limits": ...,
}
```

```python
CanonicalActionToken = {
    "time_index": int,
    "effector_id": int,
    "semantic_type": str,
    "value": float,
    "valid": bool,
}
```

或计算友好的：

```python
actions: [H, E, D_sem]
effector_mask: [E]
action_semantic_mask: [E, D_sem]
dt: float
```

其中可统一：

- translation xyz
- rotation xyz / quaternion-normalized representation
- gripper
- joint delta
- base velocity
- arm ID

风险模型看到的是：

> **action semantics，而不是 benchmark-specific raw vector index。**

---

# 7. Observation contract 也必须升级

不要固定：

```text
front + wrist
```

改成：

```python
CanonicalObservation = {
    "views": [
        {"role": "external/front", "image": ...},
        {"role": "wrist/left", "image": ...},
        ...
    ],
    "view_mask": ...,
    "robot_state_tokens": ...,
    "task_text": ...,
    "elapsed_time": ...,
}
```

这样才能自然支持：

- LIBERO；
- CALVIN；
- SimplerEnv；
- RoboTwin；
- RoboCasa。

---

# 8. Shared Risk Core

## 8.1 输入

\[
x_t =
[
o_{t-L:t},
q_{t-L:t},
a^{src}_{t:t+H},
c,
d_{policy},
d_{robot},
d_{operator}
]
\]

其中：

- `d_policy`：无 outcome 的 policy behavior descriptor；
- `d_robot`：embodiment descriptor；
- `d_operator`：纠错操作 descriptor。

---

## 8.2 推荐架构

```text
multi-view RGB history
        |
        v
light visual encoder
        |
        +----------------------+
                               |
proprio history -> temporal ---+
                               |
source action semantic tokens -+--> temporal action-state fusion
                               |             |
task embedding ----------------+             |
                                             v
policy descriptor ----------------------> Shared Risk Core
robot descriptor ----------------------->      |
operator descriptor -------------------->      |
                                             |
          +----------------+----------------+----------------+
          |                |                |                |
          v                v                v                v
   source risk       urgency/window    operator value      OOD
                                             |
                                             v
                                    candidate verifier
```

---

## 8.3 模型规模

建议分两级：

### Teacher / research model

用于确认信息是否存在：

- 20–100M；
- 可以使用冻结 vision foundation features；
- 不作为部署结果。

### Online student

最终论文模型：

- 3–15M 参数；
- < 15–30 ms GPU inference 为目标；
- ONNX/TensorRT/TorchScript 可部署；
- 不需要 online world model。

不要在基础 signal 尚未成立前扩大模型。

---

# 9. 输出 heads

建议不是 5 个，而是 6 个。

## A. Source local-risk head

\[
p_{risk}^{src,H}
\]

---

## B. Urgency / recoverability-loss head

\[
p(T_{loss}\le t+H)
\]

或 survival hazard。

---

## C. Operator-prior advantage head

\[
\Delta^{prior}_c
\]

在不 query correction 的时候估计值得不值得调用。

---

## D. Candidate success / value head

实际 candidate action 出来以后：

\[
p_{success}(s,a_c,c)
\]

使用 Beta/Beta-binomial 或 ensemble interval。

---

## E. Cost head

预测：

- extra VLA calls；
- fallback steps；
- wall-clock latency；
- episode remaining steps。

---

## F. OOD / abstention head

输入落到未支持区域时：

- 降低 confident switching；
- 禁止 confident handback；
- 必要时 SAFE_ABORT。

---

# 10. 最关键的训练目标变化

## 10.1 不再把 trajectory final failure 复制到每一个 timestep

这是非常容易造成错误监督的做法。

---

## 10.2 建立 Decision-Point Counterfactual Dataset

对于自然 source rollout：

在 outcome-independent 的时间点 snapshot。

例如：

- 固定时间 stride；
- elapsed quantiles；
- contact transition；
- gripper transition；
- action curvature / jerk transition；
- progress phase transition。

注意：

> phase selection 不能使用 future outcome。

---

## 10.3 每个 snapshot 做 correction branches

例如：

```text
state s_t
  |
  +-- continue source
  +-- shorten + requery
  +-- resample #1
  +-- resample #2
  +-- corrective policy
  +-- optional replan
```

对每个 branch 记录：

- success count / trials；
- remaining steps；
- collision / drop；
- termination reason；
- correction calls；
- compute latency。

---

## 10.4 标签

### Source risk

不是：

```text
final episode failed
```

而是：

```text
continue-source becomes dominated
or
critical precursor happens within H
```

---

### Correction value

\[
\Delta_c =
p_{succ}(c)-p_{succ}(continue)
-\lambda C_c
\]

---

### Urgency

\[
U_t =
\max_c p_{succ}(s_t,c)
-
E[\max_c p_{succ}(s_{t+\Delta},c)]
\]

---

# 11. 如何处理 stochastic rollout

这是 R10 以后必须保留的原则。

每个 state/operator group：

\[
y_{i,k}\sim Bernoulli(p_i)
\]

不要硬标签。

建议：

- K=2：只做 smoke；
- K=3：初始估计；
- 如果 1/3 或 2/3，继续 sequential sampling；
- K 上限 8 或 12；
- posterior CI 到达精度条件后停止。

比如：

```text
if posterior interval width < eps:
    stop repeats
else:
    collect another rollout
```

这样比所有 state 固定 K=8 更省算力。

---

# 12. Zero-shot 还是 post-training？——建议采用三层主张

这是论文成败非常关键的一点。

## 不建议：

把主要贡献写成：

> “一个模型完全 zero-shot 适配所有新 VLA、新 benchmark，并使用统一阈值。”

你已有实验已经告诉我们：

- base rates 不同；
- action statistics 不同；
- calibration 不同；
- failure modes 不同。

这条主张太容易失败。

---

## 推荐三层结果

### Level 1 — RASE-ZS：真正 zero-shot

新 VLA 上：

- 不使用 success/failure label；
- 不更新参数；
- 只计算 outcome-free behavior descriptor；
- 使用在线无标签 score normalization。

目标：

> **哪怕提升很小，也要求方向一致。**

论文主张不是 SOTA：

> “The shared controller yields consistent positive zero-shot transfer.”

这正符合你的预期。

---

### Level 2 — RASE-UCal：无标签校准

允许：

- 8 / 16 / 32 条 unlabeled rollouts；
- 估计 action statistics；
- score mean/std；
- policy descriptor；
- OOD support。

不使用 outcome。

这是非常好的中间层。

---

### Level 3 — RASE-PT：轻量 supervised post-training

允许：

- 16 / 32 / 64 / 128 条 labeled trajectories；
- 只训练 policy adapter / calibrator / small head；
- **source VLA 保持冻结。**

目标：

> 比 zero-shot 提升明显更多。

这样论文可以同时回答：

- “能不能直接用？”——能，有小收益；
- “如果愿意做一点适配呢？”——收益明显放大。

---

## 为什么不要把 VLA 本身 fine-tune 当主结果

如果 RASE 发现 failures 以后再 fine-tune source VLA：

reviewer 会问：

> 到底提升来自 risk controller，还是来自 VLA post-training？

这会混淆核心贡献。

因此建议：

**主论文：冻结 VLA。**

额外 appendix 可以做：

> RASE-generated failure data 是否还能帮助 VLA LoRA post-training。

但不要把它作为必要组件。

---

# 13. Cross-VLA 泛化实验

不要只做 shared vs per-VLA。

完整 ladder：

1. per-VLA oracle upper-bound；
2. shared without policy condition；
3. shared + policy ID；
4. shared + behavior descriptor；
5. shared + behavior descriptor + embodiment descriptor；
6. leave-one-VLA-out zero-shot；
7. zero-shot + unlabeled normalization；
8. + 8 labeled；
9. + 16 labeled；
10. + 32 labeled；
11. + 64 labeled。

最重要的曲线：

```text
Success gain / failure reduction
          ^
          |
          |                    PT
          |                *
          |             *
          |          *
          |      *
          |  ZS *
          +------------------------>
             labeled trajectories
```

---

# 14. Cross-benchmark 设计

## 14.1 主 benchmark 建议

### Tier 1：LIBERO

必须保留。

原因：

- 当前基础设施最成熟；
- π0.5 / π0-FAST / OpenVLA 系列都有较成熟生态；
- 四个 suite；
- 可以直接继承现有实验。

---

### Tier 2：CALVIN

价值：

- long-horizon；
- language-conditioned；
- 和 LIBERO 的任务组织方式明显不同；
- 适合验证 progress / temporal risk。

---

### Tier 3：SimplerEnv

价值：

- 不同 simulator / embodiment；
- 真实机器人 policy 的 simulated evaluation；
- 非常适合证明 adapter 不只适配 LIBERO。

---

### Tier 4 Stretch：RoboTwin 2.0

价值最高但工程成本也最高：

- bimanual；
- action dimensionality 显著不同；
- domain randomization；
- 如果能成功，会非常强地证明 embodiment generality。

如果时间有限：

> **LIBERO + CALVIN + SimplerEnv 是 minimum multi-platform package，RoboTwin 是 high-value stretch。**

---

# 15. 不需要做 3×3 全笛卡尔积

如果 3 个 benchmark × 3 个 VLA 全跑，成本非常高。

更合理的是**connected evaluation graph**。

要求：

- 每个 benchmark 至少 2 个 VLA；
- 至少 1 个 VLA 横跨两个 benchmark；
- 至少 1 个 benchmark 有 3 个不同 architecture；
- 至少 2 个 source/corrective pair 通过 opportunity gate。

例如：

| Benchmark | VLA-A | VLA-B | VLA-C |
|---|---:|---:|---:|
| LIBERO | ✓ | ✓ | ✓ |
| CALVIN | ✓ | ✓ |  |
| SimplerEnv |  | ✓ | ✓ |

这样已经足够支持：

- cross-policy；
- cross-benchmark；
- held-out policy；
- held-out environment。

---

# 16. CVPR 应该增加一个 vision-centric evaluation

RASE 如果只做 robot control，会更像 CoRL / ICRA。

为了 CVPR，建议加入视觉分布偏移：

- camera viewpoint shift；
- lighting；
- occlusion；
- distractors；
- object appearance；
- instruction counterfactual；
- spatial perturbation。

可使用：

- LIBERO-X；
- LIBERO-Plus / PRO 类 robustness setting；
- 自己冻结的视觉 perturbation protocol。

重点不是要提升所有 OOD。

重点是：

> **risk calibration 是否在视觉 distribution shift 下仍然能提前发现 source 即将进入失败区域。**

这会让 CVPR 的“vision”贡献更清楚。

---

# 17. 实时性应该如何定义

不要只写“real-time”。

至少报告：

1. Risk-core latency / query；
2. Candidate-verifier latency；
3. source extra calls；
4. fallback calls；
5. wall-clock episode time；
6. GPU memory；
7. warning lead time；
8. correction trigger frequency。

---

# 18. 新的核心 evaluation metric

最终成功率仍然必须报，但不足以表达你的贡献。

建议加入：

## 18.1 Avoidable Failure Recovery Rate

\[
AFRR=
\frac{
\#\text{source failures rescued before irrecoverable point}
}{
\#\text{source failures with a valid rescue opportunity}
}
\]

---

## 18.2 Early Rescue Recall

真正存在恢复机会时：

> 有多少是在机会窗口关闭以前触发的。

---

## 18.3 Unnecessary Intervention Rate

原本 source 能成功：

> 有多少被不必要打断。

---

## 18.4 Lead Time

\[
T_{\mathrm{failure}}-T_{\mathrm{trigger}}
\]

越大不一定越好，所以同时配合 false-trigger。

---

## 18.5 Correction Utility

\[
Success
-\lambda_1 ExtraCalls
-\lambda_2 FallbackSteps
-\lambda_3 Latency
-\lambda_4 Harm
\]

---

## 18.6 Success–Compute Pareto

这是非常重要的一张主图。

x 轴：

- extra inference FLOPs / calls / latency

y 轴：

- task success

比较：

- source；
- always resample；
- fixed replan；
- always fallback；
- DVAC-like；
- RASE。

---

# 19. Selector 最终决策规则

推荐：

```text
1. Estimate source risk + urgency.
2. If low risk:
       execute supported safe prefix.

3. If risk elevated but urgency low:
       shorten chunk / requery.

4. If risk high or urgency high:
       run Operator Prior.

5. Query only correction operators whose
       LCB(expected advantage) > query cost threshold.

6. Candidate verifier scores generated candidates.

7. Execute candidate c only if:
       LCB(Delta_c) > tau_advantage
       AND support(c) is sufficient
       AND OOD < tau_ood.

8. Otherwise:
       safe hold / abort.
```

注意：

> `urgency high` 应允许比 ordinary risk 更早触发。

---

# 20. World Model 的位置

建议 paper v1 不把 world model 放主线。

只有在新的 Decision-Point Dataset 上：

```text
history/action baseline
```

已经有明显 signal 后，再尝试：

- action-conditioned multi-step residual；
- ensemble disagreement；
- future progress residual。

保留条件：

1. task-held-out 明显增益；
2. VLA-held-out 明显增益；
3. closed-loop success/Pareto 有增益；
4. teacher 可以被 distill；
5. online student 不需要跑大 world model。

否则放 appendix negative ablation。

---

# 21. 最重要的 baselines

至少需要：

### Detection baselines

- task/time prior；
- action norm / jerk；
- stagnation；
- policy uncertainty；
- SAFE-like failure detector；
- perturbation uncertainty；
- simple temporal classifier。

### Execution baselines

- source；
- always requery；
- fixed prefix；
- always resample M；
- always fallback；
- fixed early fallback；
- DVAC-like adaptive replanning（flow policy）；
- risk-only threshold + fixed correction；
- RASE full portfolio。

### Adaptation baselines

- per-VLA model；
- shared；
- shared + ID；
- shared + descriptor；
- zero-shot；
- few-shot calibration。

---

# 22. 论文 novelty 对比表应该长这样

| Method family | Proactive | Multi-VLA | Multi-benchmark | Multiple correction operators | Stochastic recoverability | Cost-aware | Zero-shot + few-shot |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAFE | ✓ detection | ✓ | limited | ✗ | ✗ | limited | task generalization |
| FPC-VLA | ✓ | some | ✓ | limited | ✗ | partial | ✓ |
| CycleVLA | ✓ | limited | limited | backtrack/retry | ✗ | limited | retry scaling |
| Pre-VLA | ✓ | limited | LIBERO | resample | ✗ | ✓ compute budget | limited |
| DVAC | ✓ replanning | flow-based | ✓ | replan timing | ✗ | ✓ | training-free |
| **RASE** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

最终投稿前必须重新核对这些文献最新版本，避免过度声称。

---

# 23. 推荐的论文 contribution

如果实验成立，建议写成四条。

### Contribution 1

> We formulate proactive VLA correction as stochastic, cost-aware counterfactual control rather than post-hoc failure detection.

### Contribution 2

> We introduce a model- and embodiment-agnostic runtime interface that represents heterogeneous observation histories and action chunks through semantic robot/action tokens.

### Contribution 3

> We propose a shared lightweight risk core that jointly predicts local source risk, recovery-window urgency, and correction-operator value, enabling selective resampling, replanning, and policy fallback before failures become irreversible.

### Contribution 4

> Across multiple VLA architectures and simulation benchmarks, RASE provides consistent zero-shot gains and substantially larger improvements with only lightweight controller calibration, while keeping the original VLA frozen.

---

# 24. 推荐论文题目

首选：

> **RASE: Proactive Risk-Aware Correction for Vision-Language-Action Policies Across Models and Simulators**

备选：

> **RASE: A Lightweight Runtime Risk Controller for Cross-VLA Proactive Correction**

更强调 stochastic：

> **RASE: Stochastic Recovery-Aware Runtime Control for Vision-Language-Action Policies**

---

# 25. 最小成立的 CVPR 证据包

如果想把它当“顶会方法论文”，我认为至少要有：

1. **3 个 VLA architecture / policy family**
2. **3 个 simulation benchmarks**，或 2 个不同 simulator + 1 个 strong OOD benchmark
3. 至少 **2 个 source/corrective pair** 有真实 model-free correction opportunity
4. 新 short-horizon / urgency target 在 task-held-out 下明显超过 time/policy prior
5. full controller 在至少 2 个 benchmark 上提高成功率
6. 第三个 benchmark 至少不降，并改善 compute/correction Pareto
7. held-out VLA zero-shot macro gain > 0
8. small labeled calibration 明显优于 zero-shot
9. 报告 warning lead-time
10. 报告 unnecessary intervention
11. 报告 stochastic calibration / CI
12. 报告 latency / memory / extra calls
13. visual OOD 条件下至少 risk ranking/calibration 仍有意义
14. 独立 validation cohort，而不是反复在同一开发集上调阈值

---

# 26. 最重要的 claim 分层

## Strong claim

只有全部通过才写：

> RASE is a general cross-VLA, cross-simulator proactive correction layer.

---

## Safe main claim

如果 zero-shot 不够强但 few-shot 很强：

> RASE is a shared runtime controller that transfers across VLA families and requires only lightweight calibration for new policies and environments.

这个仍然足够好，而且比硬吹 universal zero-shot 更可信。

---

## Fallback claim

如果 cross-benchmark control 太难：

> stochastic intervention benchmark + proactive risk model + multi-VLA correction on LIBERO + one external validation benchmark.

仍然比现在继续做 deterministic selector 更有机会。

---

# 27. 最终建议：不要把“zero-shot”与“post-training”二选一

最好的论文结构恰恰是同时有：

```text
Zero-shot           -> 证明 shared mechanism
Unlabeled calib     -> 证明可快速部署
Few-shot RASE-PT    -> 证明可获得明显增益
```

其中：

- zero-shot 要求“稳定正收益”，不要求最大；
- post-training 负责主要数值提升；
- VLA 主体始终冻结。

这会让你的 paper 同时拥有：

- generalization；
- practicality；
- strong numbers；
- clean attribution。

---

# 28. 一句话总结新版 idea

> **RASE 不等待机器人已经失败，而是在 VLA 闭环执行过程中持续估计“继续执行是否会使错误变得不可恢复”，并在 resample、requery、replan、corrective-policy takeover 等候选干预中进行风险—收益—成本联合选择；一个共享轻量风险核心通过 policy/embodiment descriptors 实现跨 VLA、跨仿真平台迁移，并以 zero-shot 正向迁移 + lightweight post-training 大幅提升作为最终泛化故事。**

---

# 29. 参考定位（截至 2026-08-13）

以下仅用于定位论文竞争关系，投稿前应重新检查最新版本：

- SAFE: Multitask Failure Detection for Vision-Language-Action Models, arXiv:2506.09937
- FPC-VLA: A Vision-Language-Action Framework with a Supervisor for Failure Prediction and Correction, arXiv:2509.04018
- CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding, arXiv:2601.02295
- Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts, arXiv:2605.22446
- Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies, arXiv:2606.03847
- vla-eval: A Unified Evaluation Harness for Vision-Language-Action Models, arXiv:2603.13966
- StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing, arXiv:2604.05014
- RoboTwin 2.0, arXiv:2506.18088
- CALVIN, arXiv:2112.03227
