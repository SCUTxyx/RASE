# RASE：面向多 VLA 的轻量随机风险控制与策略切换

**版本日期：** 2026-08-13  
**文档状态：** 新 canonical idea；吸收 PRE-C0-R3 至 R10-B 的全部正负证据  
**工作名称：** RASE  
**英文名称：** **Risk-Aware Switching and Execution for Multi-VLA Policies**  
**目标论文形态：** Stochastic Risk Estimation + Source-Conditional Policy Switching + Optional Dynamics Evidence

---

## 0. 一句话定义

RASE 是部署在冻结 VLA 与机器人执行器之间的轻量实时控制层。它读取当前多视角观测、
机器人状态、任务、source VLA 提议的 action chunk 和短时历史，估计：

1. source 继续执行的失败风险；
2. 候选 fallback 在当前状态上的成功概率分布；
3. 延迟切换导致 fallback 可恢复性下降的概率；
4. 接管或继续接管的预计成本；
5. 当前预测是否超出模型支持范围。

控制器在安全约束下选择：继续 source、切入 fallback、保持 fallback、在证据充分时交还
source，或安全中止。模型不生成新的机器人动作；动作仍由冻结 source/fallback VLA 提供。

RASE 的“通用”含义不是一个零样本阈值适配所有 VLA，而是：

> 一个共享的轻量风险核心，通过标准化的观测/action contract、无标签行为描述符和很小的
> VLA/fallback 校准模块，迁移到多个 source VLA 和多个有机会空间的 corrective policy pair。

世界模型不是默认在线组件，也不是失败方法的救火工具。它只以 action-conditioned
multi-step residual、预测分歧或其蒸馏结果作为附加动态证据；只有跨 VLA 的 state-level
Pareto 增益通过预注册门槛，才进入最终模型。

---

## 1. 动机：为什么需要 RASE

现代 VLA 可以直接从视觉和任务语言生成动作，但部署时至少存在四个未解决问题：

- VLA 通常不会给出经过校准的“当前动作会不会失败”的概率；
- 不同 VLA 的动作尺度、chunk 长度、控制频率和失败机制不同；
- 一个更强的 fallback 并非在任意进入时刻都能救回，延迟切换可能永久错过恢复窗口；
- fallback outcome 本身可能具有闭环随机性，相同可见状态和首动作不保证相同最终结果。

传统 uncertainty trigger 往往只回答“模型是否不确定”，并不回答：

- source 是否真的会失败；
- fallback 是否比 source 更好；
- 现在切换是否仍来得及；
- 额外 teacher/fallback 成本是否值得；
- 当前风险是可观测的 epistemic uncertainty，还是无法通过更多表示消除的 aleatoric risk。

RASE 因此不是普通 OOD detector，也不是新的动作生成策略，而是一个受安全约束的、
策略条件化的随机最优停止与切换层。

---

## 2. 证据驱动的 idea 演化

新版 idea 不回避已有负结果。每一项方法设计都对应一个已验证的问题。

### 2.1 Success-only selector 已被否定

PRE-C0-R3 的 72 个状态中：

- Student：18/72 = 25.0%；
- persistent OFT：66/72 = 91.7%；
- 全 operator success oracle：66/72 = 91.7%；
- oracle minus best fixed：0pp。

如果目标只有成功率，永远调用 persistent OFT 已经等于 oracle。训练
`P(success | operator)` 只会学会永远选择 OFT，不构成研究方法。

因此 RASE 的目标必须是**安全—成功—成本 Pareto 控制**，而不是成功率最大化的 operator
分类。

### 2.2 短动作纠正不是主要恢复机制

PRE-A1/A2 表明，少量 OFT action prefix 无法稳定修复 source 的系统性能力错误；交回
source 后错误行为会再次出现。一个“动作修正 head”如果要真正解决问题，实际上需要学习
一个完整、更强的恢复策略，这是另一个研究方向。

因此当前 RASE：

- 不训练生成式 action replanner；
- 不声称撤销已经执行的物理动作；
- 只在动作执行前拒绝 chunk 或切换到现成的完整策略；
- simulator restore 仅用于构造离线反事实标签。

### 2.3 任意时刻 dwell selector 已被否定

R6-C/R6-C.1 的五 seed OOF 均为 0/5。Pi0Fast+OFT 的主要机制是：fallback 的可救率随
进入时间快速下降，t>=32 后再切换经常已经太晚。two-boundary dwell 反而把干预推入失效区。

因此 selector 必须显式建模**机会窗口**，而不是在整条轨迹上使用同一阈值与固定 dwell。

### 2.4 单帧 source-risk 证据不足

R7-A 在 191 个可复现独立状态、48 tasks、五 seed task-held-out OOF 上得到：

- 平均 AUROC 0.631；
- seed 范围 0.564--0.675；
- 0/5 通过 AUROC、bootstrap、校准和分 suite gate。

这说明当前单一 t=0 观测不足以稳定预测 source 最终失败。项目不能依靠更大 MLP 或事后
调阈值把近随机风险排序包装成通用风险模型。

### 2.5 稀疏 hazard target 不可直接学习

R9-B 的 480 个时间边界只有 12 个 recoverability-loss positive；Object、Spatial 和 camera
扰动为零，两个外层 fold 单类。虽然 temporal-state probe 在三个有效 fold 上达到 0.788，
all-causal 0.829 仍未超过 policy+horizon prior 0.842。

这否定的是**当前自然 cohort 上稀疏的无条件 hazard target**，而不是所有时序风险。

### 2.6 R10-B 证明 recoverability 是随机变量，而非确定标签

R10-B 从旧 K=2 标签冻结 33 cases/33 controls，并重新采集 66 groups x K3：

- 198/198 trajectories 完成；
- t8 deployable features 66/66 replica parity；
- K3 内稳定 57/66，不稳定 9/66；
- 只有 32/66 保持旧 K2 标签；
- 33 个旧 cases 中仅 8 个仍为稳定 case，19 个变 control，6 个不稳定。

因此“t8 一定可救、t16 一定不可救”不是可靠的 state property。用小 K 选择极端二元样本
会产生显著回归均值和 winner's curse。

保留全部 group 的 K3 概率诊断得到 42/198 hazard events，但：

- temporal-state AUROC 0.609，task-bootstrap lower 0.429；
- temporal+action AUROC 0.591；
- all-causal AUROC 0.523；
- Pi0.5/Pi0Fast all-causal AUROC 0.512/0.563。

所以简单把 BCE 换成 Beta-binomial 仍不够。必须先区分标签随机性来源和可观测性。

### 2.7 世界模型静态 latent 替换没有带来收益

V-JEPA 2-AC pooled latent 替换曾从 baseline AUC 0.865 降至 0.573；正确配对的单步 latent
delta 也未提供 state-level Pareto 增益。这说明通用视频表征不等于边界风险表征。

仍允许研究的世界模型证据只有：

- action-conditioned multi-step prediction residual；
- dynamics ensemble disagreement；
- source/fallback action 下的未来可行性差异；
- 上述证据蒸馏到轻量模型后的增量。

---

## 3. 新的核心科学假设

### H1：风险是 policy-conditioned、history-conditioned 的

真正要估计的是：

\[
P(Y^{src}=0\mid o_{t-L:t}, q_{t-L:t}, a^{src}_{t:t+H}, c, d_{src}),
\]

其中：

- \(o\)：多视角 RGB；
- \(q\)：proprio/contact/progress 历史；
- \(a^{src}\)：source 提议的标准化 action chunk；
- \(c\)：任务文本或语义表示；
- \(d_{src}\)：source VLA 的可部署行为描述符。

模型不是预测“这个任务难不难”，而是预测“这个 VLA 在当前时序状态下提出的这组动作有
多危险”。

### H2：fallback 应预测成功分布，而不是硬标签

对于 fallback \(f\)：

\[
Y^{f}_{t,k}\sim Bernoulli(p_f(s_t, a^f, d_f)),
\]

并允许状态间过度离散：

\[
p_f(s_t)\sim Beta(\alpha_t,\beta_t).
\]

模型输出 posterior mean、concentration、LCB/UCB，而不是只输出一次 rollout 的 0/1。

### H3：selector 需要预测干预优势，而不是只预测 source risk

进入 fallback 的价值为：

\[
\Delta_{enter}(s_t)=
P(Y=1\mid ENTER_f,s_t)-P(Y=1\mid CONTINUE_{src},s_t)
-\lambda_c C_f(s_t)-\lambda_l L_f.
\]

只有当 source risk 高、fallback 成功概率有支持、且 \(LCB(\Delta_{enter})>0\) 时才接管。

### H4：机会窗口是随机生存问题

定义 fallback 可恢复性丢失时间：

\[
T_{loss}=\inf\{t:p_f(s_t)<\tau_f\}.
\]

模型估计离散 hazard：

\[
h_t=P(T_{loss}\in(t,t+\Delta]\mid T_{loss}>t,\mathcal H_t).
\]

这比“t8 case / t16 control”的确定分类更符合 R10 证据。它允许 censoring、重复 rollout 和
不同状态的随机成功概率。

### H5：通用性来自共享机制与小型校准，不来自统一零样本阈值

不同 VLA 具有不同的：

- base failure rate；
- action chunk 统计；
- 闭环反应速度；
- 任务优势与盲点；
- 风险分数校准。

因此主张应是：共享 backbone 学习跨策略的视觉—状态—动作风险机制，新 VLA 通过无标签行为
descriptor 和少量 calibration trajectories 完成适配。完全 zero-shot 只作为挑战指标。

---

## 4. 系统控制状态与动作语义

RASE 的控制器有五个显式状态：

```text
SOURCE_ACTIVE
    ├─ low supported risk ───────────────► execute source chunk
    ├─ high risk + fallback advantage ──► FALLBACK_ACTIVE
    └─ high risk + no supported rescue ─► SAFE_ABORT / ask human

FALLBACK_ACTIVE
    ├─ recovery not established ─────────► continue fallback
    ├─ handback LCB passes ──────────────► SOURCE_ACTIVE
    └─ fallback irrecoverable/unsafe ────► SAFE_ABORT
```

严格术语：

- `REJECT_SOURCE_CHUNK`：拒绝尚未执行的 source 动作；
- `ENTER_FALLBACK`：切换到完整 corrective policy；
- `CONTINUE_FALLBACK`：保持接管；
- `HAND_BACK_TO_SOURCE`：证据充分后交回；
- `SAFE_ABORT`：没有受支持的自动行为时停止。

“动作回退”不能表示物理时间倒流或撤销动作。真实部署中没有 simulator rollback。

当前证据下，`HAND_BACK_TO_SOURCE` 是 gated extension；若没有独立机会支持，fallback 默认
persistent 到终止。

---

## 5. 通用输入接口

### 5.1 Canonical observation

```python
CanonicalObservation = {
    "images": {"front": uint8[3,H,W], "wrist": uint8[3,H,W]},
    "proprio": float[Dq],
    "contact": optional_float[Dc],
    "task_text": str,
    "elapsed_steps": int,
    "controller_state": str,
}
```

保留最近 \(L\) 步历史，而不是只使用单帧。部署模型不得读取：

- simulator object state；
- future frame/outcome；
- task ID 或 suite ID 的直接编码；
- fallback 最终 cost 或成功标签；
- held-out VLA 的成功率统计。

### 5.2 Canonical action chunk

```python
CanonicalActionChunk = {
    "actions": float[H,7],
    "valid_mask": bool[H],
    "dt": float,
    "frame": "eef_delta",
    "rotation": "axis_angle",
    "gripper": "continuous_normalized",
    "policy_family": str,
}
```

每个 VLA adapter 负责坐标、单位、频率、gripper convention、chunk padding 和 mask。风险模型
必须看到完整 chunk 或其因果编码；不能把 Pi0Fast 的 10-step proposal 退化成第一个 7-DoF
动作。

### 5.3 Outcome-free behavior descriptor

为适配未见 VLA，从独立、无标签 calibration rollouts 计算：

- action mean/std/quantiles；
- chunk 内变化率和 jerk；
- gripper switching frequency；
- observation-conditioned action sensitivity；
- action latency/control frequency；
- proposal ensemble dispersion（若 VLA 原生支持）；
- action encoder 的均值和 covariance sketch。

descriptor 不得包含成功、失败、fallback rescue 或 teacher cost。

---

## 6. 轻量风险模型架构

### 6.1 总体结构

```text
RGB history ─► Tiny multi-view encoder ──────────────┐
Proprio/contact history ─► temporal state encoder ──┤
Task text ─► frozen/hash semantic encoder ──────────┤
Source action chunk ─► canonical action encoder ────┤
Source behavior descriptor / seen-ID ───────────────┤
                                                      ├─► Shared Risk Core
Fallback descriptor + proposed fallback chunk ──────┤
Optional WM residual/disagreement ──────────────────┘
                                                      │
          ┌───────────────────────────────────────────┼───────────────┐
          ▼                                           ▼               ▼
 Source-risk distribution                  Fallback success     Window hazard
          │                                distribution         / survival
          ├───────────────────────────────────────────┬───────────────┤
          ▼                                           ▼
 Intervention advantage                       Cost quantiles / OOD
```

这里的 fallback action 输入必须按控制阶段解释：

- **进入前 canonical 模式**：不调用昂贵 fallback；只使用 source action、当前状态和冻结的
  fallback descriptor 预测进入价值。这样才能真正减少 fallback action-selection calls；
- **进入前 action-query baseline**：允许查询一次 fallback proposal，但这次调用必须计入
  teacher latency、action-selection calls 和总成本，不能当作免费特征；
- **进入后模式**：fallback 已经接管，其当前/历史 action 是自然可用的因果输入，可以用于
  recoverability、handback 和 remaining-cost heads；
- **离线 teacher 模式**：可以同时使用 source/fallback action 做反事实 dynamics，但蒸馏后的
  在线学生若仍需要预调用 fallback，必须按照上一条计费。

因此实际实现应使用两个共享 backbone 上的小型决策接口：`PreEntryRiskCore` 与
`InFallbackRiskCore`，而不是把所有字段强行填入同一个固定向量。

### 6.2 推荐规模

- shared visual/state/action core：1--5M 参数；
- VLA descriptor adapter：每个 VLA 10K--100K；
- fallback calibration adapter：10K--100K；
- 3--5 个 task-bootstrap ensemble members；
- 最终部署包建议不超过 10--25 MB；
- 支持 TorchScript/ONNX，教师和 simulator 依赖必须完全移除。

参数目标是方法约束，不是贡献本身。轻量模型只有在校准、任务泛化和控制 Pareto 通过后才有
意义。

### 6.3 输出 heads

#### A. Source-risk head

输出 \(p_{fail}^{src}\) 与 epistemic uncertainty。训练使用 task-clustered outcome 标签，并
报告 per-VLA/per-suite calibration。

#### B. Fallback success head

输入当前状态、fallback descriptor 和候选动作，输出 Beta/Beta-binomial 参数：

\[
(\mu_f,\kappa_f),\quad
\alpha_f=\mu_f\kappa_f,\quad
\beta_f=(1-\mu_f)\kappa_f.
\]

#### C. Recoverability-loss survival head

使用 discrete-time survival likelihood，支持：

- 某边界仍可恢复；
- 首次观察到不可恢复；
- episode 提前成功/结束造成的 censoring；
- K-repeat 的经验事件计数。

#### D. Intervention-advantage head

预测 `ENTER_FALLBACK` 相对 `CONTINUE_SOURCE` 的成功差和成本差。它是 selector 的直接价值
目标，但必须与 source/fallback 独立 heads 共同训练，避免纯 value head 隐藏错误来源。

#### E. Cost quantile head

输出 fallback 剩余动作数/延迟的 q10/q50/q90，而不是只预测均值。

#### F. OOD/abstention head

融合 ensemble disagreement、descriptor distance、action support 和可选 world-model residual。
高 OOD 时不得自信 handback；控制器应选择保守 fallback 或 SAFE_ABORT。

---

## 7. 随机标签与数据协议

### 7.1 三种不同的随机性

必须分开记录：

1. **Source-policy randomness**：source 采样导致的动作/outcome 差异；
2. **Fallback-policy randomness**：相同 snapshot 上 fallback 后续 chunk 分叉；
3. **Environment/termination randomness**：完整动作序列相同但 outcome 翻转。

这三者不能混成一个“label noise”。它们分别决定：增加模型输入、增加重复次数，还是修复
仿真/终止协议。

### 7.2 重复采集单位

数据的独立统计单位是 task/episode/state-policy group，不是 boundary row 或 replica。

每个 counterfactual group 保存：

```python
snapshot_hash
source_policy_hash
fallback_policy_hash
rollout_seed
replica_id
boundary_t
causal_feature_hash
source_action_trace_hash
fallback_action_trace_hash
success
termination_reason
executed_steps
```

同一 snapshot 的 repeats 是概率测量，不是更多独立样本。split 和 bootstrap 以 task 为单位。

### 7.3 禁止再次使用极端小 K 选择作为正式标签

R10 已证明从 K2 的 `2/2 -> 0/2` 选择 cases 会严重回归均值。后续必须：

- 用 outcome-independent metadata 冻结 cohort；
- 或将 label-balanced case-control 明确限制为 discovery；
- formal confirmation 使用新独立 cohort；
- soft-label 训练记录 successes/trials，而不是先硬化成 0/1；
- 报告 posterior uncertainty 与 overdispersion；
- 不删除不稳定状态来制造确定标签。

### 7.4 推荐重复数

- K=3：collector/repro smoke；
- K=5--8：估计 label entropy 与 overdispersion；
- K=8--16：训练概率 head 的最低候选；
- K 或 sequential stopping 由预注册精度目标决定；
- 最终闭环 evaluation 的 episode 数不由 counterfactual K 替代。

---

## 8. Selector：约束风险下的策略切换

### 8.1 进入 fallback

```text
ENTER_FALLBACK only if
    UCB(source_failure_risk) >= tau_source
and LCB(fallback_success - source_success) >= tau_advantage
and LCB(fallback_success) >= tau_fallback
and OOD <= tau_ood
and current_time <= latest_supported_entry_time
```

canonical pre-entry selector 的 fallback-success/advantage 来自状态与 fallback descriptor，
不要求先运行 fallback。若某个实验版本查询 fallback action，它属于更昂贵的显式 baseline，
所有查询均计入 intervention cost。

如果 source risk 高但 fallback 没有证据可救，动作不是盲目切换，而是 SAFE_ABORT/人工接管。

### 8.2 保持 fallback

```text
CONTINUE_FALLBACK if
    handback evidence insufficient
or recoverability/cost uncertainty high
or source is outside calibrated support
```

### 8.3 交还 source

```text
HAND_BACK only if
    LCB(success_if_handback_now) >= tau_handback
and UCB(source_failure_after_handback) <= tau_risk
and expected saved fallback cost >= tau_cost
and temporal consistency/hysteresis passes
```

在当前证据下，handback 不属于已验证主模型。只有独立 model-free opportunity gate 和概率
support 通过才解锁。

### 8.4 控制目标

RASE 优化的不是单一 accuracy，而是：

\[
\max_{\pi_{switch}}
P(success)-\lambda_1 E[fallback\ steps]-\lambda_2 P(harm)-\lambda_3 latency
\]

满足：

- success gap vs persistent fallback >= -5pp；
- absolute paired harm <= 5%；
- false continue/missed rescue <= 5%；
- fallback-step savings >= 20%；
- harm 不集中于某一 suite/VLA；
- 至少 4/5 seeds 通过。

---

## 9. 多 VLA 与多 fallback 泛化

### 9.1 两层“通用”主张

必须分开：

1. **通用风险模型**：可使用 Pi0Fast、Pi0.5、SmolVLA，以及未来接入的 source policy；
2. **通用 selector**：只能使用通过 model-free opportunity gate 的 source/fallback pair。

一个 VLA 可以提供很好的风险迁移数据，但不一定适合作为 selector pair。例如 SmolVLA 可作为
低能力 source 的风险 cohort，却不能在现有证据下与 OFT 构成有效 selector。

### 9.2 固定评估梯度

对每个 qualified VLA 依次评估：

1. per-VLA model；
2. pooled shared，无 policy condition；
3. shared + seen-VLA ID；
4. shared + outcome-free behavior descriptor；
5. shared + descriptor + tiny calibration；
6. leave-one-VLA-out zero-shot；
7. 新 VLA 0/8/16/32 unlabeled adaptation；
8. 单独报告 0/8/16/32 labeled calibration。

主结论要求 shared+descriptor+small calibration 接近 per-VLA 上限；zero-shot 不作为唯一硬门。

### 9.3 OpenVLA 的角色

当前服务器上的 OpenVLA checkpoints 是 suite-specific OpenVLA-OFT fallback，经 oracle server
运行；它们不是已经验证的第四个 direct source。把同一 OpenVLA-OFT 同时当 source 和
fallback 会产生退化 selector。

若未来将 OpenVLA 作为 architecture-held-out source，需要：

- 实现 oracle-backed source continuation；
- 冻结 checkpoint/config hashes；
- source-only parity 与 label-support audit；
- 使用不同的 corrective arm；
- 重新通过 model-free opportunity gate。

---

## 10. 世界模型的正确位置

### 10.1 不存在可直接“切下来”的风险模块

V-JEPA 2-AC、Ctrl-World 等模型提供的是未来 latent/video dynamics，不包含现成的
`robot failure risk head`。所谓提取风险能力，只能是：

1. 离线运行冻结 world-model teacher；
2. 计算与真实 outcome 对齐的动态证据；
3. 训练 teacher-side probe 验证证据含信息；
4. 将通过验证的低维 risk token/residual 蒸馏到轻量学生；
5. 部署时完全移除大模型。

### 10.2 允许的动态证据

```text
history + candidate action
        │
        ▼
action-conditioned latent predictor ensemble
        │
        ├─ multi-step residual
        ├─ ensemble disagreement
        ├─ source-vs-fallback predicted feasibility difference
        ├─ contact/progress prediction error
        └─ support/OOD distance
```

禁止用 pooled latent 替换已有 causal features 后直接声称 world-model 增益。

### 10.3 何时允许启动 world-model 消融

满足以下前提之一：

- 根因审计证明后续 fallback chunk 因可观测闭环状态分叉；
- 无 world-model 的 history/action baseline 已稳定通过基础风险 gate；
- 独立 low-capacity probe 证明 residual/disagreement 超过 persistence/task/policy prior。

### 10.4 保留门槛

world-model evidence 只有同时满足才保留：

- task-held-out 和 VLA-held-out 风险 ranking 增益；
- calibration 改善；
- 至少两个 VLA 的 paired success 不降；
- selector savings/Pareto 改善；
- 在线学生满足延迟和尺寸要求；
- 增益不依赖 teacher 在部署时运行。

否则世界模型作为负消融或离线诊断工具，不进入标题和主方法。

---

## 11. 训练损失

在标签来源与可观测性通过后，推荐联合目标：

\[
\mathcal L=
\lambda_s\mathcal L_{source}
+\lambda_f\mathcal L_{fallback}^{BB}
+\lambda_h\mathcal L_{survival}
+\lambda_a\mathcal L_{advantage}
+\lambda_c\mathcal L_{cost}
+\lambda_u\mathcal L_{uncertainty}
+\lambda_w\mathcal L_{WM-distill}.
\]

- `source`：source final/short-horizon failure 的 calibrated BCE 或 survival loss；
- `fallback-BB`：successes/trials 的 Beta-binomial NLL；
- `survival`：recoverability-loss hazard，含 censoring；
- `advantage`：paired source/fallback success difference；
- `cost`：teacher steps/latency quantile loss；
- `uncertainty`：ensemble、descriptor distance、selective risk；
- `WM-distill`：仅在预注册动态证据通过后启用。

真实 counterfactual outcome 始终高于 teacher soft target。teacher 与真实标签冲突时，不允许
world-model 覆盖真实监督。

---

## 12. 评估协议

### 12.1 数据拆分

- task-held-out outer folds；
- 同 task 的 state、boundary、replica 全部在同一 fold；
- calibration 只使用 outer-train/calibration tasks；
- enrichment 只用于训练，不进入自然分布 OOF；
- case-control discovery 不能报告 prevalence/ECE/selector success；
- 新 VLA descriptor cohort 与 evaluation tasks 分离；
- test 在独立 validation 通过前保持密封。

### 12.2 风险模型指标

- per-VLA/per-suite AUROC、AP、Brier、ECE；
- selective risk/coverage；
- task-cluster bootstrap CI；
- probability count NLL 与 overdispersion；
- leave-one-VLA-out 和 adaptation curves；
- 与 task-only、policy-only、horizon-only prior 比较。

### 12.3 控制指标

- paired success gap vs persistent fallback；
- absolute paired harm；
- missed rescue/false continue；
- fallback action-selection calls 与 executed steps；
- latency、内存、intervention burden；
- suite/VLA worst-group performance；
- privileged oracle 与 best fixed policy。

### 12.4 必要 baselines

- pure source VLA；
- persistent fallback；
- best fixed early entry；
- task/policy/horizon prior；
- source-risk-only + fixed fallback；
- deterministic stagnation/progress trigger；
- per-VLA risk model；
- pooled shared；
- shared+descriptor/calibration；
- world-model-free history/action model；
- world-model residual/disagreement augmentation；
- privileged cost-aware oracle。

---

## 13. 当前执行阶段与下一步

截至 2026-08-13，R10-B deterministic case-control 主实验正式 FAIL。风险模型、selector、
world model、validation 和 test 保持锁定。

当前运行的是**根因诊断**，不是新模型训练：

- 9 个 outcome-unstable groups；
- 9 个按 suite/source 匹配并哈希冻结的 stable controls；
- 每组 K3，共 54 trajectories；
- t8/t16 记录完整 fallback action-trace SHA256 与 shape；
- 不能用于训练、阈值选择或论文正结果。

此前已确认 t8/t16 fallback 首动作在 66/66 groups 三副本间 bitwise identical，最大差异为
0.0。根因实验完成后：

1. **完整 fallback trace 相同、outcome 翻转**  
   运行 fixed-action replay，审计 simulator/termination randomness；后续方法需建模不可约风险，
   而不是继续增加视觉模型。
2. **首动作相同、后续 trace 分叉**  
   审计每次 chunk-query 的 observation/proprio/action hash；只有证明分叉来自可观测 dynamics，
   才解锁 multi-step residual/disagreement probe。
3. **未复现翻转**  
   保持 R10-B FAIL；用序贯概率设计估计所需 K，并在新的 outcome-independent cohort 上确认。

任何分支都不会直接解锁 selector。新的 probabilistic information gate 必须先通过。

---

## 14. 顶会论文的最小成立条件

RASE 只有满足以下证据链，才能成为方法论文，而不是数据分析报告：

1. 至少两个 VLA 在自身任务内具有真实、task-held-out 的风险可观测信号；
2. shared+descriptor+small calibration 接近 per-VLA 上限；
3. 至少两个 source/fallback pair 通过 model-free opportunity gate；
4. 五 seed OOF 中至少 4/5 同时满足安全、成功和成本门；
5. 新独立 validation cohort 通过；
6. 100+ paired closed-loop episodes、四 suites、两 seeds 和第二 policy pair；
7. 报告 task-clustered intervals、worst suite、延迟、内存和 intervention burden；
8. world-model 只有在两 VLA 的 state-level Pareto 增益成立时才进入主贡献。

如果风险可观测性或双 policy-pair 控制门失败，应将贡献诚实转为：

- stochastic counterfactual risk benchmark；
- reproducibility/label-noise protocol；
- model-free opportunity atlas；
- 或独立的 recovery-policy/DAgger 项目。

---

## 15. 明确不做什么

- 不训练一个新生成式 VLA 来替代风险控制问题；
- 不把 simulator restore 描述成真实机器人 rollback；
- 不用单次 rollout 作为确定 recoverability 真值；
- 不删除不稳定状态以改善指标；
- 不用 case-control enrichment 报告自然 prevalence 或 calibration；
- 不把 pooled policy base rate 当成跨 VLA 泛化；
- 不把 zero-shot 统一阈值作为主要结论；
- 不用更大 MLP 或世界模型挽救接近随机的基础信息门；
- 不因 row-level AUC 提升就声称 controller 更安全；
- 不在 validation 通过前解封 test。

---

## 16. 最终方法主张模板

若完整证据链通过，论文主张应写成：

> RASE is a lightweight stochastic risk-control layer for frozen vision-language-action
> policies. It jointly models source-policy failure, fallback success distributions,
> recoverability-window hazards, and intervention cost from causal observation and action
> histories. A shared risk core, paired with outcome-free behavior descriptors and small
> policy-specific calibration modules, transfers across multiple VLAs. RASE improves the
> success-cost Pareto frontier of validated source/fallback pairs while abstaining under
> unsupported or irreducibly stochastic conditions. Optional action-conditioned world-model
> evidence is retained only when it yields cross-policy state-level gains and is distilled
> out of the online controller.

中文表述：

> RASE 是一个服务于冻结 VLA 的轻量随机风险控制层。它从因果观测与动作历史联合预测
> source 失败、fallback 成功分布、恢复窗口丢失风险和干预成本；共享风险核心通过无标签
> 行为描述符与小型策略校准模块迁移到多个 VLA。对于通过 model-free opportunity gate 的
> source/fallback pair，RASE 在不显著损害成功率的前提下减少 fallback 接管成本，并在风险
> 不可观测或具有不可约随机性时保守拒识。世界模型只在跨策略 state-level 增益成立时作为
> 动态证据，并从最终在线控制器中蒸馏移除。

---

## 17. 最核心的变化

旧问题是：

> 能否训练一个分类器判断何时切换/交还？

新问题是：

> 在多个冻结 VLA、随机 fallback outcome 和部分可观测状态下，能否用轻量共享模型估计
> source/fallback 的概率风险与恢复窗口，并通过保守校准的小型 selector 改善成功—成本
> Pareto，同时识别什么时候应该拒识而不是自信决策？

这保留了原始 idea 的核心——实时风险预测、多 VLA 泛化、策略切换/纠正和可选世界模型——
但把训练目标、随机性建模、泛化边界与验证顺序改成了与现有实验证据一致的形式。
