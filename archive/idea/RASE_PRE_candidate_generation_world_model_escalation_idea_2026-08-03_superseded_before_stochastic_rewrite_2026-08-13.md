# RASE：面向多 VLA 的实时概率风险预测与纠正接管

**版本日期：** 2026-08-09  
**状态：** R6 修订主线；safe handback 已由 B24 gate 正式关闭  
**工作名称：** RASE  
**英文名称：** **Risk-Aware Switching and Execution for Multi-VLA Policies**  
**目标论文形态：** Multi-VLA Risk Control + Corrective Takeover；Safe Optimal Stopping 仅为未来 gated extension

---

## 0. 一句话定义

RASE 是一个部署在冻结 VLA 与机器人执行器之间的轻量实时安全控制层：它预测执行
当前 VLA action chunk 的失败风险，在风险过高时拒绝该动作并切换到纠正策略。当前
默认控制合约是接管后保持 persistent correction 到 episode 终止；safe handback 只有在
独立、概率化 opportunity gate 重新通过后才允许恢复为主模型组件。

世界模型不是在线最终裁判。它是离线动力学教师和不确定性证据源；部署时运行的是可在
多个 VLA 之间共享的轻量概率风险模型。

```text
Frozen source VLA proposal
        │
        ▼
Canonical action adapter + lightweight risk model
        │
        ├── low risk ───────────────► execute source action chunk
        │
        └── high risk
                │
                ▼
        reject chunk / enter corrective policy (OFT)
                │
                ▼
        persistent corrective policy (OFT)
                │
                ▼
        continue correction to episode termination

Safe handback: gated future extension only
```

---

## 1. 哪些原始想法保留，哪些必须修改

### 1.1 保留的核心

以下方向仍然成立，是项目的主贡献候选：

1. 对冻结 VLA 做实时、动作条件的风险预测；
2. 在动作执行前拒绝高风险 action chunk；
3. 使用更强或更稳健的策略执行局部纠正/恢复；
4. 通过统一 action/state contract 适配多个 VLA；
5. 世界模型仅作为预注册的离线辅助证据，且必须证明 state-level Pareto 增益。

“恢复后安全交还”不再是当前主贡献。B24 只有 1/24 个真正由 OFT 创造的 all-K-safe
恢复状态，无法支持该 head 的训练、task-held-out 评估或顶会 claim。

### 1.2 被正式否定的版本

以下主张不得继续使用：

- success-only operator selector 可以超过 persistent OFT；
- 单次 `success_if_handback_now` 是确定真值；
- row-level AUC 高即可证明 controller 安全；
- V-JEPA 单步 latent delta 必然提高风险判别；
- simulator restore 可以作为真实机器人 rollback 操作；
- 当前 0.6M checkpoint 已通过部署验证；
- 原 M4/M5 和 `p=0.0` 结果可作为论文证据。

### 1.3 “动作回退”的严格含义

论文和代码统一使用：

- `REJECT_ACTION_CHUNK`：当前动作尚未执行时拒绝；
- `ENTER_CORRECTIVE_POLICY`：切换到 OFT 或其他恢复策略；
- `CONTINUE_CORRECTIVE_POLICY`：恢复尚未达到安全交还条件；
- `HAND_BACK_TO_SOURCE_VLA`：把控制权交还原 VLA；
- `SAFE_ABORT`：没有受支持的自动恢复行为时停止或请求人工。

不得把已经执行的物理动作“撤销”描述为部署能力。Simulator restore 只用于生成离线
反事实监督和协议审计。

---

## 2. 最新证据与路线转变

### 2.1 Success-only selector 没有机会空间

PRE-C0-R3 的 72 个 train states：

- Student：18/72 = 25.0%；
- persistent OFT：66/72 = 91.7%；
- 全 operator success oracle：66/72 = 91.7%；
- oracle minus best fixed：0pp。

因此，只最大化成功率的 selector 最优解就是始终选择 persistent OFT。RASE 的科学问题
必须从“选择哪个成功 operator”改成：

> 在保持 persistent-OFT 成功率的同时，何时可以更早、安全地停止 OFT 接管？

### 2.2 有限恢复时长仍有成本机会

- h96：48/72 = 66.7%；
- h128：51/72 = 70.8%；
- persistent：66/72 = 91.7%。

有限 handback 无法单独匹配 persistent，但存在 state-dependent 的最短恢复时长。这给
cost-aware optimal stopping 留下真实机会空间。

### 2.3 修正后的 0.6M 模型尚不安全

泄漏安全、task-held-out、state-level earliest-handback 的 5-seed 结果：

| 指标 | 均值 | 最小 | 最大 |
|---|---:|---:|---:|
| row AUC | 0.9270 | 0.9221 | 0.9319 |
| controller success | 83.10% | 81.69% | 85.92% |
| gap vs persistent | -8.45pp | -9.86pp | -5.63pp |
| conditional false handback | 9.23% | 6.15% | 10.77% |
| OFT savings | 22.97% | 18.52% | 30.60% |

0/5 seeds 通过全部 gates。27/71 states 的 handback 时刻随 seed 改变。结论是排序信号
存在，但安全操作点不稳定。

### 2.4 Handback 标签是概率事件

四套件真实 validation smoke：

- persistent replay parity：4/4；
- 历史标签与新 collector：22/27 一致；
- 新 collector 两次重测：25/26 共同边界一致；
- Goal h64 出现标签翻转，且 trajectory termination 时刻不同。

同一个 Goal boundary 的 K=5 same-snapshot 结果：

- h0：3/5 success，经验概率 0.60，one-sided 95% Wilson LCB 0.272；
- h64：5/5 success，经验概率 1.00，但 LCB 只有 0.649。

所以模型目标必须是条件成功概率和不确定性，不能再把一次 rollout 的 0/1 结果当成
无噪声真值。

2026-08-09 随后完成的四套件 K=5、h={0,64} smoke 进一步得到：

- 8 个 boundaries / 40 continuations；
- persistent replay parity 4/4，repeat 字段完整率 100%；
- 3/8 boundaries 为非退化概率；
- Spatial：h0 4/5，h64 0/5，呈显著非单调；
- Goal：h0 3/5，h64 4/5；
- Object 与 Long：两个边界均为 0/5；
- 没有任何边界的 one-sided 95% Wilson LCB 达到 0.5；
- 描述性 probability oracle 为 0.40，best fixed h0 为 0.35，差 5pp，但只有四个 states，
  不作推断性 claim。

该 pilot 通过采集协议 gate，但只允许推进到 16-state 标签熵实验，不允许直接训练或解封
test。

### 2.5 当前世界模型证据的边界

正确配对的 V-JEPA Student/OFT delta 已通过 exact-key 和 distinct-delta 审计，但没有
改善成功—成本 Pareto。该结果只否定“单步 V-JEPA delta 直接拼接”这一实现，不否定：

- 多步 action-conditioned prediction residual；
- ensemble disagreement；
- source/OFT action 的预测分歧；
- 世界模型作为离线 teacher；
- 预测未来风险上升的时序特征。

任何新的 world-model 特征都必须预注册并单独证明 held-out 增量。

### 2.6 B24 正式关闭 safe-handback 主线

冻结的 B24 paired-probability screen 完成 24 states、122 reachable boundaries 和
610 same-snapshot continuations，persistent parity 24/24，协议 gate READY。结果只有
4/24 个 live finite-safe states，其中 3 个在 h0 已经 all-K safe，真正由 OFT 创造安全
交还条件的 state 仅 1/24；没有两个被至少 3 个 states 支持的有限 stopping bins。因此
probability opportunity gate NOT READY，Beta-binomial handback head、第二 VLA、世界模型
和 test 均未解锁。

QC71 上 source-risk privileged trigger 可保持 persistent 的 65/71 success，并将 teacher
steps 从 11049 降到 8630，节省 21.89%。但要达到 20% 部署目标，learned model 必须捕获
91.35% 的 privileged saving，方法裕量过窄。R6 因而先做多个 policy pair 的 model-free
opportunity atlas；只有至少两个 pair 的 privileged savings 达到 30% 才训练实时 risk
model。

---

## 3. 新的核心科学问题

### Q1：当前 source VLA action 是否会把系统带入高失败风险区域？

目标不是判断图像“看起来是否异常”，而是估计：

\[
p_{fail}^{src}=P(Y=0\mid z_t, a_{t:t+H}^{src}, c, \pi_{src}).
\]

风险必须以 proposed action 为条件；同一状态下不同 VLA 的动作风险可能不同。

### Q2：切换纠正策略是否真的增加成功概率？

\[
\Delta_{enter}
=P(Y=1\mid ENTER\_OFT)-P(Y=1\mid CONTINUE\_SOURCE).
\]

只有风险高且纠正优势的置信下界为正时才接管，避免对原本会成功的 source trajectory
造成不必要干预。

### Q3：当前恢复状态是否足以安全交还？

\[
p_{hb}(t)=P(Y=1\mid HAND\_BACK\_NOW,z_t,\pi_{src}).
\]

handback 是概率化最优停止问题。系统需要在失败风险、额外 OFT 成本和置信度之间做决策。

### Q4：风险控制器能否跨 VLA 泛化？

风险不是完全 policy-agnostic。正确目标是共享机器人/任务动力学表示，同时显式条件于：

- source VLA identity/capability；
- action chunk 长度与控制频率；
- canonicalized proposed action；
-历史执行上下文。

### Q5：世界模型相对普通历史编码器是否提供真实增量？

世界模型进入主贡献的必要条件：

1. next-latent prediction 明显超过 persistence baseline；
2. 改善 task-held-out / VLA-held-out calibration；
3. 改善 controller success-cost Pareto；
4. 增益在至少两个 seeds 和第二个 VLA 上复现。

---

## 4. 系统接口与多 VLA 适配

### 4.1 Canonical observation contract

```python
CanonicalObservation = {
    "images": {"front": ..., "wrist": ...},
    "proprio": float[D_p],
    "task_text": str,
    "elapsed_steps": int,
    "control_mode": str,
    "source_vla_id": str,
}
```

视觉编码允许使用每个 VLA 的冻结 latent，也必须提供一个 VLA-independent 的公共视觉
分支，避免共享模型完全绑定某个 backbone 的 hidden dimension。

### 4.2 Canonical action chunk

所有策略输出转换成：

```python
CanonicalActionChunk = {
    "actions": float[H, 7],
    "valid_mask": bool[H],
    "dt": float,
    "frame": "eef_delta",
    "rotation": "axis_angle",
    "gripper": "continuous_normalized",
    "source_policy": str,
}
```

VLA adapter 负责：

- 坐标系转换；
- action range 和单位归一化；
- chunk padding/mask；
- 控制频率对齐；
- gripper convention；
- source policy metadata。

### 4.3 多 VLA 模型结构

```text
Public visual/proprio/history encoder ───────────────┐
                                                     │
Source-VLA frozen latent adapter ────────────────────┤
                                                     ├─► shared state z_t
Canonical action chunk encoder ─────────────────────┤
                                                     │
VLA capability embedding ───────────────────────────┘
                                                     │
                 action-conditioned dynamics evidence
                                                     │
       ┌─────────────────────────────────────────────┼──────────────┐
       ▼                                             ▼              ▼
 source failure head                         recoverability head  handback head
       │                                             │              │
       └──────────────────────► cost/remaining head ◄──────────────┘
```

主模型采用 shared backbone + small VLA adapter/head。独立 per-VLA 模型是强 baseline，
不是最终方法。

---

## 5. 世界模型的正确角色

### 5.1 离线教师，而非在线必需组件

推荐两层结构：

1. **Offline Action-Conditioned Teacher**
   - 可以较大、较慢；
   - 预测 action-conditioned latent transition；
   - 产生 residual、disagreement 和 counterfactual ranking；
   - 只用于训练、审计和消融。
2. **Online LightRiskStudent**
   - 目标规模 0.5M--2M 参数；
   - 不依赖教师 checkpoint；
   - 单次风险推理满足控制周期；
   - 输出校准概率和不确定性。

### 5.2 不做像素视频生成

主线不训练像素级视频模型。推荐预测低维 latent distribution：

\[
(\mu_{t+1},\log\sigma^2_{t+1})
=f_{wm}(z_{t-k:t},a_{t:t+H},c,\pi).
\]

有价值的证据包括：

- next-latent mean/residual；
- multi-step rollout residual；
- ensemble epistemic variance；
- source action 与 OFT action 的 predicted-delta distance；
- latent feasibility / support score；
-对接触、gripper 和任务进度的预测误差。

### 5.3 World-model kill rule

若以下任一条件成立，则世界模型只保留为辅助数据工具：

- prediction 不超过 persistence baseline 10%；
- 加入证据后 controller success-cost Pareto 无改善；
- 改善只存在于 row AUC，不存在于 state-level stopping；
- held-out VLA 增益消失；
- 在线蒸馏模型仍需教师才能推理。

---

## 6. 概率化反事实标签

### 6.1 同一 snapshot 的 K-repeat 标签

在每个 boundary snapshot \(s_{t,h}\) 上运行 K 个 source-VLA continuation：

\[
y_{h,k}\sim Bernoulli(p_{hb}(s_{t,h})),\quad k=1,\dots,K.
\]

保存：

```python
handback_repeat_seeds
handback_repeat_successes
handback_repeat_steps
handback_repeat_stop_reasons
handback_success_count
handback_success_probability
handback_success_wilson_lcb_95_one_sided
handback_success_wilson_ucb_95_one_sided
```

### 6.2 K 的作用不能混淆

- K=3--5：发现标签熵、调试 restore 和估计数据难度；
- K=8--16：训练 soft target 的候选配置；
- K>=52 且全成功：在当前 Wilson 口径下才可能让单边界 LCB 超过 0.95；
- 正式 controller 认证仍以独立 states/tasks 的闭环结果为主，不能把同一 snapshot 的
  52 个重复当成 52 个独立场景。

### 6.3 避免 trajectory-prefix 混淆

重复整个 OFT prefix 测量的是“prefix trajectory sensitivity”，不等于同一 snapshot 的
handback stochasticity。数据中必须区分：

- `same_snapshot_repeat_id`；
- `prefix_repeat_id`；
- `student_policy_seed`；
- `simulator_seed`；
- snapshot fingerprint；
- policy/checkpoint hashes。

---

## 7. 轻量概率风险模型

### 7.1 推荐输出头

1. `source_failure_probability`：继续 source VLA 的失败概率；
2. `enter_recovery_advantage`：进入 OFT 相对继续 source 的成功优势；
3. `handback_alpha_beta`：handback 成功率的 Beta 分布参数；
4. `persistent_success_probability`：继续 OFT 的成功概率；
5. `remaining_oft_cost_quantiles`：剩余接管步数分位数；
6. `ood_or_epistemic_score`：任务/VLA 分布外不确定性。

### 7.2 推荐损失

\[
\mathcal L=
\lambda_{src}\mathcal L_{src-risk}
+\lambda_{hb}\mathcal L_{beta-binomial}
+\lambda_{persist}\mathcal L_{persist}
+\lambda_{cost}\mathcal L_{quantile}
+\lambda_{dyn}\mathcal L_{latent-dynamics}
+\lambda_{distill}\mathcal L_{teacher}.
\]

K-repeat handback 数据使用 Beta-binomial 或 binomial NLL，不把 success fraction 强行阈值化
成 hard label。

### 7.3 不确定性分解

- aleatoric：同一 snapshot、不同合法 source-policy sample 的结果波动；
- epistemic：不同 ensemble/bootstrap model 的预测差异；
- shift：未见任务、VLA、控制频率或 observation adapter。

三者必须分别报告。不能把所有不确定性都称为 world-model uncertainty。

---

## 8. 控制器与最优停止

### 8.1 接管规则

```python
ENTER_OFT = (
    source_failure_lcb >= enter_risk_threshold
    and recovery_advantage_lcb > 0
    and not out_of_support
)
```

若风险不确定且没有安全纠正证据，使用 `SAFE_ABORT` 或继续经过认证的 conservative
baseline，而不是强制选择模型最偏好的动作。

### 8.2 交还规则

```python
HAND_BACK = (
    handback_success_lcb >= handback_threshold
    and handback_advantage_lcb >= -allowed_success_cost
    and safe_boundary_streak >= dwell_boundaries
    and predicted_remaining_cost_saved > switching_cost
)
```

默认 `dwell_boundaries=2`。接管和交还使用不同阈值形成 hysteresis，避免频繁切换。

### 8.3 必须保留的 baseline

- source VLA；
- persistent OFT；
-最佳固定 finite duration；
- risk-only trigger + fixed duration；
- deterministic progress/stagnation handback；
- history-only lightweight model；
- per-VLA independent model；
- shared model without world-model evidence；
- full shared probabilistic model；
- privileged probability/cost oracle。

---

## 9. 数据划分和防泄漏协议

冻结的 R5 split：

- train：72 states / 24 tasks；
- validation：24 states / 8 tasks；
- test：24 states / 8 tasks；
- task/state overlap 均为 0。

正式扩展后最小目标：

- development：至少 300 states；
- calibration：至少 100 个独立 persistent-rescuable states；
- frozen test：至少 100 states；
- 四个 suites；
- 两个 source VLA；
- 第二组 source/corrective policy pair。

所有来自同一 state/episode 的 boundary 和 repeat 必须进入同一 split。标准化、特征选择、
world-model 投影、阈值、dwell 和 ensemble vote 都只能在 train/calibration 上确定。

---

## 10. 跨 VLA 实验协议

### E1：Within-VLA task-held-out

分别在 VLA-A、VLA-B 上做 task-held-out，验证共享模型不是靠 task identity shortcut。

### E2：Joint multi-VLA

比较：

- 两个独立 per-VLA 模型；
- 共享 encoder + per-VLA head；
- 共享 encoder/head + policy embedding；
- 去掉 VLA identity 的错误 baseline。

### E3：Leave-one-VLA-out

在 VLA-A 上训练，VLA-B 零样本测试；然后仅用小规模 VLA-B calibration 调整 adapter/
temperature。报告 zero-shot 与 few-shot 两种结果。

### E4：Unseen policy pair

训练 source-A + recovery-X，测试 source-B + recovery-Y，验证模型是否只学会特定 OFT 的
行为特征。

### E5：Matched-compute

所有方法匹配：

- policy calls；
- world-model calls；
- action chunks；
- GPU latency；
- executed OFT steps。

---

## 11. 下一阶段执行计划

### R5-A：概率标签协议

目标：确认 same-snapshot stochasticity 的规模和来源。

1. 四 suite smoke states，h={0,64}，K=5；
2. 16-state stratified pilot，h={0,16,64,128}，K=5；
3. 分开报告 same-snapshot 与 prefix-repeat 变异；
4. 审计 snapshot、checkpoint、projection、task ID 和 repeat seeds；
5. 输出 label entropy、Wilson interval、continuation-cost 分布。

Gate：

- persistent replay parity = 100%；
- repeat 字段完整率 = 100%；
- 四 suite 均覆盖；
- 至少 20% boundaries 呈现非退化概率或明确证明近似确定；
- 若重复协议本身不能复现 snapshot，则先修 simulator，不训练模型。

### R5-B：概率 head 与 conservative controller

1. binary head 改为 Beta-binomial/binomial-NLL head；
2. cost head 改为 quantile prediction；
3. task bootstrap ensemble；
4. LCB + two-boundary dwell；
5. 5-seed nested OOF；
6. 报告 state-level controller 指标，而非只报 row AUC。

Gate：

- success gap >= -5pp；
- conditional false handback <=5%；
- OFT-step savings >=20%；
- 5 seeds 中至少 4 seeds 通过；
- task-cluster 95% interval 不否定上述约束。

### R5-C：扩数据和独立 calibration

在 R5-B 不通过时，不继续调 MLP；先扩展 independent states。校准集与模型开发集完全分离。

### R6-A：第二 VLA

接入第二个 source VLA，通过 `CanonicalActionChunk` 和 VLA adapter 进入共享模型。先做
public-contract parity，再做 leave-one-VLA-out。

### R6-B：受控 world-model 增量

只比较三组预注册证据：

1. history/state baseline；
2. + one/multi-step latent residual；
3. + ensemble disagreement 和 source/OFT delta contrast。

禁止继续搜索大量 post-hoc projection/scalar 组合。

### R7：冻结测试与闭环

只有 R5/R6 validation gates 全通过后才解封 test。最终至少：

- 100 paired episodes；
- 四 suites；
- 两个 seeds；
- 两组 source/corrective policy pairs；
- task-cluster intervals；
- compute、latency、intervention burden 和 clean-success harm。

---

## 12. 顶会版主张边界

### 可以主张的方向

在未来实验通过后，可主张：

1. safe handback 是策略条件、概率化的最优停止问题；
2. 多 VLA 共享风险模型可以通过 canonical action contract 泛化；
3. 轻量在线 controller 能在接近 persistent recovery 成功率时减少专家接管；
4. world-model teacher 的某类证据在 held-out task/VLA 上提供可复现增量；
5. same-state counterfactual 数据可用于训练，但安全认证必须依赖独立闭环 episodes。

### 当前可以立即报告的诚实结论

- success-only selector 没有 oracle headroom；
- corrected 0.6M model 有 ranking signal，但没有达到安全控制 gate；
- V-JEPA 单步 delta 没有改善 controller Pareto；
- binary handback labels 隐藏了显著的 outcome stochasticity；
- 概率标签和独立安全校准是下一阶段的必要条件。

### 当前禁止的主张

- “模型已经安全部署”；
- “世界模型已经显著提升风险预测”；
- “适配器存在就等于实现跨 VLA 泛化”；
- “模拟器 restore 等于机器人 rollback”；
- “K 次同状态 rollout 等于 K 个独立测试样本”；
- “高 AUC 等于闭环非劣效”。

---

## 13. 推荐论文叙事

### 标题方向

**RASE: Probabilistic Risk-Aware Switching and Safe Handback for Multi-VLA Robot Policies**

### 核心叙事

冻结 VLA 的安全覆盖层不能只检测异常，也不能把恢复看成永久切换到强专家。它必须解决
两个耦合问题：什么时候拒绝 source action，以及什么时候恢复到足以交还。RASE 通过
策略条件的动作表示、概率化反事实监督和保守最优停止，把多个 VLA 与纠正策略统一到
一个轻量实时控制层。世界模型提供离线动作条件证据，但在线执行不依赖大型生成模型。

论文的关键不是比 persistent OFT 多成功几个状态，而是在严格 non-inferiority 约束下，
显著降低 OFT 步数、接管时间和干预负担，并证明该能力能够跨任务和跨 VLA 迁移。

---

## 14. 当前最终决策

- 保留多 VLA 实时风险预测：**GO**；
- 保留动作拒绝、OFT 纠正和 safe handback：**GO**；
- 将 handback 改为概率预测与置信控制：**必须执行**；
- 世界模型作为离线教师/证据源：**有条件 GO**；
- V-JEPA 单步 delta 主线：**NO-GO**；
- 当前 0.6M checkpoint 部署：**NO-GO**；
- 单次二元标签 full-val/test：**PAUSE**；
- R5 same-snapshot K-repeat 和扩展 calibration：**当前最高优先级**。
