# RASE 顶会推进方案 v4（2026-07-29 更新）

## 2026-07-31 dated outcome addendum（优先于下文执行状态）

本 addendum 只记录预注册 gate 的最终 outcome；下文 2026-07-29 方案、原
preregistration 与 kill criteria 原样保留，作为决策前记录。

- W9C 修复了 clean task identity：LIBERO-Plus `suite.tasks[0:10]` 不是 official
  clean-10；clean control 改为 exact-name vanilla BDDL/init，Plus 仅用于
  failure/challenge。
- 对齐 probe **PASS**：Spatial 0.45、Object 0.85、Goal 0.80、Long 0.35，mean
  SR 0.6125；clean32 coverage **PASS**（32/32）；episode-disjoint 与
  task-disjoint readiness 均 **PASS**。
- 最小 ridge 已按计划执行，但两个主比较均触发 kill：episode-held-out 和
  task-held-out 的 learned − action-matched-random mean utility difference 均为
  **0**，bootstrap CI 均跨 0（task n=8，CI [-0.0075, 0.0075]）。
- **最终决策：`kill_method_branch`。** 不推广 ridge，不调参追逐 held-out，不训练
  MLP，不进入 offline/online RL。原“task-held-out 不优于 matched-random → 不上
  MLP/RL”标准已被执行，不得事后放宽。
- **论文姿态：benchmark / diagnosis。** 主证据仍是 W7/W8 的 policy-relative
  recoverability、direct escalation 和 candidate-specific rescue=0；selector 只作为
  诚实负结果、appendix 或 future work。
- **task-held-out 限制：** test split 只有 8 个 clean-control states、没有
  failure-challenge state；learned action 为 7×continue、0×escalate、1×abstain。
  因而该 split 不支持 failure routing、跨任务 escalation 或 selector 成功主张；它只
  支持“预注册 method gate 未通过”。

证据：`runs/ngc_w9c_selector_gate_summary.{json,md}`、
`runs/ngc_w9c_selector_task_splits.json`、
`progress/2026-07-31_w9c_selector_gate_result.md`。

## 2026-07-29 冻结结论（W6/W7）

- W6 配对矩阵：Smol portfolio `0/8`，OFT portfolio `2/8`；方向性差异存在，
  但小样本 exact McNemar `p=0.5`，不能写显著优越。
- W7 prefix attribution：两个 W6 OFT-only 状态中，候选特异性救援为 **0/2**。
  一个状态 direct OFT、zero-prefix、8 个 candidate-prefix 全部成功；另一个状态
  zero-prefix 已成功，因此 candidate 4/5 的成功不能归因于候选内容。
- 原“learned candidate fallback”机制主张停止。当前可支持的机制假设更新为
  **state-conditioned continuation / escalation**，且后续必须加入 time-matched passive
  prefix、clean regret 与 strong-policy cost 控制。
- 已冻结与 W6 episode-group 完全不重叠的 W7 24-state cohort（四个 dim×level cell
  各 6 个状态、24 个独立 episode group）。候选已生成完毕，配对 held-out 流水线运行中；
  观察 held-out outcome 后不得改温度、K 或 horizon。
- W7 的 `t0` 审计为 min=0、median=10、max=36，仍属于 early-stage cohort；它修复
  episode 泄漏与样本量问题，但没有修复 phase coverage，mid/late 必须在下一独立 cohort 验证。
- W7 held-out 最终矩阵：Smol `0/24`、prefix+OFT portfolio `8/24`，exact
  McNemar `p=0.0078125`。W8 从相同 snapshots 直接升级 OFT 得到 `9/24`：Goal
  `4/8`、Long `5/10`、Object `0/2`、Spatial `0/4`。这把主机制进一步收敛为
  direct policy escalation；`8/24` 与 `9/24` 仍须按 state 对齐后才可比较。
- W7/W8 state 对齐已完成：both-success 7、prefix-only 1、direct-only 2、
  both-fail 14；两条 OFT route 的 exact McNemar `p=1.0`。direct OFT 覆盖
  prefix portfolio 命中的 7/8，且是单一可部署动作，因此方法主线冻结为
  direct escalation；不宣称其成功率显著高于 prefix portfolio。

## 0. 结论

当前证据足以支撑一篇有价值的 benchmark / diagnosis 工作，但不足以支撑原设计中的
“SmolVLA learned fallback selector 显著提升恢复”主张。核心原因不是工程失败，而是科学
对象发生了变化：W3–W5 在多个受控 cohort 上反复得到 SmolVLA 候选续完近零，而 OFT
portfolio 在 W4 的同一批候选上恢复 17/32 状态。这个结果应被正面利用，转化为
**policy-relative recoverability 与 cross-oracle asymmetry**，而不是继续把预算堆在同一
SmolVLA 续完协议上。

建议将论文收敛为：

> 一个由视觉扰动诱导、按 policy pair 定义、带 episode-disjoint split 和 per-action
> counterfactual outcome 的恢复决策 benchmark；在此基础上训练一个成本敏感的轻量
> selector，在候选执行、升级到强 policy、rollback/replan 与 abstain 之间决策。

其中 benchmark 是主贡献，selector 是验证 benchmark 可用性的 method contribution。
不要再把“五种 fallback 的统一空间”本身写成 novelty；HELM、VoLo、B2FF、AEGIS 已经
显著压缩了这个空间。

## 1. 现有证据能说什么

可以进入论文主表：

- SmolVLA clean LIBERO 70.0%（2,000 episodes，冻结配置）。
- LIBERO-Plus camera/robot full collapse 0.38%（3,142 completed）。
- ForkableEnv 的 deterministic snapshot/restore gate 已通过。
- W4 ADEQUATE 32 状态：SmolVLA 0/1,536，OFT portfolio 17/32，候选命中 65/256。
- W5 t=0.7：OFT-recovered cohort 与 failure frontier 上 SmolVLA screen 均零命中。

必须降级或禁止的主张：

- 不把 failure-biased pool 上的 32 个状态外推为全局 NGC 率。
- 不把 OFT deterministic one-shot outcome 写成 Wilson-certified candidate probability。
- 不把“SmolVLA 候选 + OFT 续完可救”写成 SmolVLA 自身可恢复。
- 不用当前全 `t0=0` 的 W4 样本做恢复时机结论。
- 不在同一 episode 的不同 snapshot 之间随机切 train/test。
- 没有真正 rollout 过 fallback action 前，不训练或宣称统一 fallback selector。

## 2. 论文问题重写

### RQ1：恢复性是否是 policy-relative 的？

定义 `R(candidate_policy, continuation_policy, state)`，明确区分：

- Smol→Smol；
- Smol→OFT（escalation / portfolio）；
- OFT→OFT；
- 至少一个鲁棒化 backbone→自身或 OFT。

主结果不是“一个 oracle 是真理”，而是不同 policy pair 的 recoverability matrix、方向性
不一致和 paired exact test。

### RQ2：什么扰动、阶段与物理状态决定可恢复性？

至少覆盖：suite × camera/robot × L1–L5 × early/mid/late t0 × episode outcome。
统计单位是 episode 或 task，不能把同轨迹 snapshot 当独立样本。因果措辞仅用于受控扰动
对照；观察性 pool 切片统一写 association。

### RQ3：能否在同预算下学会“何时执行、何时升级/恢复、何时 abstain”？

第一版 method 不应直接上 online DQN。先用已标注 action outcome 做 cost-sensitive
supervised selector，证明：

1. 比 candidate-only reranker 降低 FEB；
2. 比 always-escalate 节省强 policy 调用；
3. 比 rule threshold 降低 clean regret；
4. 在 task-held-out split 上仍成立。

只有监督 selector 在 held-out task 上出现稳定正增益后，才进入 offline RL / online DQN。

## 3. 数据协议（必须先完成）

### 3.1 分层与拆分

- 新池至少保留 success 与 failure episode；当前 3666/3666 failure 池不能承担 clean-regret。
- snapshot 分层覆盖 t0，而非再次使用 earliest-only。
- train/val/test 按 `(task_id, episode_id)` 成组；最终主表再增加 task-held-out 版本。
- 每个 split 发布 suite/dim/level/label 的精确 cell count 与空 cell 警告。

仓库已新增 `*.benchmark-splits.json`：它按 episode 分组并提供 split audit。旧的
`*.splits.json` 只是 label 索引，不得用于训练/测试划分。

### 3.2 标签语义

每个 action arm 保存：policy pair、candidate/fallback 类型、trial 数、成功数、seed、成本、
终止原因。对确定性 OFT 必须写 `deterministic_one_shot`；对随机 continuation 使用预注册的
sequential Wilson 协议。

### 3.3 fallback 最小集合

2026-07-29 冻结首版为三个具有清晰部署语义的 arm：

1. `CONTINUE_SMOL`：从 snapshot 空前缀直接由 SmolVLA 续完；
2. `ESCALATE_OFT`：从 snapshot **直接**切换 OFT，不带候选或 zero prefix；
3. `ABSTAIN`。

`candidate_0..K-1` 不作为 K 个 selector class；它们仅保留为 proposal 诊断和
candidate-oracle/reranker 上界。`any_of_K` portfolio outcome 不是一个可部署动作，不能直接当
`CONTINUE/ESCALATE` 标签。仓库的 selector readiness audit 会拒绝此类 `proxy=true` 数据。

selector 输入不得包含 ground-truth `perturb_dim/sub/level` 或 episode outcome。
这些字段只服务分层统计。W9 首版使用当前 RGB summary、proprio 与 t0；若 held-out
能力不足，再增加 frozen DINO/Smol feature，不允许用标注泄漏换取可分性。

随后按信息增益加入 `ROLLBACK_REPLAN`（HELM-like）和 `MILESTONE_REPLAN`（B2FF-like）。
`WAIT` 在静态 LIBERO-Plus 中预计价值有限，放到消融或删除；`REPLAN-text` 因模型忽略语言
风险高，只作负对照。这样可以避免先实现五个复杂 fallback、最后没有足够正例可训练。

## 4. 方法最小可发表版本

输入先用低成本、能在部署时获得的信号：

- ACC / action statistics；
- 当前 VLA frozen feature 或 DINO feature；
- episode progress、suite、上一次 action；
- candidate dispersion；
- 一个从成功/失败轨迹训练的 calibrated value probe。

模型按顺序增加：logistic/linear probe → 2-layer MLP → candidate-shared attention selector。
每个模型输出 action utility 与 abstain/escalation cost。先做 supervised cost-sensitive learning，
再决定是否需要 CQL/DQN。V-JEPA 2 world-model signal 放在方法消融后加入，避免基础标签还不
稳定时增加系统复杂度。

## 5. 必做对照与统计

最低主表：

- frozen base policy；
- random trigger（相同 escalation budget）；
- always escalate（性能上界/成本下界参照）；
- candidate-only oracle/reranker（FEB identity 实测）；
- AEGIS-like calibrated risk threshold；
- HELM-like fixed trigger + rollback/replan；
- learned selector。

实验采用 paired common-random-number episodes。主比较报告 paired difference bootstrap CI；
二元 paired outcome 报 exact McNemar，并对预注册主比较做多重校正。所有比例同时报告状态/
episode 级 Wilson CI。至少 3 个训练 seed；评测 episode seed 固定且各方法共用。

主指标：task success、net success、FEB、clean regret、strong-policy usage、wall-clock latency。
不要只报恢复成功率，否则 always-escalate 或 always-abstain 可以投机。

## 6. Kill criteria

以下条件触发转向，避免浪费 GPU：

- L1–L2 + success-retained pool 仍无法产生至少 100 个可恢复和 100 个不可恢复 episode-group：
  暂停 selector，投稿 diagnosis/benchmark track。
- `ESCALATE_OFT` 相对 base 在 paired pilot 上净恢复低于 5pp：停止复杂 selector。
- 简单 linear/MLP selector 在 task-held-out split 不优于 random-trigger：先修数据与特征，
  不上 DQN。
- rollback/replan 在 50-state pilot 中没有超过执行成本的净收益：从主动作空间删除。

## 7. 推荐执行顺序

### Gate A：代码与 release protocol（现在）

1. 全量 unit tests 与 preflight。
2. 重新生成 W4 dual-oracle summary，获得 agreement/McNemar 字段。
3. 导出 recovery dataset 与 episode-grouped benchmark splits。
4. 审计 split cell、候选 artifact、标签语义。

### Gate B：低成本诊断（半天内）

1. **已完成：** t=0.3/0.7/1.0 在同一 24-state frontier 上合计 `0/576`；
   温度元数据、候选差异与 diversity 均通过核查，proposal-temperature 线永久停止。
2. **已完成：** L1–L2 采集得到 40/40 failure episodes、762 states；结合既有
   full collapse 的 camera L1/L2 `0/558`，不再用暴力补采 success 作为阻塞 gate。
   冻结 `dim × level` 四 cell、每 cell 两个不同 episode 的 8-state
   failure-challenge cohort，并显式条件化报告。
3. **已完成：** 冻结 challenge pilot 的 Smol→Smol 为
   `0/64 candidates、0/8 states`，Smol→OFT 为 `10/64 candidates、2/8 states`；
   状态级 exact McNemar `p=0.5`。随后 prefix attribution 得到候选特异性救援
   `0/2`，因此将发现解释为 continuation/escalation sufficiency，而非 proposal quality。
4. **已完成：** episode-disjoint 24-state held-out 矩阵为 Smol `0/24`、
   prefix+OFT `8/24`，exact McNemar `p=0.0078125`。W8 direct OFT 为
   `9/24`；CPU-only state overlap analysis 已实现，禁止用两个边际比例替代配对表。
5. clean-success control 作为独立 cohort 构建，用于 clean-regret 与升级成本；不得与
   failure-challenge 合并成一个无条件比例。
6. **已完成：** pool-support audit 为 0 clean-success episode groups；W9 必须新采
   clean L0 controls，不能通过降低 readiness threshold 绕过缺失数据。

### Gate C：最小 selector 数据（1–2 周）

0. **代码 MVP 已完成：** 三臂 state-level schema、proxy/leakage/label-collapse readiness gate、
   dependency-free ridge utility baseline、always/matched-random/oracle 对照和 episode/task split 工具。
1. **已完成：** W7 全 24 states 的 `direct OFT from snapshot` 为 `9/24`。W8
   candidate-0 export 仅作历史诊断；W9 将补 true direct Smol 后才形成最终
   deployable failure-challenge action outcomes。portfolio matrix 仅作诊断。
2. 将 failure-challenge 与 clean-success control 分层扩展，保持 episode-disjoint；
   优先补稀缺 action-outcome positive group，而不是强行平衡采集 episode outcome。
3. 标注 `CONTINUE/ESCALATE/ABSTAIN` 三臂 outcome 与成本。
4. readiness audit 通过后训练 linear selector，并同时跑 episode-heldout 与 task-heldout；
   linear 未超过 matched-random 时不进入 MLP/RL。

### Gate D：方法扩展（仅 Gate C 通过）

1. 加 HELM-like 与 B2FF-like arm；
2. 加 ACC/value-probe 与 world-model feature 消融；
3. 再考虑 offline CQL / online DQN；
4. 第二 backbone 与鲁棒化模型验证 policy-relative 结论。

## 8. 投稿定位

顶会版本必须同时具备：可复现 benchmark release、至少两个能力层级 backbone、严格无泄漏
split、paired statistical protocol，以及一个在成本/成功率 Pareto 上优于 rule baselines 的
learned selector。若 method 未达标，优先保住高质量 benchmark/diagnosis，不要用过度复杂但
证据不足的 RL 叙事稀释现有最强发现。
