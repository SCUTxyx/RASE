# RASE K3 独立动作信号 Pilot（revised protocol）

## 0. 目标与边界

本阶段只回答三个问题：

1. 在同一物理状态下，`continue / requery / resample / fallback / abort` 候选是否能够被完整、同步、可复现地捕获；
2. 动作轨迹本身是否能在 task-held-out 条件下改善短期风险、success 或 progress 预测；
3. 基于冻结风险阈值的离线 candidate 选择，是否优于 identity/continue 基线。

本阶段不声称：multi-VLA 泛化、实时闭环增益、D-PASS、semantic pretraining gain、OPD/RL/world-model gain。

当前 A 为 A-PARTIAL，历史 Phase-C cohort 仍为 B-FAIL。因此本阶段是明确标注的单策略开发 pilot，不是 pooled multi-VLA 主实验。

## 1. 术语固定：provider 与 semantic perturbation 分开

本 K3 的 candidate operator 指 selector-level provider，而不是 D0 的动作扰动：

```text
continue.source
requery.source
resample.source/candidate.0
resample.source/candidate.1
fallback.persistent
abort.safe
```

其中：

- `continue.source`：当前 VLA 已生成的继续动作块；
- `requery.source`：同一 boundary、同一 VLA 的重新推理动作块；
- `resample.source/candidate.0/.1`：同一 source inference 的两个独立候选样本；
- `fallback.persistent`：fallback 控制器的完整动作块；
- `abort.safe`：安全停止/回退控制事件，不一定有 simulator action chunk。

D0 的五种 `identity / sign flip / temporal reverse / gripper shift` 仍是独立的 semantic-counterfactual protocol，不与本 K3 provider 组合。否则样本量和归因都会失控。

所有 root 必须使用完全相同、顺序固定的六个 operator slot。缺失或不可执行候选不能事后删除，只能写入 capability mask。

## 2. K3-E0：工程准入门槛

### 2.1 inference-time capture 修复

修改：

- `scripts/collect_rase_vnext_discovery.py`
- `rase/vnext/candidate_capture.py`

source/requery/resample 必须在 inference-time 直接保存 native model output；禁止从后续 queue 状态反推动作。候选生成与物理状态快照之间的关系必须是：

```text
freeze physical root + observation
        ↓
one inference event with frozen seed ledger
        ↓
capture native output chunks
        ↓
canonical conversion / hash / mask
        ↓
execute branches from the same root snapshot
```

### 2.2 冻结字段

每个 operator slot 至少保存：

```text
root_id
boundary_id / decision_point_id
operator_id
source_policy_id
candidate_generation_seed
execution_seed(s)
horizon
shape
dtype
raw action array
canonical action array
valid-step mask
raw_sha256
canonical_sha256
raw_canonical_raw error
capture timestamp / inference event id
capability status and reason
```

action provider 的标准 horizon 暂定为 `H=10`、shape=`[10,7]`、finite float32。`abort.safe` 的预注册协议为 `H=0`、shape=`[0,7]` 或显式 control-event schema；不能用零动作伪装成 abort。

若 source/requery/resample 不能提供完整 H=10 chunk，则 K3-E0 失败；不能通过 padding、重复最后一步或读取后续 queue 补齐。

### 2.3 E0 审计与测试

扩展 `tests/test_vnext_candidate_capture.py`，至少覆盖：

- 完整 native chunk 捕获，而非 queue snapshot；
- 所有六个 operator slot 的 schema、horizon、dtype、mask；
- 同一 boundary/root 关联；
- raw→canonical→raw 最大误差为 0；
- frozen seed 下重复 capture hash 一致；
- branch execution 使用同一个 root snapshot；
- 不可执行候选被标记 `incapable`，不会自动转成普通 failure；
- 任一 capture parity 失败时，collector 以非零状态退出且不生成正式 K3 manifest。

E0 通过条件：所有冻结 operator 的 capture parity 通过；任何系统性 source/requery/resample 缺块都停止后续实验。

## 3. 独立 cohort 与冻结协议

### 3.1 cohort

正式 cohort：

```text
8 tasks × 3 roots/task = 24 physical roots
```

默认每个 suite 选 2 个 task：Spatial、Object、Goal、Long 各 2 个。Object/Long 即使在 D0 中表现差，也必须保留，用于 worst-group 和 capability 分析。

从 `root_catalog_v1.json` 或新 catalog metadata-only 选择。必须排除：

- 所有 B2/D0 root_id；
- 相同 state snapshot hash；
- 预注册近重复规则命中的 qpos/qvel/image snapshot；
- 同一物理 episode 的重复 decision point，若 protocol 规定不允许。

近重复阈值必须在 outcome 观察前写入 protocol；建议至少固定 qpos/qvel L∞ 阈值和图像 perceptual-hash 阈值，不能运行后再调。

root、task、decision point、checkpoint、simulator version、seed ledger 和 exclusion 规则全部在 freezer 输出后冻结。

### 3.2 rollout 数量

六个 operator slot、每个 root/operator 执行 `K=3` 个匹配 seed 的真实 rollout：

```text
8 tasks × 3 roots/task × 6 operators × 3 repeats = 432 operator slots
```

其中：

- capture 层始终冻结 432 个 operator slots；
- 若 `abort.safe` 按 protocol 不进入 simulator，则其 72 个 slots 仍必须有完整 control-event/capability record，但不计入 simulator rollout；
- 此时 simulator executions 为 `8 × 3 × 5 × 3 = 360`；
- `incapable` 不计入普通 failure，也不得事后从分母剔除。

`K=3` 定义为每个 root/operator 的三次独立执行，不是“三个 operator”。三次执行共享同一个已冻结 candidate chunk，使用三个冻结的 execution seeds；三个 operator 使用匹配的 seed triplets。

### 3.3 freezer 输出

必须生成：

- `frozen_manifest.json` 与 sha256；
- `PROTOCOL.json`；
- cohort selection report；
- root/state/near-duplicate exclusion report；
- operator schema and capability contract；
- seed ledger；
- B2/D0 exclusion list。

freezer 不得读取 baseline success、progress、candidate outcome 或 capability 结果来选 root/task。

## 4. K3 collector 与 richer outcomes

以 D0 的 restore、preflight、resume-safe、same-root execution 为基础，新建 K3 collector/runner。

每次重复都必须：

1. 从相同 root snapshot restore；
2. 使用同一个 frozen candidate chunk；
3. 使用 operator-matched execution seed；
4. 记录完整执行 provenance；
5. 将 capture、execution、metric 三层输出分开审计。

每个 `root/operator/repeat` 保存：

```text
success
progress_delta
distance_to_target_curve
collision / out_of_bounds
gripper/contact phase
recovery_time
query_count
fallback_steps
latency samples
termination_reason
full MotionTrace
capability mask
```

任何指标不可用必须写入 `metric_mask` 和原因，不得用 0、failure 或 horizon 代替。

capability 状态至少区分：

```text
executable
incapable_missing
incapable_short_chunk
incapable_invalid_action
incapable_actuator_mismatch
control_only_abort
execution_error
```

`incapable_*` 不等价于任务失败；`execution_error` 也必须与 simulator terminal failure 分开。

## 5. 预注册特征与数据拆分

扩展 `scripts/export_rase_vnext_phase_c_features.py`，只导出四组预注册输入：

1. `state-only`：proprio/state/history；
2. `raw-action`：原始动作统计；
3. `trace-only`：CanonicalMotionTrace；
4. `trace+semantic`：仅由动作自身计算的运动学/时序特征。

`trace+semantic` 可以使用速度、加速度、jerk、方向变化、路径长度、gripper event timing、temporal phase 等；不得使用 outcome、未来状态、成功标签、候选最终分数或 operator ID。

模型第一阶段限定为固定正则的 linear/ridge，超参数只能在 inner fold 选择；不得用 outer fold outcome 调参。

正式评估使用 task-held-out folds：

- 8 tasks 固定为 4 folds，每 fold 2 tasks；
- 同一 task 的 3 roots 和 3 repeats 必须在同一 fold；
- 同一 root 的所有 operator branches 不得跨 fold；
- fold 划分在 outcome 产生前冻结。

## 6. 评估设计

### 6.1 主指标

- same-root pairwise ranking：先在 root 内聚合 K3，再做 task-level 汇总；
- action-swap sensitivity；
- risk-coverage；
- Brier score / ECE；
- progress prediction。

次指标：oracle regret、干预后 success、恢复时间、query/fallback/latency 综合成本。

所有指标同时报告 overall、suite、task、operator、capability status，并提供 task-level bootstrap 区间。

### 6.2 action-swap

action-swap 必须只在 capability-compatible 的候选之间执行，并保持 state 不变：

- same-task cross-root action swap；
- cross-task action swap；
- operator-matched swap，例如只交换两个 `fallback.persistent` chunk。

预测模型不得使用 operator ID，因此 action-swap 的变化只能来自动作/trace 特征。报告 swap coverage，不能把无效动作或 incapable candidate 当作有效 swap。

### 6.3 离线 utility

离线 selector 只能在 inner fold 学习风险阈值/选择规则，outer fold 只评估一次。比较：

- identity/continue baseline；
- always fallback；
- frozen risk-threshold selector；
- oracle candidate choice（仅作上界）。

selector 输出必须记录 selected operator、risk、coverage、success/progress、fallback cost 和 latency。

## 7. 事前 Gate

### K3-CAPTURE-PASS

必须满足：

- 432 个 operator slots 都有完整 provenance；
- 所有可执行 provider 的 native chunk 满足 H/shape/dtype/mask；
- raw/canonical hash 与 replay parity 通过；
- root/operator/repeat 关联无缺失、无重复、无泄漏；
- incapable/control-only 状态按 protocol 明确记录。

任何系统性 source/requery/resample 缺块，停止统计分析并回到 E0 修复。

### K3-SIGNAL-PASS

在运行前冻结以下 provisional pilot criteria：

- `raw-action` 或 `trace+semantic` 相对 `state-only` 的 primary same-root ranking 增益至少 `0.03`；
- task-level bootstrap 方向为正，且至少 3/4 task folds 方向一致；
- same-task 和 cross-task action-swap 方向符合预期，至少 3/4 folds 不退化；
- Brier 增幅不超过 `0.02`，ECE 增幅不超过 `0.05`；
- 结果必须按 capability mask 分层报告。

这些是 K3 pilot 的继续/停止门槛，不是最终论文显著性标准。

### K3-UTILITY-PASS

冻结阈值下的离线 selector 相对 identity/continue baseline 必须满足：

- success 或 progress 有至少 `0.05` 的 paired improvement，或 oracle regret 相对下降至少 `10%`；
- recovery/query/fallback 综合成本不增加超过 `10%`；
- risk-coverage 不恶化；
- 至少 3/4 task folds 方向一致。

若只有 capture 通过，结论为工程链路可用；若有 outcome sensitivity 但 task-held-out 不稳定，保留 `PILOT_SIGNAL_WEAK`；只有 SIGNAL 与 UTILITY 同时通过，才进入小规模 semantic selector 设计。

## 8. 执行顺序与停止规则

### K3-E0 smoke

用一个不进入正式 cohort 的 task/root 做 6-slot capture + capability smoke。若 source/requery/resample 不能得到完整 native chunk，立即停止。

### K3-Freeze

E0 通过后冻结 8 tasks × 3 roots cohort、协议、seed、fold 和 operator slots。冻结后禁止更换 task、root、operator、horizon 或阈值。

### K3-Collect

按 432 slots 采集；`abort.safe` 若不进入 simulator，则仍保留 72 条 control-only records，并按预注册口径统计 360 个 simulator executions。

### K3-Analyze

先做 capture/execution/metric audit，再导出 feature、OOF predictions、action-swap、bootstrap、calibration 和 offline utility。

### 后续限制

在 K3-SIGNAL-PASS 与 K3-UTILITY-PASS 之前，不启动：

- multi-VLA pooled selector；
- OPD 训练；
- RL；
- world model；
- 大规模公开数据/teacher 下载；
- 实时 closed-loop deployment。

即使 K3 两个 gate 都通过，也只能进入更大 K5、held-out policy 和独立 simulator 验证。

## 9. 产物与可复现性

原始结果目录只追加，不覆盖 B2、D0 或历史 Phase-C。必须保存：

- frozen manifest / protocol / hashes；
- root selection 与 exclusion report；
- 每个 capture、execution、metric record；
- capability summary；
- feature manifest；
- fold assignment；
- OOF predictions；
- action-swap report；
- bootstrap/calibration report；
- offline utility report；
- 一页 gate decision；
- 单元测试、静态检查、E0 smoke 日志。

最终论文口径只能根据 gate 状态更新，不能把 K3 pilot 的 feasibility 或 weak signal 写成已完成的 multi-VLA 风险控制系统。

