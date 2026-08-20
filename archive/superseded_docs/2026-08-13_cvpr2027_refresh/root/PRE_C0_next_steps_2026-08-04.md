# PRE-C0 下一阶段执行计划

**日期：** 2026-08-04  
**适用阶段：** PRE-C0 same-policy corrective headroom audit  
**目标：** 完成 natural same-policy headroom 审计；只有 Gate A 明确失败时，才运行 privileged guidance audit。

---

## 0. 当前状态

```yaml
pre_a3_method_gate: closed
hidden_pre_a3_test24: sealed
world_model_gate: closed

guidance_infrastructure: green
guidance_unit_tests: green
guidance_protocol_wiring: green_with_qc_pending

trajectory_collection_24: running
deviation_mining: pending
stage_key_qc: pending
natural_gate_a_smoke: pending
natural_gate_a_48_state_audit: pending
privileged_gate_b: not_run
```

已经完成：

- 修复 `GuidanceResult.__eq__`，使 numpy action tensor 可确定性比较；
- `tests/test_guidance_trust_region.py` 与 `tests/test_guidance_determinism.py` 共 14 项测试通过；
- 在 `rase/collect/pre_c0.py` 中加入 `analyze_guided_headroom()`；
- 新增 `scripts/run_privileged_guidance_audit.py`；
- 将 `scripts/generate_smolvla_corrective_candidates.py` 接入 `RecedingHorizonSmolVLAContinuation`；
- 扩展 `tests/test_pre_c0_protocol.py`，覆盖 Gate B PASS 与 frozen-NOGO 路径。

当前结果只证明 guidance 基础设施可运行、可测试，不证明 Gate A 或 Gate B 已通过，也不证明 privileged guidance 能提升 SmolVLA 成功率。

---

# 1. 必须遵守的执行顺序

```text
完成 24-trajectory collection
→ collection integrity QC
→ deviation mining
→ stage-key QC
→ 冻结 48-state manifest
→ Natural Gate A smoke
→ Natural Gate A full audit
→ 仅当 Gate A FAIL 时运行 privileged guidance audit
→ Gate B analysis
→ 根据 PASS / NOGO 进入下一阶段
```

禁止：

- 在 collection 未完成前解释中间成功率；
- 在 stage-key QC 前挑选“表现更好”的状态；
- 在看到结果后修改 Gate A/B 阈值；
- 因 guidance 代码已经完成而强行运行 Gate B；
- 解封 PRE-A3 hidden test24；
- 重开 finite handback、termination selector 或 world model gate；
- 将当前 CPU action refinement 描述为 SmolVLA flow API guidance。

---

# 2. 完成 24 条轨迹采集

## 2.1 确认正式结束

先确认 tmux `pre-c0` 中出现：

```text
PRE_C0_COLLECT_EXIT
```

若进程停止但没有该标记，应按异常退出处理。

## 2.2 Collection 完整性检查

记录：

```yaml
expected_trajectories: 24
completed_trajectories: <actual>
failed_trajectories: <actual>
timed_out_trajectories: <actual>
missing_snapshots: <actual>
restore_failures: <actual>
duplicate_seeds: <actual>
invalid_episodes: <actual>
```

逐项检查：

- 每条 trajectory 的 task、episode、seed 是否唯一；
- RGB、proprioception、action history 是否完整；
- simulator snapshot 是否可恢复；
- 是否有异常提前终止；
- success、failure、timeout 是否使用统一定义；
- 是否有重复 seed、snapshot ID 或输出目录；
- 是否有部分写入或损坏的数据；
- collection 完成顺序是否与任务难度或轨迹长度相关。

建议输出：

```text
progress/2026-08-04_pre_c0_collection_integrity.md
artifacts/pre_c0/collection_integrity.json
```

进入下一阶段的最低条件：

```yaml
collection_exit_marker_present: true
all_included_snapshots_restorable: true
missing_required_history: 0
duplicate_seed_or_snapshot_id: 0
exclusion_rules_frozen: true
```

---

# 3. Deviation mining

## 3.1 需要提取的阶段

每条失败 trajectory 尽量提取：

```text
T0: last stable snapshot
T1: first measurable deviation
T2: first sustained deviation
T3: clear failure-in-progress
T4: terminal failure snapshot
```

同时保留：

- clean success snapshots；
- benign high-risk snapshots；
- contact / non-contact 状态；
- camera perturbation；
- robot perturbation；
- 不同任务阶段。

## 3.2 阶段定义

### T0：Last stable

偏离发生前最后一个稳定状态。不得根据候选 rollout 或恢复结果事后选择“最容易恢复”的 T0。

### T1：First measurable deviation

由部署时可观察信号定义，例如：

- observed motion 与预期 motion 不一致；
- task progress 首次明显回退；
- grasp stability 下降；
- contact event 异常；
- 当前 suffix 与最新 observation 不一致。

### T2：First sustained deviation

要求偏离信号持续若干步，避免单帧噪声。

### T3：Failure in progress

系统已明显向失败发展，但尚未达到 terminal failure。

### T4：Terminal failure

必须区分：

```text
true terminal failure
episode timeout
safety termination
simulator error
```

## 3.3 防止标签泄漏

Stage mining 不得使用：

- future candidate rollout outcome；
- candidate oracle；
- guided candidate 表现；
- hidden PRE-A3 test；
- 事后人工挑选的最佳恢复窗口。

允许使用当前及过去的 observation、proprioception、action，以及预先固定的阶段规则。

---

# 4. Stage-key QC

## 4.1 时序一致性

每条轨迹应满足：

\[
T0 < T1 \leq T2 \leq T3 < T4
\]

缺失阶段必须明确标记，不能强行补齐。

## 4.2 唯一性与覆盖

检查：

- 同一 snapshot 是否被重复标记为多个阶段；
- 同一状态是否因多个 mining rule 被重复计入；
- 单条 trajectory 是否在 48-state audit 中占比过高；
- suite、task、扰动、contact 和阶段是否尽量平衡；
- clean success 与 benign high-risk 是否有负样本覆盖。

建议输出：

```text
artifacts/pre_c0/stage_keys.json
progress/2026-08-04_pre_c0_stage_key_qc.md
```

每个 state 至少记录：

```yaml
state_id:
trajectory_id:
task_id:
suite:
episode_id:
seed:
stage:
timestamp:
snapshot_path:
history_start:
history_end:
contact_state:
perturbation_type:
base_outcome:
mining_rule_version:
```

---

# 5. 冻结 48-state audit manifest

运行 Gate A 前冻结：

- 48 个 state 的完整列表；
- inclusion / exclusion rules；
- stage mining 版本；
- simulator、代码和 policy commit；
- SmolVLA checkpoint；
- sampling 参数；
- candidate 数量；
- generation horizon；
- execution horizon；
- success / harm 定义；
- Gate A/B 阈值；
- bootstrap 方法；
- 随机种子。

建议输出：

```text
artifacts/pre_c0/pre_c0_48_state_manifest.json
artifacts/pre_c0/pre_c0_protocol_lock.yaml
```

建议锁定：

```yaml
repository_commit:
policy_checkpoint:
simulator_version:
candidate_generator_version:
continuation_runner_version:
analysis_version:
stage_mining_version:
```

看到主结果后不得修改：

- Gate A 5 pp 阈值；
- cluster bootstrap 单位；
- harmful replacement 定义；
- 主结果状态集合；
- candidate arm 数量；
- execution horizon；
- invalid rollout 处理方式。

---

# 6. Natural Gate A smoke test

Smoke test 只验证协议和代码路径，不作科学结论。建议使用 4–8 个预先冻结的状态。

## 6.1 Candidate arms

单次候选：

```text
A0: Current SmolVLA suffix
A1: 8 × strict same-distribution resamples
A2: 4 × fresh SmolVLA replans
```

闭环协议：

```text
P0: full chunk
P1: fixed execution horizon = 1
P2: fixed execution horizon = 2
P3: fixed execution horizon = 4
```

## 6.2 Strict resample contract

必须保持：

- 同一 simulator snapshot；
- 同一 RGB / proprioception；
- 同一 history window；
- 同一语言指令；
- 同一 cache 初始化；
- 同一 temperature、top-p / top-k；
- 同一 flow / diffusion schedule；
- 同一 generation horizon；
- 同一 execution protocol；
- 只改变随机 seed。

## 6.3 Fresh replan contract

必须：

- 丢弃旧 suffix；
- 使用最新 observation 和 proprioception；
- 重建最近 history；
- 重建 policy cache；
- 使用同一 SmolVLA checkpoint。

Fresh replan 不得称为 strict same-distribution resample。

## 6.4 Smoke 验收

确认：

- 所有 arm 使用同一 continuation runner；
- episode budget、success detector、harm detector 一致；
- action clipping 和 control frequency 一致；
- cache reset 行为一致；
- seed 可复现；
- first-action latency 被记录；
- receding horizon 没有被混入 candidate generation 对比。

---

# 7. Natural Gate A 48-state audit

## 7.1 嵌套指标

\[
S_0 = S_{current}
\]

\[
S_1 = S_{current+strict\ resample\ oracle}
\]

\[
S_2 = S_{current+resample+fresh\ replan\ oracle}
\]

\[
S_3 = S_{natural+fixed\ short\ horizon}
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

## 7.2 必须分开报告

- current vs strict resample；
- strict resample vs fresh replan；
- full chunk vs \(H=1\)；
- full chunk vs \(H=2\)；
- full chunk vs \(H=4\)；
- best fixed horizon；
- per-state horizon oracle。

计算：

\[
H_{adaptive-horizon}
=
S_{per-state\ horizon\ oracle}
-
S_{best\ fixed\ horizon}
\]

只有该值足够大，才允许训练 adaptive horizon selector。

## 7.3 Gate A 判定建议

```yaml
natural_headroom_point_estimate: ">= 5 pp"
trajectory_cluster_bootstrap_lower_bound: "> 0"
positive_direction_in_at_least_two_suites: true
not_driven_by_single_task: true
clean_success_harm_controlled: true
```

同时报告：

- snapshot-level success；
- trajectory-cluster bootstrap CI；
- suite、task、perturbation 分组；
- leave-one-task-out sensitivity；
- clean-success preservation；
- harmful replacement；
- latency；
- VLA forward 次数；
- wall-clock time。

## 7.4 Gate A 分支

### PASS

```yaml
natural_same_policy_gate: open
candidate_critic_gate: eligible
privileged_guidance_gate: not_required
```

先确认增益来自 sampling、reconditioning 还是 fixed short-horizon。不要为了使用已完成的 guidance 代码而强行运行 Gate B。

### INCONCLUSIVE

```yaml
natural_same_policy_gate: inconclusive
```

扩大 natural audit，检查 cluster CI、单 task 驱动和 best fixed horizon。暂不训练复杂模型。

### FAIL

```yaml
natural_same_policy_gate: fail
privileged_guidance_gate: eligible
```

只有此时才运行 privileged guidance audit。

---

# 8. Privileged guidance audit

## 8.1 准确命名

当前没有 SmolVLA flow API injection，因此应称为：

```text
privileged trust-region action refinement
```

或：

```text
post-hoc privileged candidate optimization
```

不要称为 `guided SmolVLA flow generation`。

## 8.2 核心 arms

```text
B0: Natural Best-of-K
B1: Matched-compute larger Best-of-K
B2: Privileged trust-region action refinement
B3: Oracle candidate selection
```

真正的 guidance 独立增量：

\[
H_{guidance}
=
S_{privileged\ refinement}
-
S_{matched-compute\ Best-of-K}
\]

不能只比较 privileged refinement 与 current，否则候选数量和额外计算会被错误归因给 guidance。

## 8.3 Matched-compute 资源

记录并尽量匹配：

- VLA forward 次数；
- candidate 数量；
- simulator rollout 次数；
- privileged reward evaluations；
- CPU 时间；
- GPU 时间；
- wall-clock latency；
- peak memory。

## 8.4 Trust-region 审计

至少检查：

- per-step action delta；
- full-chunk norm；
- velocity、acceleration、jerk；
- workspace violation；
- gripper discontinuity；
- first-action discontinuity；
- action saturation；
- invalid action rate；
- collision；
- object drop；
- irreversible event。

数值 trust region 通过，不代表动作仍处于 SmolVLA 高概率支持区，必须以真实 closed-loop rollout 作为最终证据。

## 8.5 Gate B 判定建议

```yaml
gain_over_matched_compute_best_of_k: positive
guidance_headroom_point_estimate: ">= 8-10 pp"
trajectory_cluster_bootstrap_lower_bound: "> 0"
positive_direction_in_at_least_two_suites: true
clean_success_harm_controlled: true
trust_region_violation_rate: below_limit
invalid_action_rate: below_limit
irreversible_harm_not_increased: true
```

---

# 9. Gate B 后续分支

## 9.1 Gate B PASS

```yaml
privileged_guidance_gate: open
learned_recovery_critic_gate: eligible
```

下一步：

1. 识别 privileged objective 中真正有效的监督；
2. 训练 learned recovery critic；
3. 比较 learned guidance 与 privileged guidance；
4. 计算：

\[
\rho_{guided}
=
\frac{
S_{learned}-S_{natural}
}{
S_{privileged}-S_{natural}
}
\]

建议最低：

\[
\rho_{guided}\geq 0.3
\]

5. 做 task-held-out 和 perturbation-held-out 验证；
6. 检查 clean-success harm、calibration 和 uncertainty gating。

## 9.2 Gate B NOGO

```yaml
frozen_same_policy_guidance: nogo
candidate_critic_for_guidance: closed
world_model_gate: closed
```

进入：

- recovery LoRA / adapter；
- OFT-to-SmolVLA recovery distillation；
- state-transition distillation；
- abstention / request help。

不得通过修改状态集合、提高 guidance strength 或调整阈值来挽救失败结果。

---

# 10. 统计要求

同一 trajectory 内的多个 stage state 不独立，主置信区间必须使用：

```text
trajectory / episode cluster bootstrap
```

必须报告：

- snapshot-level success；
- trajectory-cluster CI；
- task-level direction；
- suite-level direction；
- perturbation-level direction；
- leave-one-task-out；
- leave-one-trajectory-out；
- clean-success harm；
- irreversible harm；
- invalid action rate；
- latency；
- compute；
- intervention frequency。

主指标应在 protocol lock 中预先固定，建议为：

```text
terminal success
clean-success preservation
irreversible harm
compute-normalized success gain
```

---

# 11. 推荐输出文件

## Collection 与 QC

```text
progress/2026-08-04_pre_c0_collection_integrity.md
progress/2026-08-04_pre_c0_stage_key_qc.md
artifacts/pre_c0/collection_integrity.json
artifacts/pre_c0/stage_keys.json
```

## 冻结协议

```text
artifacts/pre_c0/pre_c0_48_state_manifest.json
artifacts/pre_c0/pre_c0_protocol_lock.yaml
```

## Gate A

```text
progress/2026-08-04_pre_c0_gate_a_smoke.md
progress/2026-08-04_pre_c0_gate_a_results.md
artifacts/pre_c0/gate_a_results.json
artifacts/pre_c0/gate_a_rollouts.parquet
```

## Gate B，仅 Gate A FAIL 时

```text
progress/2026-08-04_pre_c0_gate_b_results.md
artifacts/pre_c0/gate_b_results.json
artifacts/pre_c0/guidance_trust_region_audit.json
artifacts/pre_c0/gate_b_rollouts.parquet
```

---

# 12. 执行清单

## 立即执行

- [ ] 确认 `PRE_C0_COLLECT_EXIT`
- [ ] 生成 collection integrity report
- [ ] 排查失败、超时、缺失 snapshot 和重复 seed
- [ ] 完成 deviation mining
- [ ] 完成 T0–T4 stage-key QC
- [ ] 冻结 inclusion / exclusion rules
- [ ] 冻结 48-state manifest
- [ ] 冻结 Gate A 参数与统计协议

## Gate A

- [ ] 运行固定状态 smoke test
- [ ] 验证 strict resample contract
- [ ] 验证 fresh replan contract
- [ ] 验证 continuation parity
- [ ] 运行完整 48-state natural audit
- [ ] 计算 sampling、reconditioning、closed-loop headroom
- [ ] 计算 best fixed horizon 与 adaptive horizon oracle
- [ ] 运行 trajectory-cluster bootstrap
- [ ] 判定 PASS / INCONCLUSIVE / FAIL

## Gate B，仅 Gate A FAIL

- [ ] 运行 matched-compute Best-of-K
- [ ] 运行 numerical trust-region probe
- [ ] 运行真实 closed-loop privileged refinement
- [ ] 检查 action smoothness、workspace 和 harm
- [ ] 计算相对 matched-compute baseline 的独立增量
- [ ] 判定 PASS / NOGO

## 保持封闭

- [ ] 不解封 PRE-A3 hidden test24
- [ ] 不重开 finite handback method gate
- [ ] 不重开 termination selector gate
- [ ] 不重开 world model gate
- [ ] 不把 CPU refinement 描述为 SmolVLA API guidance

---

# 13. 最终决策树

```text
Natural Gate A
│
├── PASS
│   ├── 确认收益来源
│   ├── candidate critic eligible
│   └── privileged guidance 非主线必需
│
├── INCONCLUSIVE
│   ├── 扩大 natural audit
│   ├── 检查 cluster CI 与单 task 驱动
│   └── 暂不训练复杂模型
│
└── FAIL
    │
    └── Privileged Gate B
        │
        ├── PASS
        │   ├── learned recovery critic eligible
        │   ├── learned guidance audit
        │   └── held-out confirmation
        │
        └── NOGO
            ├── 关闭 frozen same-policy guidance
            ├── recovery adapter / LoRA
            ├── OFT-to-SmolVLA distillation
            └── abstention / human escalation
```

---

# 14. 当前最重要的问题

当前 guidance package 已经具备运行、审计和输出 NOGO 结论的能力，但仍然只是备用分支。

下一步首先要回答：

\[
\text{自然同策略候选空间是否存在真实、跨任务、低误伤的 headroom？}
\]

在该问题得到答案之前：

- 不训练 candidate critic；
- 不运行 world model；
- 不把 privileged refinement 提升为主方法；
- 不改变已经封闭的 PRE-A3 结论。
