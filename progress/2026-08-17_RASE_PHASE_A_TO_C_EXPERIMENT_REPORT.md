# RASE Phase A→C 实验报告（2026-08-17）

> 状态：FINAL；全 48-task × K3 prefix-only pilot 已完成并冻结。

## 1. Phase A：confirmation 结论

- 采集完整：4800/4800 rows；4320 available，480 contract-masked；所有完整性检查通过。
- formal opportunity：FAIL；non-abort opportunity：FAIL。
- pooled oracle-minus-best-fixed：0.0470955；task-bootstrap 95% CI `[-0.0351537, 0.0973778]`。
- π0.5：0.0099077，未过单策略实用效应门。
- π0-fast：0.0414635，通过单策略效应门。
- 严格 verdict：`A_PARTIAL / SINGLE_POLICY_PILOT_AND_INDEPENDENT_CHALLENGE_COHORT`。
- 只解锁：明确标注的 `labeled_single_policy_pilot`。
- 继续锁定：pooled universal selector、semantic pretraining、world model、RL、OPD、closed-loop claim。

## 2. Phase B：adapter/parity 修正

核实 robosuite `OSC_POSE` 后发现旧 scaffold 的动作物理语义错误，已修正：

- translation：normalized action × 0.05 m；base frame；
- rotation：normalized action × 0.5 rad；scaled axis-angle / rotation vector；
- orientation composition：`R_goal = R_delta @ R_current`；
- gripper：`-1=open, +1=closed`；
- LeRobot nested `robot_state` 显式转换为 `[eef_pos, eef_axisangle, gripper_qpos]`；
- raw→canonical→raw 误差为 0；MotionTrace 94/94 转换通过；
- π0-fast `resample.source` 按 frozen capability audit mask，原因是无 native candidate diversity。

测试：服务器全套 `tests/test_vnext_*.py` 为 47 passed（包含 practical-tie 回归测试）。

8-task × K3 smoke 中 48 group 扫描完成，47 group 精确动作哈希对齐；1 个
`libero_goal_000007 / source.step.8 / replica0` 连续 5 次未复现。因此当前 Phase B verdict 是
`B_FAIL_REPRODUCIBILITY`，不能宣称正式 B-PASS。

## 3. 8-task source-action smoke

该 cohort 按 frozen manifest 每套件字典序选择两个 task，不读取 outcome。结果：

- 47 个可对齐 group 中，continue 在 utility 上 47/47 高于 requery；
- 实用 tie margin 0.01 后：44 ties、3 continue wins、0 requery wins；
- 因此 verdict 应为 `PILOT_SUPPORT_FAIL`，而不是把负结果解释为“动作语义假设已被否定”；
- operator prior 的 1.0 pairwise accuracy 来自单一标签，不能作为 selector 成功；
- MotionTrace 相对 prior 的表面负增益没有可识别的双向 preference 支持，不能作为表征能力的决定性反证。

## 4. 全量 π0-fast operator atlas（480 groups）

tie margin 0.01：

### 4.1 continue vs requery

- practical ties：400；
- continue wins：40；
- requery wins：40；
- source-only oracle-minus-best-fixed：0.0833267；
- support gate：PASS。

### 4.2 continue / requery / fallback

- unique wins：continue 40、requery 40、fallback 170；
- fallback vs continue：205 wins / 275 losses；
- oracle-minus-best-fixed：0.0450793；
- support gate：PASS。

这说明 A_PARTIAL 的可学习结构真实存在，但 8-task smoke 没有覆盖；同时 fallback 是主要 operator，
未来新 collection 必须在 outcome rollout 同时保存 fallback action chunk，不能再只存 hash。

## 5. 全 48-task × K3 source-action pilot

范围为 48 tasks × 2 decision points × K3 = 288 groups；只重放 source prefix，不重跑
post-decision outcome；只保存与原 confirmation 首动作 SHA256 精确匹配的 continue/requery 特征。

### 5.1 Collection / parity

- 288/288 group 均有终态元数据；287 `COMPLETE`，1 `UNREPRODUCIBLE`，0 missing；
- collection status：`PARTIAL_REPRODUCIBLE`；
- Phase B：`B_FAIL_REPRODUCIBILITY`；
- 唯一失败：`libero_10_001082 / source.step.16 / replica0`；
- 该组 source boundary 与 continue 首动作 hash 一致，requery 首动作在 5 次 bounded replay 后仍不一致；
- 其余 574 个 action chunks 的 raw→canonical→raw 和 MotionTrace 转换全部通过。

### 5.2 实用 support

使用预注册 practical tie margin 0.01：

- 239/287 groups 为 practical ties；
- 48 informative groups，来自 13/48 tasks；
- continue unique wins 24，requery unique wins 24；support gate PASS；
- suite 分布：Goal 24、Spatial 12、Object 9、Long 3；
- winner 在 task 内高度成簇，说明有效独立监督单位接近 13 个 task，而不是 48 个 group。

### 5.3 5-seed task-held-out OOF

Primary 为 `proprio + CanonicalMotionTrace`：

- practical pairwise accuracy：0.475；
- operator prior：0.400；
- task-bootstrap mean gain：+0.04615；
- task-bootstrap 95% CI：`[-0.13846, 0.20000]`；
- 3/5 seeds 方向为正（门槛 4/5）；
- trace minus shuffled-trace：+0.04167（通过 +0.02 诊断门）；
- mean oracle regret：0.11344；operator prior 为 0.11737；
- raw action 为 0.400，motion trace 为 0.4375，proprio+trace 为 0.475。

正式 pilot verdict：`PILOT_SIGNAL_WEAK`，不是 PASS。

### 5.4 评估实现审计

第一次分析错误地让 primary pairwise 使用默认 tie margin 0，而 support 使用 0.01，导致 239 个
实用等价组的微小 query/latency cost 被当成标签，operator prior 虚高到 0.845。已修正为两个 gate
统一使用预注册的 0.01；旧报告保留为 `analysis_pre_practical_tie_fix.json`，修正后报告为
`analysis.json`。这不是事后调阈值，而是修复同一冻结阈值在两个指标间的不一致。

### 5.5 Direct paired-target 诊断（exploratory，不替代 primary）

为区分“动作无信号”和“逐候选 utility 回归目标不匹配”，另跑了直接 same-root winner ranking：

- `state × MotionTrace delta`：0.525；
- 最佳 state-only control：0.5125；
- task-bootstrap gain：+0.07692，95% CI `[-0.24615, 0.38462]`；
- 3/5 seeds 同向；
- true trace 相对 shuffled trace：+0.1500。

该结果强化“存在动作信息但有效 task 太少”的诊断，但准确率接近随机且 CI 很宽，只能标记为
`EXPLORATORY_NOT_A_GATE`，不能覆盖 B-FAIL，也不能解锁 Phase D。

## 6. 当前科学判断

1. 核心 RASE idea 未被否定：全量 atlas 有明确的状态依赖 operator opportunity。
2. `动作单独可泛化判断 operator` 尚未成立；弱正信号和 shuffled control 表明动作有信息，但
   13 个 informative tasks 不足以确认，而且没有 task/vision 语义时无法判断“这段运动是否朝正确目标”。
3. 当前 pooled multi-VLA 假设尚未获得支持：π0.5 opportunity 不足，pooled CI 跨 0。
4. 当前最强瓶颈不是“大模型不够”，而是：
   - operator capability 不对称；
   - fallback 动作未同步保存；
   - requery candidate 事后重推不能保证动作级复现；
   - 239/287 组没有实际差异，有效监督仅集中在 13 个 task；
   - 当前模型缺少显式 task-action / state-action alignment。
5. OPD、world model 和 RL 现在都不会修复上述数据可识别性问题，继续保持 gated/locked。

## 7. 下一阶段：C2 observability repair（不改变核心 idea）

按以下顺序执行，不越级：

1. **冻结本轮 artifact。** 当前结果只标记为 π0-fast 单策略 development pilot。
2. **修复采集 contract。** 在 branch rollout 当时同步保存 continue、requery、fallback 的完整 action
   chunk、candidate/RNG ID、task instruction、双视角 observation、proprio 和 adapter version；不再依赖
   事后重推 candidate。
3. **建立 outcome-independent challenge cohort。** 使用新 roots/新 seeds，覆盖全部 48 tasks；选择规则只
   看 suite/task/decision 元数据，不看当前 outcome。每个 operator 固定 K，保留概率/count 标签。
4. **把 fallback 纳入。** 全量 atlas 中 fallback unique wins=170，是主要纠正臂；如果不保存其动作，
   无法训练真正 selector。
5. **直接训练 same-root paired target。** 预测 `ΔU(continue,requery,fallback)`；0.01 内作为 tie/abstain，
   不让微小 query cost 支配分类。
6. **低容量 observability ladder。** no-action → task/vision only → raw action → MotionTrace →
   `task × state × MotionTrace` interaction → shuffled/action-swap。先用冻结视觉特征或小编码器，不下载
   大规模公开数据，不训练 world model。
7. **C2 gate。** 必须同时满足：B contract PASS；至少 32 informative tasks；两个以上 operator 有支持；
   task-bootstrap gain lower bound > 0；至少 4/5 seeds 同向；trace 优于 shuffled ≥0.02。
8. **再做 π0.5。** 先单独重新验证 operator opportunity/capability；未通过前不能形成 pooled multi-VLA
   或 leave-one-VLA-out 主张。
9. **只有 C2-PASS 后**才进入 action-semantic pretraining；OPD、world model、RL 继续 locked。

## 8. 代码与 artifact

服务器：`/root/autodl-tmp/RASE`

- `rase/vnext/motion_trace.py`
- `rase/vnext/libero.py`
- `rase/vnext/phase_c_pilot.py`
- `scripts/export_rase_vnext_phase_c_features.py`
- `scripts/analyze_rase_vnext_phase_c_pilot.py`
- `scripts/analyze_rase_vnext_phase_c_paired_diagnostic.py`
- `scripts/analyze_rase_vnext_policy_operator_atlas.py`
- `scripts/run_rase_vnext_phase_c_pi0fast_source_full.sh`
- `runs/rase_vnext/phase_a_v1/phase_a_audit.json`
- `runs/rase_vnext/phase_c_pi0fast_source_smoke_v2/`
- `runs/rase_vnext/phase_c_pi0fast_source_full_v1/`

本地审计副本：

- `work/phase_c_full_results/collection_report.json`
- `work/phase_c_full_results/analysis.json`
- `work/phase_c_full_results/analysis_pre_practical_tie_fix.json`
- `work/phase_c_full_results/EXPORT_CONTRACT.json`
- `work/phase_c_full_results/paired_diagnostic.json`
