# RASE 下一步行动规划(基于 2026-08-17 代码库/实验分析)

> 分析时间:2026-08-17 18:50 CST | 服务器:AutoDL bjb2:25659(autodl-container-9e6e4e8364-93f7f0be)
> 状态:GPU 空闲、磁盘 64G 可用、r10b diagnostic 已闭合、E0 smoke v1/v2 已跑(prefix_available=False)

## 1. 当前科学状态(冻结口径)

| Gate | 状态 | 关键证据 |
|---|---|---|
| A | A-PARTIAL | 4800/4800 rows;π0-fast gap 0.0415 过门;π0.5 gap 0.0099 不过 |
| B(历史 Phase-C) | B-FAIL | 288 groups 中 libero_10_001082 requery 5 次重放不复现 |
| B2(新同步捕获) | B2_CAPTURE_PASS | 4/4 groups、12 chunks、raw→canonical→raw 误差 0、MotionTrace 12/12 |
| C(48-task pilot) | PILOT_SIGNAL_WEAK | pairwise 0.475 vs prior 0.400(+0.046),3/5 seeds;239/287 ties;informative tasks 仅 13 |
| D0 | D0_FEASIBILITY_PASS | 20 rollouts、6 success;Goal/Spatial 有 within-root diversity;Object/Long 全 F |
| K3-E0 | FAIL | prefix_available=False;π0-fast 在 boundary 消费旧 queue,无新 predict_action_chunk |

## 2. 已闭合的根因诊断(R10B)

`runs/pre_c0_r10/r10b_chunk_input_divergence_audit_v1.json`:
- 18 groups 矩阵审计;34 单元 `B_CLOSED_LOOP_INPUT_DIVERGENCE`,2 单元 `D_NO_REPRODUCED_CHUNK_DIVERGENCE`;
- decision:`FREEZE_CLOSED_LOOP_AMPLIFICATION_EVIDENCE`;
- 机制:boundary 首次 query 输入完全一致(first_query_input_exact=true),但 8 步后
  agentview/wrist 图像哈希发散 → 闭环放大,不是 t=0 特征损坏;
- 结论:同 root 的 outcome flip 是闭环执行发散,与 K3 的"概率化标签 + K=3 重复 + capability mask"设计相互印证;
  r10b diagnostic 不阻塞 K3,但**不支持任何确定性小-K 硬标签**。

## 3. E0 失败的代码级根因(已定位)

关键文件(工作目录 `/root/autodl-tmp/RASE`):

1. `scripts/collect_rase_vnext_discovery.py`
   - `_peek_native_chunk()`(L61-93):**从 policy 的 action queue 反推**完整 chunk
     (`while len(actions) < horizon and not policy._action_queue_empty()`),最后恢复 queue。
     违反 K3 协议"禁止从 queue 反推";queue 不足时无法凑满 H=10。
   - `prefix_to_decision()`(L94-155):执行 source prefix 到 decision_step,只保存 1 步
     boundary_action + hash,不保存原生 chunk。
   - `collect_group()`(L253+):continue.source 走 `_peek_native_chunk`;requery.source
     reset 后重调 `_policy_action`。
2. `rase/collect/policy_step.py`
   - `select_env_action_with_native_chunk()`(L70-123):**已写好但未接入**的正确方法——
     monkey-patch `predict_action_chunk` 拦截原生输出,校验 executed first action == chunk[0];
     但 LeRobot policy 在 action queue 非空时 `select_action` **不调用** `predict_action_chunk`,
     此时 `native is None` → RuntimeError → `prefix_available=False`(E0 smoke v2 的直接失败点)。
3. `rase/vnext/candidate_capture.py`:捕获文件写入/审计工具(write_candidate_capture、
   audit_candidate_capture、pad_action_chunk),本身无问题。

**失败机制一句话**:π0-fast 的 queue 消费语义导致"boundary 时刻没有新推理发生",
而旧的 capture 逻辑依赖 queue 反推,新的 capture 逻辑依赖新推理,两者都无法拿到
boundary 处的完整原生 chunk。

## 4. E0 最小修复方案(下一阶段唯一阻塞项)

### 4.1 设计:inference-time 持久化 + 显式 requery

```text
每次 predict_action_chunk 被调用时(统一 wrapper):
    → 立即冻结 {inference_event_id, native_chunk[1,T,D], seed_ledger, timestamp}
    → 持久化到 policy_bundle["last_inference"]

continue.source  = policy_bundle["last_inference"].native_chunk(当前 queue 的来源,已带生成时关联)
requery.source   = 清空 queue + 强制一次新推理 → 捕获新原生 chunk(独立 operator seed)
resample.source  = 单次推理多采样;π0-fast 无 diversity → 保留 capability mask(协议已有)
fallback.persistent = 实际执行的 fallback 完整动作轨迹(Phase-C 教训:fallback 170 wins,必须存完整 chunk)
abort.safe       = control-event record,不伪装成零动作 chunk
```

### 4.2 改动点(最小集)

| 文件 | 改动 |
|---|---|
| `rase/collect/policy_step.py` | ①把 wrapper 改为"每次调用都持久化 last_inference";②新增 `force_requery()`(清 queue + 强制推理 + 捕获);③`select_env_action_with_native_chunk` 复用持久化路径 |
| `scripts/collect_rase_vnext_discovery.py` | ①`_peek_native_chunk` 替换为读取 `last_inference`(continue);②requery 分支改用 `force_requery`;③`prefix_to_decision` 的 boundary 调用走统一 capture;④删除 queue 反推逻辑 |
| `tests/test_vnext_candidate_capture.py` | 新增:queue 非空时 continue 捕获完整 H=10;6 operator schema/horizon/dtype/mask;raw→canonical→raw 误差 0;冻结 seed 重复 hash 一致;incapable 标记;capture parity 失败时 collector 非零退出且不产出正式 manifest |

### 4.3 验证路径(按序)

1. 单元测试 + ruff(CPU,无 GPU);
2. 单 root E0 smoke(v3,不入正式 cohort):6-slot capture + capability,要求
   `prefix_available=True` 且 6 个 operator 的 chunk 完整;
3. 通过后进入 K3-Freeze。

## 5. K3 正式实验(修复后,严格按 08-17 revised 协议)

### 阶段 2:K3-Freeze(0.5-1 天)
- 8 tasks × 3 roots(每 suite 2 task: Spatial/Object/Goal/Long);
- 从 `runs/rase_vnext/frozen/root_catalog_v1.json`(192 roots / 48 tasks,
  r7a_pi0fast_reset_pool_v1)metadata-only 选择;排除 B2/D0 的 4 个 root;
- 近重复规则在 outcome 前冻结:qpos/qvel L∞ 阈值 + 图像 perceptual-hash 阈值;
- 冻结:PROTOCOL.json、seed ledger、4 folds × 2 tasks(同 task 3 roots 同 fold、
  同 root 所有 operator 同 fold)、operator schema、capability contract、exclusion 报告;
- 输出 frozen_manifest.json + sha256;freezer 不得读取任何 outcome。

### 阶段 3:K3-Collect(1-2 天 GPU)
- 24 roots × 6 operators × K3 = **432 operator slots**;abort.safe 若 control-only
  → **360 simulator rollouts**(72 条 control-only records 仍完整);
- 每 root/operator/repeat 保存:success、progress_delta、distance_to_target_curve、
  collision/out_of_bounds、gripper/contact phase、recovery_time、query_count、
  fallback_steps、latency samples、termination_reason、MotionTrace、capability mask;
- 任何指标不可用写 `metric_mask` + 原因;incapable 不等价于失败,不得从分母剔除;
- capture / execution / metric 三层分开审计;系统性 source/requery/resample 缺块
  → 停止统计并回到 E0。

### 阶段 4:K3-Analyze(1-2 天)
- 特征 ladder(仅四组预注册输入):state-only / raw-action / trace-only / trace+semantic;
- 模型:linear/ridge,超参只在 inner fold 选择;4 folds task-held-out;
- 主指标:same-root pairwise ranking(先 root 内聚合 K3)、action-swap sensitivity、
  risk-coverage、Brier/ECE、progress prediction;
- 基线:identity/continue、always fallback、frozen risk-threshold selector、oracle(上界);
- 全部指标按 overall/suite/task/operator/capability 分层 + task-level bootstrap。

### 阶段 5:Gate 决策与分支
- **K3-SIGNAL-PASS**:raw-action 或 trace+semantic 相对 state-only 的 same-root ranking
  增益 ≥0.03;≥3/4 folds 同向;swap 不退化;ΔBrier ≤0.02、ΔECE ≤0.05;
- **K3-UTILITY-PASS**:paired improvement ≥0.05 或 oracle regret ↓≥10%;成本增 ≤10%;
  risk-coverage 不恶化;≥3/4 folds 同向;
- 双 PASS → 小规模 semantic selector 设计(E0→E4 ladder:prior → state/history →
  raw action stats → CanonicalMotionTrace → trace+frozen visual history);
- PILOT_SIGNAL_WEAK → 保留弱信号;预注册 K5 扩大决策树(3/4 folds 同向但幅度不足 → K5
  48-task × K3;方向不一 → 停止);
- 任一 FAIL → 停止,回到 E0 或关闭 learned selector 主线。

## 6. 并行/可选工作

1. **π0.5 独立 challenge**:outcome-independent 8-task × K3 opportunity smoke,
   不能把 π0-fast 的 D0 结果迁移成 pooled 证据;
2. **fallback 完整动作同步保存**(已在 4.1 纳入 E0 修复,不必另立任务);
3. 数据盘 64G 够用,但每次大规模收集前保留 20 GiB preflight(已有惯例)。

## 7. 明确不启动(直到 K3 双 gate 通过)

multi-VLA pooled selector / OPD 训练 / RL / world model / 大规模公开数据下载 /
实时 closed-loop deployment。

## 8. 建议执行顺序与预估

| 序号 | 工作 | 预估 |
|---|---|---|
| 0 | 归档 r10b diagnostic 产物 + 本规划写入 progress/ | 0.5 天 |
| 1 | E0 修复(4.1/4.2)+ 单测 + E0 smoke v3 | 1-2 天 |
| 2 | K3-Freeze(冻结 cohort/协议/folds) | 0.5-1 天 |
| 3 | K3-Collect(432 slots / 360 rollouts) | 1-2 天 GPU |
| 4 | K3-Analyze(特征/OOF/swap/bootstrap/utility) | 1-2 天 |
| 5 | Gate 决策 + 分支(写 progress 记录) | 0.5 天 |

合计约 1 周;其中 E0 修复是纯代码工作,可立即开始(GPU 空闲不冲突)。
