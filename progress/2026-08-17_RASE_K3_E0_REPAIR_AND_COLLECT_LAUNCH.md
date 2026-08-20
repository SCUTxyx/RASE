# RASE K3-E0 修复与 K3 正式收集启动记录(2026-08-17)

## 1. E0 修复(完成,`E0_CAPTURE_PASS`)

### 代码级根因(与用户修正一致)
- 旧 `_peek_native_chunk` 从 LeRobot action queue 反推 chunk(违反 K3 协议);
- `prefix_to_decision` 只在 boundary 存 1 步 action + hash;
- pi0-fast 的 `select_action` 消费 `_action_queue`,LeRobot 的 `_queues` 恒空,
  导致 `_action_queue_empty()` 误判 → capture 时无新推理 → `prefix_available=False`。

### 实现(已上传,测试 493 passed)
- `rase/collect/policy_step.py`:
  - `InferenceEvent`(immutable):inference_event_id / native_chunk / env_chunk /
    chunk_size / candidate_generation_seed / boundary_step / policy_state_hash;
  - `capture_inference_event()`:monkey-patch `predict_action_chunk` 在
    **推理发生时**冻结原生输出(绝不从 queue 反推);
  - `policy_state_snapshot/restore/fingerprint`:queue(`_queues`+`_action_queue`)
    与 torch/numpy RNG 的快照与恢复(requery 隔离的基础);
  - `clear_policy_queues()`:清空两种 queue 容器。
- `rase/collect/forked_rollout.py`:
  - `InProcessLeRobotContinuation` 增加 event 记录 + cursor 维护
    (`consumed_in_current_event`),`_action_queue_empty()` 优先检查
    `_action_queue`(pi0-fast)。
- `scripts/collect_rase_vnext_discovery.py`:
  - continue.source 读 boundary inference event,chunk =
    `env_chunk[queue_cursor_at_boundary:]`(queue 耗尽时隔离预推理);
  - requery.source 在 **独立 policy 状态**(snapshot → clear queue → 强制推理 →
    rollout → restore)下执行,与 continue 严格隔离;
  - resample 双候选(6-operator 格式),capability **数据驱动**:两候选首动作
    bitwise 相同 → `incapable_missing`(非普通失败);
  - fallback = 实际执行轨迹;abort = control-only;
  - 全部 row 带 `capability_status` / `chunk_origin` / `execution_status`。
- `rase/vnext/candidate_capture.py`:schema v2(per-operator capability、origin、
  event_id、cursor、native chunk hash、full env chunk、boundary action)。

### E0 smoke v3 结果(`runs/rase_vnext/k3_e0_native_capture_smoke_v3/E0_VERDICT.json`)
- `prefix_available=True`;capture audit PASS(alignment 全 True);
- continue=executable(cursor=9, event 关联)、requery=executable(cursor=0)、
  fallback=executable、abort=control_only_abort、resample=incapable_missing
  (π0-fast 无 native diversity,确定性 record)。

## 2. K3 正式 cohort 冻结(`runs/rase_vnext/frozen/k3_cohort_manifest_v1.json`)
- sha256 `6971fc3d4461fb2061e09786249125785e54ad8d5af514bd28e1f67966150b4c`;
- 8 tasks × 3 roots(每 suite 2 task),metadata-only hash 排序选择;
- 排除 confirmation/B2/D0/E0 roots;近重复阈值 0.005(同 task qpos L∞,
  跨 task 不视为重复采样);无替换发生;
- 4 folds × 2 tasks(冻结);432 operator slots / 360 simulator executions;
- 6 operators(continue/requery/resample.candidate.0/.1/fallback/abort)× 3 repeats;
- 选择与替换规则写入 manifest(`selection_rule`),未读任何 outcome。

## 3. K3 收集(进行中)
- runner:`scripts/run_rase_vnext_k3_collect.sh`,tmux `k3collect`;
- 输出:`runs/rase_vnext/k3_collect_v1/`(按 suite 切换 OFT oracle server);
- 每 root/operator/repeat 保存 success/progress/steps/latency/capability/
  MotionTrace/capture(npz+json v2)。

## 4. 待办
- [ ] K3 收集完成(72 groups)
- [ ] `scripts/analyze_rase_vnext_k3.py`:capture/capability audit → 特征 ladder
      (state-only/raw-action/trace-only/trace+semantic)→ task-held-out ridge OOF
      → same-root pairwise ranking → action-swap → risk-coverage/Brier/ECE →
      offline utility → Gate 判定(SIGNAL/UTILITY)
- [ ] 一页 Gate decision 写入 progress
