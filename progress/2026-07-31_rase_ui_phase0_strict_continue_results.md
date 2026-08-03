# RASE-UI Phase-0 结果与顶会级后续路线（2026-07-31）

## 1. 结论先行

原 W9C 的 `continue_smol` 实际会重置 SmolVLA，只能解释为 `REPLAN`。本轮已经补齐真正的 strict `CONTINUE`：在 action chunk 内部保存尚未执行的 env-space action suffix，恢复后先执行该 suffix，再让源策略继续生成下一 chunk。

真实 smoke 显示 `CONTINUE` 与 `REPLAN` 确实不是同一个干预：

- 8 个同状态配对：`CONTINUE` 6/8，`REPLAN` 5/8；
- 5 对均成功，2 对均失败，1 对仅 `CONTINUE` 成功，0 对仅 `REPLAN` 成功；
- McNemar 双侧精确检验 `p=1.0`，样本量不足，不能宣称总体优越；
- 唯一的 continue-sensitive 状态追加 5 个 paired seeds 后，`CONTINUE` 5/5 成功，平均 41.4 步；`REPLAN` 0/5 成功，全部跑满 225 步；双侧精确 `p=0.0625`；
- 8 个保存 suffix 的 source-rollout parity 最大误差全部为 `0.0`。

这证明了“干预语义与状态依赖差异存在”，但还没有证明“可学习的多算子分配问题成立”。当前最正确的决策是扩展算子与任务覆盖，不训练世界模型。

## 2. 已完成代码

服务器仓库：`/root/autodl-tmp/RASE`

### 2.1 Decision-context v2

- `rase/interventions/decision_context.py`
  - schema：`rase-decision-context/v2`；
  - 保存 source policy、真实 env step、chunk size/offset；
  - 保存 env-space active suffix 与 SHA-256；
  - 保存最近 public RGB/proprio/action history；
  - 只有 source 轨迹逐动作 parity 通过后，状态才支持 strict `CONTINUE`。

- `rase/collect/lerobot_libero_plus_adapter.py`
  - 支持 `capture_decision_context_v2`；
  - 默认在 chunk 内 `offset=5/10` 截图，而非 chunk 边界；
  - 从 `policy._queues[ACTION]` 复制剩余动作，经 policy postprocessor 和 env postprocessor 转换到 LIBERO env action space；
  - 在线对照源轨迹随后真正执行的动作，parity 不完整或不通过的快照不会发布。

### 2.2 实验入口

- `scripts/export_decision_context_keys.py`
  - 只导出 parity 通过的 strict-CONTINUE 状态；
  - 支持 task round-robin 限额；
  - 支持 `--include-key` 做预注册的定点重复。

- `scripts/rollout_smol_interventions.py`
  - 同一状态、同一 continuation seed 下配对运行：
    - `CONTINUE(smol active suffix)`；
    - `REPLAN(smol current observation)`；
  - 结果写入统一 `operators.json / snapshots.jsonl / outcomes.jsonl`；
  - crash-safe scheduler、fresh/resume manifest、checkpoint hash；
  - 输出成功率、env steps、latency、paired disagreement、McNemar exact p-value。

- `configs/collect_rase_ui_phase0_smoke.json`
  - 两任务语义 smoke 配置；正式论文实验不可直接用该规模。

### 2.3 测试

- 新增 `tests/test_decision_context.py`；
- 新增 `tests/test_smol_intervention_runner.py`；
- 关键兼容集：24 项通过；此前统一干预与 selector 兼容集 37 项通过；
- Ruff 全通过。

没有提交 commit，因为服务器 worktree 在本轮开始前已经包含大量用户修改和未跟踪研究产物。

## 3. 本轮实验与可复现位置

### 3.1 8-state smoke

- pool：`runs/rase_ui_phase0_smoke_parity_pool`
- frozen keys：`runs/rase_ui_phase0_smoke_parity_keys.json`
- paired results：`runs/rase_ui_phase0_smoke_parity_paired`
- summary：`runs/rase_ui_phase0_smoke_parity_paired/summary.json`
- log：`runs/rase_ui_phase0_smoke_parity_tmux1.log`

逐状态结果：

| task | chunk | CONTINUE | C steps | REPLAN | R steps |
|---|---:|---:|---:|---:|---:|
| object-10 | 0 | 1 | 128 | 1 | 123 |
| spatial-4 | 0 | 1 | 80 | 1 | 81 |
| object-10 | 2 | 1 | 107 | 1 | 103 |
| spatial-4 | 2 | 1 | 58 | 1 | 57 |
| object-10 | 4 | 1 | 90 | 1 | 84 |
| spatial-4 | 4 | 1 | 42 | 0 | 225 |
| object-10 | 6 | 0 | 205 | 0 | 205 |
| spatial-4 | 6 | 0 | 205 | 0 | 205 |

在五个“双成功”状态中，`CONTINUE - REPLAN` 的步数差为 `+5,-1,+4,+1,+6`，平均 `+3`：普通早期状态上 REPLAN 略快；但 spatial-4/chunk-4 上，丢弃 active suffix 会从快速成功变成彻底失败。这正是状态依赖分配信号，而不是“某个算子全局更强”。

### 3.2 continue-sensitive 定点重复

- state：`sp1_7d71b4cc40a489c2ff7ab733f6e015e6`
- task：`libero_spatial_000004`
- chunk：4（实际 decision env step 为 45，chunk offset 5）
- keys：`runs/rase_ui_phase0_target_continue_only_key.json`
- results：`runs/rase_ui_phase0_target_continue_only_seed5`
- summary：`runs/rase_ui_phase0_target_continue_only_seed5/summary.json`

五个 paired seeds：

- CONTINUE：5/5，env steps `42,41,40,41,43`；
- REPLAN：0/5，env steps 全部 `225`；
- 平均节省 183.6 个剩余环境步；
- paired exact p=`0.0625`。再增加一个独立 seed 且方向不变时，双侧精确 p 将降到 `0.03125`，但单状态显著不等于跨任务结论。

### 3.3 非权威的旧 smoke

第一次在加入在线 source parity 前生成的 `runs/rase_ui_phase0_smoke_pool` 及未完成的 `runs/rase_ui_phase0_smoke_paired` 只用于调试，不得进入论文统计。所有正式分析必须使用带 `parity` 的路径。

## 4. 顶会论文应该如何收敛

建议论文主线：

> **When Should a Robot Continue, Replan, or Switch? A Causal Same-State Intervention Benchmark for VLA Recovery**

三项贡献必须彼此独立成立：

1. **Benchmark**：统一、可部署的干预算子集合；同一恢复状态下分叉；严格禁止 privileged simulator state 进入 selector；按 task/episode 防泄漏划分。
2. **Causal evaluation**：同状态、paired seed、真实 suffix provenance；报告 best fixed、matched random、same-state oracle、harm/futility、cost-sensitive Pareto frontier。
3. **Method**：先用 history-only operator-value baseline；只有它稳定超过 matched random 和 best fixed 后，才增加 operator-conditioned short-horizon world model。

现实机器人只能作为最后的外部有效性实验，不应在 simulator benchmark 和方法都未过 gate 时提前投入。

## 5. 下一阶段按顺序执行

### Gate A：补齐最小算子矩阵

先实现并验证：

1. `CONTINUE(smol active suffix)`：已完成；
2. `REPLAN(smol current observation)`：已完成；
3. `SWITCH_POLICY(OFT current observation + public handoff)`：下一项；
4. `ABSTAIN(safe stop)`：实现真实终止/请求帮助语义，不能只写一个离线常数；
5. `LOCAL_CORRECT(retreat_realign_v1)`：通过几何无关、有限步、失败安全的 feasibility 合同后再加入；
6. `REWIND`：最后实现，必须是物理逆向控制，不能用 simulator teleport 冒充。

只有前三个执行算子加 ABSTAIN 覆盖完整，才运行正式 opportunity pilot。

### Gate B：正式 opportunity pilot

建议最小规模：

- 10 个 frozen clean tasks，四个 LIBERO suites 均覆盖；
- 再加入 camera/robot/object/spatial failure frontier，避免 clean early-state ceiling；
- 每 task 至少 5 个状态，按 early/mid/late 与 success/failure frontier 分层；
- 每状态每算子 3 个 continuation seeds；
- 4 个首批 operators，共至少 `10 × 5 × 3 × 4 = 600` 个 arm outcomes；
- task-level cluster bootstrap 95% CI；
- primary test 为 paired operator allocation，不按单 arm 独立样本计算显著性。

预注册 Go 条件：

- complete snapshots ≥ 50；
- same-state oracle − best fixed ≥ 5 percentage points（或预注册 utility 0.05）；
- 至少 3 个算子在至少 2 个不同任务上成为严格 winner；
- 存在可测量 harm 与 futility；
- strict CONTINUE parity 通过率 100%；
- 不通过则收缩论文为 benchmark/negative result，不训练 world model。

### Gate C：history-only baseline

输入仅允许：最近 public RGB、proprio、已执行动作、instruction、operator profile。禁止：sim state、未来帧、oracle outcome、candidate rollout video。

先做：

- linear/ridge per-operator value；
- small MLP；
- calibrated success probability + expected cost；
- task-disjoint、episode-disjoint split；
- matched random、best fixed、oracle、cost-blind classifier 对照；
- Brier/ECE、selective risk、bootstrap regret。

Gate：对 best fixed 的 improvement 95% CI 下界 > 0，且跨至少三个 suite 方向一致。否则停止 method branch。

### Gate D：world model

只有 Gate C 通过才做：

- 预测 5–20 步短视野的 progress/contact/safety，而不是生成长视频；
- operator-conditioned latent dynamics；
- 与同参数量 history encoder、公平 compute budget 比较；
- 主要消融：无 action suffix、无 history、无 cost、无 operator token、无 uncertainty；
- 报告 world model 是否真正改善 operator ranking，而不是只改善 reconstruction。

## 6. 下一批代码优先级

1. 把现有 OFT direct runner接入统一 `InterventionOutcome` 和同状态 matrix scheduler；
2. 冻结 RASE-UI 10-task schedule（task/init/policy seed/checksum），不要再依赖运行时随机抽样；
3. 增加 failure-frontier sampler：优先选择 source policy value 在 0.2–0.8、或邻近时间点出现 operator disagreement 的状态；
4. 实现 ABSTAIN 的真实 safe-stop outcome 与人工代价字段；
5. 增加 task-cluster bootstrap、paired permutation、utility-lambda sensitivity；
6. 为 continue-sensitive 状态保存 RGB/action trace，做局部时间邻域 `t-10...t+10` 的 intervention map；
7. 之后才实现 history-only value model。

## 7. 可直接复现实验的命令

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python

$PY -m pytest -q \
  tests/test_decision_context.py \
  tests/test_smol_intervention_runner.py \
  tests/test_lerobot_collection_adapter.py \
  tests/test_forked_rollout_contract.py \
  tests/test_intervention_dataset.py

$PY -m ruff check \
  rase/interventions/decision_context.py \
  rase/collect/lerobot_libero_plus_adapter.py \
  scripts/export_decision_context_keys.py \
  scripts/rollout_smol_interventions.py
```

全新 smoke（目标目录必须不存在）：

```bash
$PY scripts/collect_state_pool.py \
  --config configs/collect_rase_ui_phase0_smoke.json \
  --summary-output runs/rase_ui_phase0_smoke_parity_collection_summary.json

$PY scripts/export_decision_context_keys.py \
  --pool runs/rase_ui_phase0_smoke_parity_pool \
  --output runs/rase_ui_phase0_smoke_parity_keys.json \
  --max-states 8

$PY scripts/rollout_smol_interventions.py \
  --config configs/collect_rase_ui_phase0_smoke.json \
  --state-keys-json runs/rase_ui_phase0_smoke_parity_keys.json \
  --output-dir runs/rase_ui_phase0_smoke_parity_paired \
  --continuation-seeds 1 \
  --fresh-run
```

中断后的恢复：

```bash
$PY scripts/rollout_smol_interventions.py \
  --config configs/collect_rase_ui_phase0_smoke.json \
  --state-keys-json runs/rase_ui_phase0_smoke_parity_keys.json \
  --output-dir runs/rase_ui_phase0_smoke_parity_paired \
  --continuation-seeds 1 \
  --resume
```

不要再次输入 `<pilot-config.yaml>` 这类尖括号占位符；Bash 会把它解释为输入重定向。

## 8. 当前论文级判断

- **可以写**：strict CONTINUE 的可执行定义、suffix provenance/parity 合同、同状态 paired evaluator、一个稳定 continue-sensitive 案例。
- **不能写**：CONTINUE 总体优于 REPLAN；RASE-UI 已经存在多算子 oracle gap；selector/world model 有效；真实机器人可迁移。
- **最有价值的新假设**：active suffix 在接触/对齐临界状态保存了短期动作承诺，重新规划会破坏这一承诺；在普通状态 REPLAN 可能略快。方法应学习“何时保留承诺”，而不是无条件升级到大模型。
