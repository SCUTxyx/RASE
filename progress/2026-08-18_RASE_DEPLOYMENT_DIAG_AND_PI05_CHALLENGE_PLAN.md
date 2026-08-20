# RASE 下一阶段:部署诊断(A)+ π0.5 Challenge(B)落地规划(2026-08-18)

> 依据用户 v3 规划细化。事实基线已验证(2026-08-18 服务器检查):
> - branches.jsonl 含原始成本组件:`branch_wall_s`、`intervention_query_count`、
>   `fallback_steps`、归一化 `query_cost/fallback_cost/latency_cost`;
>   **`source_prefix_wall_s` 未落盘**(缺失,记账时如实标记);
> - frozen manifest rows **无成本字段**(仅 success/utility/progress/fold 等)→
>   成本账本必须从 branches.jsonl + captures 重建,并做覆盖审计;
> - π0.5 checkpoint/tokenizer 齐备(`ckpts/pi05_libero` + `paligemma_tokenizer_35e4f46`);
>   需确认其 action-tokenizer 配置(π0-fast 用 `pi0fast_action_tokenizer_79ae83e`);
> - K3 的 24 roots 来自物理状态池 `r7a_pi0fast_reset_pool_v1`(policy 无关)→
>   **π0.5 可直接复用同 roots/tasks 做配对 cross-policy 比较**;
> - GPU 空闲。

## 0. 双线并行策略

```text
主线 A(部署诊断)  : CPU/分析线 —— 无 GPU 依赖,立即启动
主线 B(π0.5 挑战) : GPU 线 —— B0 parity smoke 先行,B1-B2 后置
```

- 两线 Gate 独立:原结论 **Signal PASS / Deployment FAIL 保持冻结**;A 线新成本
  模型只产生"新部署场景下的条件性结论";B 线只回答"动作信号是否跨 policy"。
- π0.5 复用 K3 tasks/roots → 只能称 **cross-policy challenge**,不称新的
  independent external confirmation。

---

## 主线 A:部署场景诊断

### A0. 真实成本账本(新代码,~0.5 天)

`scripts/build_rase_vnext_cost_ledger.py`,输入 k5_collect_v1(branches.jsonl +
captures + manifest.bound.json),输出:

```text
cost_ledger_v1.json 每 (root, replica, operator) 一行:
  source_prefix_steps            (sunk,所有候选共享)
  source_prefix_wall_s           (缺失率标记 —— 未落盘,补测或 None)
  requery_extra_inference        (requery 分支的 branch_wall_s 增量 + query 数)
  fallback_pre_query_count       (intervention_query_count = OFT chunk queries)
  fallback_inference_steps       (fallback_steps)
  fallback_wall_s                (branch_wall_s)
  end_to_end_env_steps           (post_decision_env_steps)
  归一化成本                    (query_cost/fallback_cost/latency_cost 保留)
  incremental_cost               (decision point 起新增,不含 sunk)
  单位说明 + 字段来源 + 计时边界 + 缺失率 + ledger sha256
```

- **sunk vs incremental**:prefix 成本对所有 operator 相同,不得重复惩罚;
  只比较"从 decision point 起选择不同候选而新增的成本"。
- **覆盖审计**:对照 `selector.py::cost_of_row` 与冻结 manifest,输出
  "字段来源 × 覆盖行数"表;缺失字段(如 prefix wall)明确标记,不臆造。
- wall-time 与归一化成本同时保留(规划要求)。

### A1. 预注册三场景(文档,~0.5 天)

`protocols/selector_deployment_scenarios_v1.json`(在 A2/A3 结果前冻结):

| 场景 | 定义 | 预期 |
|---|---|---|
| S0 current-cheap-fallback | 现行成本口径(fallback 便宜可用) | 复现 FAIL(对照组,不允许事后改写) |
| S1 latency-budgeted | fallback 可用,但受硬延迟预算(如总等待 ≤ T_budget)约束 | selector 可能胜出 |
| S2 fallback-constrained | fallback 不可用/限额(如仅 30% 请求可调用) | 反事实场景,不得用于推翻 S0 FAIL |

- λ 不再无单位:用可解释预算映射,如"每允许 100ms 延迟损失,可接受 X 的
  success 下降";同时报告 raw success × latency Pareto frontier。

### A2. Oracle 可行性上界(先算,~0.5 天)

`scripts/analyze_rase_vnext_deployment_feasibility.py`:

- per-unit 统计:**fallback 非最优率**(fallback 非最高 success 的 unit 占比)、
  continue/requery 可替代率、可挽回 success、fallback 调用削减率;
- **task/root/unit 三层计数** + task bootstrap CI(避免 4/24 聚合口径误导);
- 在**真实成本账本**下计算 oracle Pareto 上界:
  - 若 oracle 在预注册预算内都无法优于 always-fallback → **关闭该场景**,
    不再调 margin/model(停机规则);
  - oracle 有空间 → 才进入 A3。

### A3. margin_λ 冻结与评估(1 天)

扩展 `scripts/run_rase_vnext_selector_oof.py`:

- 仅在 K5 每 outer fold 的 calibration tasks 上,为**每个场景/λ** 选择 margin_λ;
  outer tasks 只评估一次;
- K3 不再用于调参;若用,标注为 **post-hoc robustness**(新成本定义下跑一次);
- 比较:selector / continue / requery / always-fallback / quota-aware fallback /
  oracle;报告 success、真实延迟、utility、coverage、fallback-call rate、regret、
  bootstrap CI;
- **Deployment-v2 PASS 条件**:预注册的至少一个真实场景/预算区间内,selector
  对目标 baseline 的 CI 下界不劣且显著减少受约束资源;否则 FAIL。

---

## 主线 B:π0.5 8-task × K3 Cross-Policy Challenge

### B0. Capability/Parity Smoke(先于一切正式收集,~0.5-1 天)

`scripts/run_rase_vnext_pi05_parity_smoke.sh` + 审计脚本,依次通过:

1. checkpoint/tokenizer 路径、hash、config/policy 版本记录
   (`pi05_libero/config.json` 的 type/chunk/n_action_steps/temperature;
   action-tokenizer 配置确认——可能复用 paligemma tokenizer);
2. 离线 loader(`load_lerobot_policy_bundle` 加载 pi05_libero 成功);
3. 单 group 收集(1 root × 6 ops,复用现有 collector);
4. 跨 suite source parity(至少 2 suite 各 1 group,prefix 首动作 hash 一致性);
5. snapshot replay / queue cursor / inference-event provenance(复用 E0 审计);
6. requery 独立生成(与 continue 不同 seed → 独立推理);
7. **native resample capability**(π0.5 是否有 diversity;若有,数据驱动
   executable,与 π0-fast 的 incapable 形成对比);
8. action schema / chunk 长度 / 归一化 / 相机状态输入 / **GPU 共存**
   (π0.5 与 OFT server 同卡 32GB 显存预算检查)。

任一 parity/provenance FAIL → **停止正式 challenge**,先修复。

### B1. 冻结 Challenge Cohort(~0.5 天)

- **复用 K3 的 8 tasks × 3 roots**(从 `k3_cohort_manifest_v1.json` 提取 roots,
  同一物理状态池)——获得**配对 cross-policy 比较**;文档明确:不是新的独立
  task confirmation;
- 冻结 policy-specific manifest(pi05_libero)、seed ledger(新 salt)、
  operator capability、capture schema v2、完整 hashes;
- 明确标注 `scientific_scope = CROSS_POLICY_CHALLENGE_NOT_EXTERNAL_CONFIRMATION`。

### B2. 收集(1-2 天 GPU,432 slots)

- 8 tasks × 3 roots × 6 ops × K3 = 432 slots(与 π0-fast K3 同口径);
- **stop/go checkpoints**:先 1-group smoke → 再 1-task(3 roots)检查 → 全量;
- runner 复用 `run_rase_vnext_k3_collect.sh` 模式(policy-path/ckpt 换 pi05);
- 主指标预注册:same-root raw-action pairwise accuracy;分层:source-source;
  次指标:fold/task 方向、bootstrap CI、operator layers、capture/provenance audit;
- **两层报告,不混为一谈**:
  1. π0-fast frozen selector 零样本迁移到 π0.5;
  2. π0.5 内 task-held-out 重新训练(in-policy 信号);
- 不池化两 policy 训练 universal selector(除非两边 Gate 各自通过后另立协议)。

### B3. π0.5 Gate

- **Policy-signal PASS**:π0.5 内 pairwise 主指标 > 0.5 且 task bootstrap CI 完整
  报告;小样本 CI 跨 0.5 → 判定 **inconclusive**(不强行 PASS/FAIL);
- **Cross-policy transfer PASS**:π0-fast frozen 模型在 π0.5 上预注册指标不退化
  到随机,并报告相对 π0.5 in-policy 模型的差距;
- π0.5 信号弱/迁移失败 → **不影响 π0-fast 已冻结的 Signal PASS**,科学结论
  收窄为 policy-dependent。

---

## 执行顺序与停机规则

```text
并行启动: A0 成本账本审计 ‖ B0 parity smoke
  │
  ├─ A2 oracle feasibility:
  │     oracle 无空间 → 关闭该场景,停止 selector 调参(硬停机)
  │     oracle 有空间 → A3 margin_λ nested-OOF → Deployment-v2 判定
  │
  └─ B0 全 PASS → B1 冻结 manifest → B2 432 slots(带 1-group/1-task checkpoints)
        → B3 Gate(policy-signal + cross-policy transfer)
        │
        └─ 两线完成后:四格结论 → 决定是否 closed-loop
```

- **closed-loop 决策**:只有真实部署场景出现 break-even 才值得做成本约束下
  closed-loop;否则**关闭 learned-selector 部署主线**。
- 全程不启动:OPD / RL / world-model / universal pooled selector / 大规模下载。

## 新代码清单

| 文件 | 用途 |
|---|---|
| `scripts/build_rase_vnext_cost_ledger.py` | A0 成本账本 |
| `protocols/selector_deployment_scenarios_v1.json` | A1 三场景冻结 |
| `scripts/analyze_rase_vnext_deployment_feasibility.py` | A2 oracle 上界 |
| `scripts/run_rase_vnext_pi05_parity_smoke.sh` + `audit_rase_vnext_pi05_parity.py` | B0 |
| `scripts/freeze_rase_vnext_pi05_cohort.py` | B1(基于 k3 roots 换 salt) |
| `scripts/run_rase_vnext_pi05_collect.sh` | B2 |
| `scripts/analyze_rase_vnext_pi05.py` | B2/B3(复用 analyzer,换 manifest) |
| A3 扩展 | OOF 脚本加场景/λ 化 margin |

## 最终四格结论表

| 问题 | 判定 | 依据 |
|---|---|---|
| 动作信号跨 policy? | B3 policy-signal | π0.5 in-policy acc + CI |
| 模型跨 policy? | B3 transfer | π0-fast 模型零样本迁移 |
| 当前场景可部署? | A3 S0 | 复现 FAIL(冻结) |
| 受约束场景可部署? | A3 S1/S2 | oracle 上界 + margin_λ OOF |

## 时间预估

- A 线:1.5-2 天(CPU)
- B 线:B0 0.5-1 天 + B1 0.5 天 + B2 1-2 天 GPU + B3 0.5 天
- 两线并行,总 ~3 天;GPU 空闲,可立即启动 B0。
