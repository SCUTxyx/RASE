# RASE 下一阶段执行报告：Same-Source Regeneration Eligibility

日期：2026-08-20
结论范围：clean LIBERO Goal/Long、SmolVLA flow-noise 重采样、K=4、temperature=0.7。

## 1. 结论

RASE 的核心 idea 保持不变：冻结 VLA，在同一物理状态比较 `continue.source`、`requery/resample.source`、跨策略 `fallback.persistent`，用 same-root 反事实后果学习相对风险，并以保守规则闭环仲裁。

本阶段证明了两件事：

1. `requery/resample.source` 是真实候选轴，不是重复动作：32 个 runtime roots 的平均 pairwise chunk L2 为 0.917；8/32 roots 同时出现成功与失败候选；相对真实 continue，重采样集合救回 4 个状态，`continue ∪ resample` 提升 12.5pp。
2. 当前 temperature-only 生成器仍不足以支撑 verifier/闭环：`oracle@4` 相对任取首样本只提升 3.125pp，未达到预注册的 5pp；重采样对 OFT fallback 没有任何独有成功，完整 oracle 仍等于 fallback 的 81.25%。

因此当前正确动作是：**不训练 verifier、不调 selector 阈值、不跑闭环；先改善 candidate generator 的语义覆盖，再重复 Eligibility Gate。** 这不是改变 idea，而是按 RASE 的候选充分性前置条件推进。

## 2. 已完成的代码完善

新增或修改：

- `scripts/rollout_pool_candidates.py`
  - 新增 `protocol.continuation_seed_mode`；
  - 正式实验使用 `common_root_rollout`：同一 root、同一 repeat 的 K 个候选共享后续 continuation RNG；
  - 候选首 chunk 仍使用各自独立 generation seed；
  - summary 和每条 scheduler result 均记录 seed mode。
- `scripts/freeze_regeneration_keys.py`
  - 只按 suite、扰动维度和 step 冻结 cohort；
  - 明确记录 `selection_uses_outcomes=false`、任务计数和 key checksum。
- `scripts/analyze_regeneration_opportunity.py`
  - 同时比较 continue、first resample、oracle@K、continue∪resample、fallback、full oracle；
  - task-cluster bootstrap 95% CI；
  - R0 候选可学性 Gate 与 R0X 跨策略互补 Gate；
  - BOKBO 式 action-diversity/outcome 相关诊断。
- `configs/crr_regeneration_clean32.yaml`
  - 固定 K=4、temperature=0.7、common-RNG、32-state clean Goal/Long 协议。
- `scripts/run_crr_regeneration_clean32.sh`
  - preflight → cohort freeze → candidate generation → rollout → audit 的可恢复一键管线；
  - fresh run 不覆盖已有结果。
- `tests/test_regeneration_protocol.py`
  - metadata-only cohort、key 顺序、common-RNG、双 Gate 合成回归测试。

服务器测试结果：15 passed；`git diff --check` PASS。GPU 已回到 0 MiB / 0% utilization。

## 3. 实验协议

- Cohort：8 tasks × 4 states（step 0/2/4/6）= 32 roots。
- Suites：Goal 16、Long 16；全部 clean。
- 选择纪律：从已冻结的 96-state artifact 仅按 metadata 筛选，不读 outcome/disagreement。
- 每个 root：SmolVLA 以 flow-noise temperature=0.7 生成 K=4 个 10-step chunks。
- 后果：执行候选 chunk 后由同一冻结 SmolVLA 继续至真实 success/horizon。
- 方差控制：同一 root 的四个分支共享 continuation RNG seed。
- 计算量：128 条真实终局 rollout；候选生成 172.8s，rollout 2837.2s；单 rollout 中位数 15.48s。

预注册 Gate：

- R0：动作多样性；mixed roots ≥10%；`oracle@K − first ≥5pp`；`continue∪resample − continue ≥5pp`；rescue 跨 ≥2 tasks。
- R0X：R0 PASS；至少 1 个 resample-only-vs-fallback；full oracle 相对 best fixed ≥5pp。

## 4. 正式结果

| 指标 | 结果 |
|---|---:|
| Continue success | 15/32 = 46.875% |
| First resample success | 18/32 = 56.25% |
| Oracle@4 resample | 19/32 = 59.375% |
| Oracle@4 − first | +3.125pp，task-bootstrap 95% CI [0, 9.375pp] |
| Continue ∪ resample | 19/32 = 59.375% |
| (Continue ∪ resample) − continue | +12.5pp，95% CI [0, 31.25pp] |
| Mixed-outcome roots | 8/32 = 25% |
| Continue failures rescued by resample | 4，跨 2 tasks |
| OFT fallback | 26/32 = 81.25% |
| Full oracle | 26/32 = 81.25% |
| Resample-only / fallback-only / both / neither | 0 / 7 / 19 / 6 |
| Full oracle − best fixed | 0pp，95% CI [0, 0] |

分 suite：

- Goal：continue 56.25%，first resample 62.5%，oracle@4 68.75%，fallback 68.75%；4 个 mixed roots。
- Long：continue 37.5%，first/oracle@4 均 50%，fallback 93.75%；4 个 mixed roots。

Gate 判定：

- R0：4/5 条件通过，仅 `oracle@K − first ≥5pp` 失败（实际 3.125pp）。
- R0X：失败；resample 对 fallback 的独有成功为 0，full oracle 无增益。
- Verifier-training gate：closed。
- Cross-policy claim gate：closed。
- Closed-loop gate：closed（candidate eligibility failed）。

## 5. BOKBO 式诊断

动作几何多样性没有稳定对应真实机会：

- mixed roots 的平均 chunk L2 = 0.496；uniform-outcome roots = 1.057；
- corr(chunk L2, mixed outcome) = -0.275；
- corr(chunk L2, oracle gain) = -0.080；
- corr(chunk L2, candidate success count) = -0.174。

最高 L2 的 roots 同时包含 4/4 全成功和 0/4 全失败。结论是：temperature 噪声能扩大动作距离，但不能保证跨越任务成功边界。不能用 disagreement 或 margin 代替真实后果标签。

## 6. 下一阶段执行顺序

### G1：Structured Regeneration Eligibility（当前最高优先级）

目标：不改候选语义，只把 `resample.source` 从单一 temperature 扰动升级为有质量约束的同策略候选集。

1. 实现 per-candidate noise schedule，例如 temperature `{0.3, 0.5, 0.7, 0.9}`，每档两个 seed，K=8；保留 candidate 0 为预注册基线。
2. 先在开发集做 8-root temperature-quality probe，记录 success、chunk L2、gripper disagreement、latency；禁止用 probe root 做最终确认。
3. 仅当 probe 显示至少 2 个 `first fail → later success` roots，才在新的 outcome-independent roots 上复跑 32-state Gate。
4. Gate 仍采用 R0/R0X，不因结果调整阈值。

这里验证的是“多尺度原生采样能否制造语义候选”，不是训练 selector。

### G2：Cross-Policy Goldilocks Screen（与 G1 同等重要，顺序执行以节省 GPU）

1. 补测 π0-fast 在 clean LIBERO Long 的直接成功率；目标窗口 30%–70%。
2. 若落在窗口内，运行 SmolVLA/π0-fast same-root 24-state mini screen。
3. 必须满足：continue-only ≥1、fallback-only ≥1、`H_within ≥5%`、oracle gain ≥5pp。
4. 若 π0-fast 仍被 OFT/或支配 SmolVLA，停止该 pair，不训练风险模型。

这一步保留 RASE 最关键的跨策略零样本与 policy-invariance 主张。

### V1：Relative verifier（仅在 G1 或 G2 Gate 打开后）

- 目标：`P(a_i ≻ a_j | s_t, instruction, a_i, a_j)`；task-held-out 为主。
- 必须包含：action-shuffled、label-permuted、state-only、no-context 对照。
- 输出：pairwise AUROC、top-1 accuracy、precision-coverage、conformal risk-control 曲线。
- 开 Gate 条件：真实模型显著超过 action-shuffled/permuted；高精度区有非零 coverage；不允许 task lookup。

### C1：Risk-Guided Regeneration 闭环（仅在 V1 PASS 后）

- 每个决策点先采样 K 个 source 候选；risk rerank。
- 全部不通过时最多重采样 B=2 轮；预算耗尽才使用 `fallback.persistent`。
- 同时报告 success、fallback rate、额外 forward 次数、wall-clock latency。
- 与 default、random-K、oracle@K、always-fallback、heuristic-margin 做 compute-matched 比较。

## 7. 明确停止项

在新的 candidate Gate 打开前，不做：

- 不继续训练当前 CRR verifier；
- 不用 8 个 mixed roots训练模型并汇报高 AUROC；
- 不调 abstain/margin 阈值；
- 不上 RGB/World Model；
- 不跑闭环；
- 不把当前 32 roots 重新包装成 held-out confirmation。

下一条最有信息增益的工作是 G1 多尺度原生采样 probe，随后是 G2 π0-fast Long Goldilocks screen。
