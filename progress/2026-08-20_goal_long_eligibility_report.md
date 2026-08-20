# RASE Goal/Long Same-State Eligibility 实验报告

日期：2026-08-20

## 结论

本轮正式 Eligibility Screen **FAIL**。在 Goal/Long 的 96 个冻结状态上，direct OFT fallback 对 strict CONTINUE 逐状态弱支配：

- fallback-only：51/96；
- tie：45/96；
- continue-only：0/96；
- 同任务严格胜者翻转：0/24 tasks；
- `H_within = 0%`，task-cluster bootstrap 95% CI `[0%, 0%]`；
- state oracle success：68.75%；
- task-best-fixed success：68.75%；
- oracle gain：`0pp`，95% CI `[0pp, 0pp]`。

因此，当前 Goal/Long 状态池仍不具备 RASE runtime arbitration 的可识别机会。继续训练 relative-risk selector 只能学习“总是 fallback”，无法形成相对 task-best-fixed 的成功率增益。

## 本轮修正和新增代码

### 状态 cohort 导出

增强 `scripts/export_decision_context_keys.py`：

- 支持 `--steps 0,2,4,6`；
- 支持可重复的 `--suite` 与 `--task-id` 过滤；
- 保留确定性排序、精确 key、decision-context 校验和 cohort checksum；
- `--step` 与 `--steps` 互斥，非法输入会立即失败。

### 正式机会分析

新增 `scripts/analyze_continue_fallback_opportunity.py`：

- 标签只取恢复同一 simulator state 后的完整 episode `success`；
- 比较 strict active-chunk CONTINUE 与 direct OFT fallback；
- tie 不使用耗时或代理 consequence 强制破局；
- 计算任务内严格胜者翻转、state oracle、task-best-fixed 和 oracle gain；
- 以 task 为 cluster 做 10,000 次 bootstrap；
- 输出 suite 与 perturbation dimension 分层结果；
- 显式审计 fallback 是否逐状态弱支配；
- 不使用 task id 作为 selector feature，不把成本写入训练标签。

### 测试

新增 `tests/test_continue_fallback_opportunity.py`，并运行相关回归测试：

```text
11 passed in 1.21s
```

覆盖新分析器、key exporter、decision context 与既有 deferred-switch 分析器。

## 实验设计

- 来源池：`runs/rase_ui_phase0g_independent48_pool`；
- suite：Goal、Long；
- 任务：24 个，每个任务属于 clean / robot / camera 之一；
- 每任务状态：step 0、2、4、6，共 4 个；
- 总状态：96；
- 状态选择：基于冻结池的 suite/step 元数据，不按 action disagreement 或 outcome 筛选；
- CONTINUE：恢复保存状态，先执行保存的 active action suffix，再由 SmolVLA 继续到成功或 episode horizon；
- fallback：恢复同一状态，direct OFT 持续执行到成功或 episode horizon；
- primary label：任务终局 success；
- Gate：`H_within >= 5%` 且 `OracleGain_within >= 5pp`。

正式运行前还执行了 2-task / 8-state smoke，验证恢复、suite 切换、direct-only summary 与 Gate FAIL 路径。

## 正式结果

| 分组 | states / tasks | CONTINUE | OFT fallback | fallback-only | tie | continue-only | H_within | oracle gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 96 / 24 | 15.63% | 68.75% | 51 | 45 | 0 | 0% | 0pp |
| Goal | 48 / 12 | 18.75% | 64.58% | 22 | 26 | 0 | 0% | 0pp |
| Long | 48 / 12 | 12.50% | 72.92% | 29 | 19 | 0 | 0% | 0pp |
| clean | 32 / 8 | 46.88% | 81.25% | 11 | 21 | 0 | 0% | 0pp |
| robot | 32 / 8 | 0% | 53.13% | 17 | 15 | 0 | 0% | 0pp |
| camera | 32 / 8 | 0% | 71.88% | 23 | 9 | 0 | 0% | 0pp |

20/24 tasks 至少有一个 informative state，但所有 informative state 都是 fallback-only。这个区别很关键：数据存在 outcome variation，也存在 fallback rescue，但不存在 comparative advantage 的方向变化。

## 科学解释

1. **Goal/Long 没有修复核心结构问题。** Long horizon 增加了失败暴露，但没有产生 source 独占成功；fallback 仍然处处不差于 source。
2. **clean 子域只解决了 source floor，没有解决 fallback dominance。** clean 的 CONTINUE 为 46.88%，符合中等难度直觉，但 fallback 达到 81.25%，且没有 continue-only。
3. **robot/camera 子域连 E1 都失败。** CONTINUE 均为 0%，这里没有可学习的双向比较优势。
4. **高 informative-state 数量不能替代 oracle headroom。** 51 个非平局全朝向 fallback，所以 oracle 与 task-best-fixed 完全相同。
5. **当前不应进入视觉、CRR 扩模或闭环 selector 训练。** 在标签单向支配时，任何高 AUROC 都不能证明 runtime selector 价值。

## 接下来怎么推进 idea

### 立即停止

- 不在当前 Goal/Long cohort 上继续调 CRR、阈值或视觉编码器；
- 不用 fallback 成本或 latency 改写 primary success label；成本只能作为部署约束和次指标；
- 不把 task router 的收益包装成 runtime risk control。

### 下一轮只做 Pre-RASE Eligibility

优先寻找一个真正的 Goldilocks regime，建议顺序：

1. 选择 intermediate source，使统一评估中的 source success 位于 30%–70%；
2. 选择与 source 失败模式不同的 fallback，优先跨架构组合，而不是更强同族教师；
3. 先跑 2-task 工程 smoke；
4. 冻结至少 24 tasks × 4–8 same-task states 的均匀 evaluation cohort；
5. 只测完整终局 success，并要求同时出现 continue-only 与 fallback-only；
6. 只有 `H_within >= 5%` 且 oracle gain `>= 5pp` 后，才进入 relative-risk 建模、视觉交互特征和闭环 selector。

结合现有资产，最合理的候选方向是 SmolVLA / π0-fast / OFT 的跨架构 same-state 筛选，或先把 source 微调到 LIBERO-90 的中等成功区间。关键不是换更大模型，而是让两个 provider 具有互补而非单向支配的失败模式。

当前这批数据应保留为论文中的 negative-control / domain-eligibility 证据：它清楚说明“长任务、更多时点、真实终局标签”仍不足以保证 runtime arbitration opportunity。

## 可复现 artifacts

服务器根目录：`/root/autodl-tmp/RASE`

```text
runs/rase_continue_fallback_goal_long96_keys.json
  sha256 6b0b98ba04191c8debf5ae6cc65e95eb9a91e839f9bd505d886960e09f346626
runs/rase_continue_fallback_goal_long96_smol/summary.json
  sha256 6c5bae13755b949d9fb66f9c19da28a0070865965686400cbd95c1c8309848ec
runs/rase_continue_fallback_goal_long96_oft_goal_v1/summary.json
  sha256 3c97e3344c56bc36c98bb367013438ad3bfa701b22cbd833a53f8e51d547840b
runs/rase_continue_fallback_goal_long96_oft_10_v1/summary.json
  sha256 11280deab2f89a60e46b147a1344d2398acc1a8936ef9de2fbb1d06372cce7f2
runs/rase_continue_fallback_goal_long96_opportunity.json
  sha256 adf24fefe5491243aa08a61eb09cf821b52cf0abaef9105f8c505b89a3933a33
```

运行时间：CONTINUE 2669.97s，Goal OFT 352.47s，Long OFT 645.13s；GPU 服务进程已清理。
