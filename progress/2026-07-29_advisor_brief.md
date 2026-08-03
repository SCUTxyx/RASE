# RASE 导师汇报简报（2026-07-29）

## 一句话结论

RASE 当前最稳固的发现不是“SmolVLA 候选能够恢复失败”，而是：**恢复性取决于
continuation policy，且强 policy 的可恢复性不能归因于候选动作本身。** 项目已据此从
candidate-fallback 叙事转为 episode-disjoint、policy-relative recoverability benchmark，
下一阶段方法目标是学习何时升级到强 policy，而不是继续优化无正例的候选 reranker。

## 已完成的硬结果

| 证据 | 结果 | 可以支持的结论 |
|---|---:|---|
| SmolVLA clean LIBERO | 70.0%，2,000 episodes | base policy 在干净分布有效 |
| LIBERO-Plus camera/robot full collapse | 0.38%，3,142 completed | 视觉扰动造成强烈 failure frontier |
| W4 failure-conditioned cohort | Smol 0/1,536；OFT portfolio 17/32 | recoverability 强烈依赖 continuation policy |
| W5 同 cohort 温度扫描 | t=0.3/0.7/1.0 合计 0/576 | proposal temperature 不是瓶颈 |
| W6 L1–L2 配对矩阵 | Smol 0/8；OFT 2/8；McNemar p=0.5 | 有方向性差异，但 pilot 尚不具显著性功效 |
| W7 discovery prefix ablation | candidate-specific rescue 0/2 | 现有 OFT 命中不能归因于 candidate 内容 |
| W7 held-out matrix | Smol 0/24；prefix+OFT 8/24；McNemar p=0.0078125 | episode-disjoint failure cohort 上存在显著 policy-relative gap |
| W8 direct OFT escalation | 9/24；Goal 4/8、Long 5/10 | 主要增益无需候选 reranking 即可出现 |

W7 的两个 OFT-only discovery state 分别被解释为：

1. direct OFT、zero prefix、8 个 candidate prefix 全成功：OFT continuation 本身充分；
2. direct OFT 失败，但 zero prefix 与 candidate 4/5 成功：被动等待/环境动力学已是充分替代解释。

因此不再宣称“learned proposal causes rescue”。这是一次有价值的机制否证，使后续论文主张更可信。

## 已完成的 held-out 主结果

- 24 个 held-out failure states，来自 24 个不同 episode group；
- 与 W6 的 state 和 episode-group overlap 均为 0；
- camera-L1、camera-L2、robot-L1、robot-L2 各 6 个状态；
- `t0` 范围 0–36、中位数 10：存在早期时点变化，但仍是 early-stage cohort，
  不能外推到 mid/late recovery；
- 冻结 `K=8`、temperature=0.7、horizon 与 checkpoint，不看结果调参；
- 对每个 state 配对比较 Smol→Smol 与 Smol→OFT portfolio；
- 主统计单位为 state，报告 exact McNemar；candidate-level hit 仅作描述；
- 最终 Smol 为 **0/24**、prefix+OFT 为 **8/24**，exact McNemar
  **`p=0.0078125`**；
- 相同 24 snapshots 上 direct OFT 为 **9/24**：Goal 4/8、Long 5/10、
  Object 0/2、Spatial 0/4；
- direct OFT 对 Smol 的配对差异为 exact McNemar **`p=0.00390625`**；
- W7 `8/24` 与 W8 `9/24` 的优劣必须等逐状态 overlap 表，不能只比较边际率。

逐状态 overlap 已完成：both-success 7、prefix-only 1、direct-only 2、both-fail
14；prefix 与 direct 的 exact McNemar `p=1.0`。因此 direct OFT 的优势是以一个
可部署动作覆盖 oracle prefix portfolio 的 7/8 命中，而不是统计上显著更高的成功率。

W8 已导出真正可部署的 direct-arm action outcome 文件。不过该文件全部来自
SmolVLA failure episodes，因此 readiness audit 应拒绝三动作 selector 训练；这避免了
把模型训练成“任何时候都升级 OFT”并隐瞒 clean regret。

## 明天汇报时的主张边界

可以说：

- 构建了确定性 snapshot/restore、episode-disjoint split、同 candidate 配对评测和
  provenance checksum 的恢复性实验基础设施；
- 多轮证据一致表明 Smol proposal/continuation 在当前 failure frontier 缺乏正例；
- OFT 在 W7 held-out 上相对 Smol 的状态级差异显著，支持 policy-relative
  recoverability 与 escalation selector 研究方向；
- 显式 prefix 控制推翻了候选特异性解释，避免将 continuation 能力误写成 proposal 能力。

暂时不能说：

- 该结果代表无条件任务成功率或所有 suite 均能恢复；
- Smol candidate 能够因果恢复状态（当前归因 0/2）；
- failure-conditioned rate 是无条件任务成功率；
- selector 已经有效（clean-regret/cost control 与训练尚未完成）。

## 下一步三道 gate

1. **OFT-route attribution（已完成）：** both 7 / prefix-only 1 / direct-only 2 /
   neither 14，主动作冻结为 direct escalation。
2. **Clean/cost control：** pool audit 已确认 0 success episode-group，下一步新采
   成功轨迹 control，测 always-escalate 的 clean regret、
   strong-policy usage、latency 和 net success。
3. **最小 selector：** 先训练 calibrated linear/MLP escalation selector；只有在
   task-held-out split 上优于等预算 random trigger/risk threshold，才进入复杂 RL。

2026-07-29 代码进展：三臂已冻结为 `CONTINUE_SMOL / ESCALATE_OFT / ABSTAIN`，
dependency-free ridge utility baseline、episode/task split、proxy/leakage/readiness gate 已实现。
下一实验是 32-state clean-control pilot：真正的 empty-prefix direct Smol 与 direct
OFT 双臂标注。ground-truth dim/level/outcome 禁止进入特征；clean-success control
合并并通过 readiness 后才允许训练。

顶会版本的真正门槛是：至少两个能力层级 backbone、无泄漏 split、配对统计、成本—成功率
Pareto，以及一个 held-out 上优于规则基线的 selector。若第三道 gate 不通过，优先投稿高质量
benchmark/diagnosis，而不是用复杂 RL 稀释当前最强证据。
