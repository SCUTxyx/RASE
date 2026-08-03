# NGC W5 proposal-temperature sweep：t=0.3 / 0.7 / 1.0

| 项 | 内容 |
|---|---|
| 日期 | 2026-07-27 |
| 状态 | **完成；proposal-temperature 诊断线关闭** |
| Cohort | failure frontier L3–L5，24 个冻结状态 |
| Config | `configs/ngc_w5_failure_frontier_screen.yaml` |
| 模式 | `smolvla-screen`，每候选 1 trial，非正式 Set 标签 |
| 候选 | K=8，chunk `[10,7]` |

## 1. 结果

| proposal temperature | states | rollouts | candidate hits | state hits |
|---:|---:|---:|---:|---:|
| 0.3 | 24 | 192 | **0** | **0** |
| 0.7 | 24 | 192 | **0** | **0** |
| 1.0 | 24 | 192 | **0** | **0** |
| **合计** | 72 state-runs | 576 | **0** | **0** |

三档均为 screen-only，不能写成 Wilson Set C；可以写为冻结 cohort 上
`0/576` one-shot candidate outcomes。

## 2. 工程假阴性排查

候选文件与温度元数据均正确：每档 24 个 artifact、24 个不同文件 hash，三档使用完全
相同的 24 个 state keys。

| temperature | mean pairwise endpoint L2 | minimum state mean | mean action L2 |
|---:|---:|---:|---:|
| 0.3 | 0.726 | 0.377 | 1.038 |
| 0.7 | 2.133 | 1.169 | 1.052 |
| 1.0 | 3.066 | 1.575 | 1.069 |

任意两档均有 `0/24` 状态的候选数组完全相等；mean absolute action delta 为
0.0314（0.3↔0.7）、0.0298（0.7↔1.0）、0.0590（0.3↔1.0）。因此温度确实改变候选，
且多样性单调增加。零命中不是候选复用、温度未生效或 diversity collapse。

## 3. 结论与决策

结合 W3 continuation-temperature ablation、W4 `0/1536`、W5 OFT-recovered smoke
`0/136`，继续扩大 SmolVLA proposal-temperature 网格的信息增益已经很低。

**Kill decision：**

1. 不跑 formal confirm（union of screen hits 为空）。
2. 不增加更多 proposal temperature 或 L3–L5 SmolVLA 重复 trial。
3. 下一步转到 L1–L2 success/failure-balanced、episode-distinct pilot。
4. 在同一冻结 cohort 上并列 Smol→Smol screen 与 Smol→OFT deterministic verify，
   将 recoverability 明确写成 policy-relative outcome。

## 4. 产物

- `runs/ngc_w5_failure_frontier_candidates_{t03,t07,t10}/`
- `runs/ngc_w5_failure_frontier_screen_{t03,t07,t10}/summary.json`
