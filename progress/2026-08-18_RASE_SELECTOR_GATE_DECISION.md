# RASE Semantic Selector Gate 判定(2026-08-18)

## 一句话结论

> **Signal Gate PASS,Deployment Gate FAIL**。same-root 动作信号在显式 pairwise
> 建模下稳健(K5 0.871、K3 外部确认 0.778,source-source 0.706/0.75),但离线
> selector 在所有成本权重下都**无法超越 always-fallback**——fallback 成功率高
> 且成本低,selector 在当前部署场景下无 break-even 区间。按协议:不宣称 semantic
> selector 成功,先诊断成本/部署场景,不启动 OPD/RL。

## 1. 冻结产物

- K5 开发集:`runs/rase_vnext/frozen/selector_dataset_manifest_v1.json`
  (sha256 `a65e76d1…`,432 rows,features sha256 `b0d3d11a…`)
- K3 确认集:`runs/rase_vnext/frozen/selector_dataset_k3_manifest_v1.json`
  (sha256 `0fd25d28…`,216 rows)
- K5 nested OOF:`runs/rase_vnext/selector_oof_v1/report.json`
- K3 一次性确认:`runs/rase_vnext/selector_oof_v1/k3_confirm.json`
- 代码:`rase/vnext/selector.py`(双模型/abstain/risk-coverage/U_λ)、
  `scripts/run_rase_vnext_selector_oof.py`、`scripts/confirm_rase_vnext_selector_k3.py`、
  `tests/test_vnext_selector.py`(6 tests passed,全套 505 passed)

## 2. P0 完成

- 特征错位 bug 修复(records/features 原子 append + 对齐审计);
- offline utility 的 oracle 回退移除(纯预测驱动,margin-based abstain);
- K5 基线重算无回归(ladder 指标一致)。

## 3. Signal Gate:PASS ✅

K5 nested OOF(task-held-out,超参仅在 calib folds 选择):

| 模型 | pairwise acc | task bootstrap | source-source |
|---|---:|---|---:|
| **raw-action pairwise(冻结)** | **0.871** | mean 0.867,CI **[0.75, 0.95]**,24/30 tasks | **0.706** |
| raw-action candidate | 0.837 | CI [0.733, 0.933] | 0.441(反向) |
| trace-only pairwise | 0.787 | CI [0.667, 0.883] | 0.471 |

K3 一次性外部确认(冻结模型,仅跑一次):

| 指标 | 值 |
|---|---:|
| pairwise acc | **0.778**(108 pairs) |
| task bootstrap | mean 0.774,CI [0.631, 0.917],5/7 tasks |
| **source-source** | **0.750** |
| cont-fb / req-fb | 0.786 / 0.786 |

**关键发现**:显式 pairwise 建模恢复了 source-source 信号(K5 0.706、K3 0.75),
而 candidate 级模型在 source-source 上反向(0.441)——此前"同策略信号不稳定"的
结论是**模型族选择问题**,不是信号缺失。**冻结:raw-action + pairwise 模型**。

## 4. Deployment Gate:FAIL ❌

margin=0.1(calib 冻结),U_λ = success − λ·cost:

| λ | selector | continue | fallback | oracle | break-even(≥fb) |
|---|---:|---:|---:|---:|---|
| 0.05 | 0.76 | 0.47 | 0.95 | 0.96 | 无 |
| 0.1 | 0.72-0.76 | 0.47 | 0.95 | 0.96 | 无 |
| 0.2 | … | … | … | … | 无 |
| 0.5 | … | … | … | … | 无 |
| 1.0 | … | … | … | … | 无 |

K3 确认同样:selector 0.71 vs fallback 0.96,break-even 空。

**原因分析**:
1. fallback 成功率 95.8% 且 protocol 权重下成本低(λ≤1 无法翻盘);
2. selector 选择 fallback 的单元(60/144)已获得其高成功,但 abstain→continue
   的单元(81/144)拉低整体——模型对 continue 优于 fallback 的场景识别不足;
3. 这不是校准问题(ΔBrier/ΔECE 均改善),是**部署场景问题**:当 always-fallback
   可用且便宜时,任何 selector 都难有净收益。

## 5. 按协议的处理

- ✅ 不宣称 semantic selector 成功(Deployment Gate 未过);
- ✅ 不启动 OPD/RL/world-model/closed-loop 扩展;
- 诊断方向(下一轮):
  1. **成本/部署场景重定义**:fallback 的真实延迟成本(预查询+推理)在 λ 定义中
     未体现;给出"fallback 不可用/高延迟预算"场景的敏感性分析;
  2. **abstain 规则细化**:margin 只在 calib 按 λ=0.1 冻结,可预注册多 λ 冻结
     (margin_λ),而不是单一 margin;
  3. **selector 定位修正**:作为"fallback 成本感知的降级策略"而非"超越 fallback
     的主策略";在 closed-loop 验证前先回答"哪些单元 fallback 并非最优"(仅
     4/24 root 级别),若太少,selector 的实用价值本身存疑。

## 6. 科学贡献(可写入口径)

- 首个在 48-task task-held-out + 外部 8-task 确认下成立的动作→结局信号
  (pairwise 0.871/0.778);
- 同策略(source-source)动作语义可学习(pairwise 0.706/0.75);
- 但动作信号的可学习性 ≠ 部署价值:fallback 主导场景下 selector 无 break-even,
  这本身是对"动作风险选择器"实用边界的诚实界定。
