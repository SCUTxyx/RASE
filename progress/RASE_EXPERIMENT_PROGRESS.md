# RASE 实验进展与结果(截至 2026-08-20)

> 服务器:AutoDL bjb2:28921(克隆实例)
> 本文档为唯一进展主表。v3 规划见 `RASE_ROADMAP_2026_08_19.md`。

---

## 1. 阶段总览

| 阶段 | 时间 | 结果 | 一句话 |
|---|---|---|---|
| 历史 R 系列 / K3 / K5 | ~08-18 | 见 §2.1 | 动作信号可学习(0.787/0.778);可学习≠可部署 |
| vFinal 冻结 | 08-19 | 完成 | 三文档 + Eligibility Screen + 两图 + sha256 |
| OFT 机会窗口 | 08-19 | PASS | 异构模型对互补 100%/0%,headroom +50pp |
| OPD 蒸馏链 | 08-19 | 部分 PASS | 风险可预测(校准 97.6%);查询表闭环 99.2%(=记忆) |
| **v3 P0(Gate A)** | 08-19 | PARTIAL | π0-fast 跨架构 PASS(0.837/0.07);goal/oft10 不可测 |
| **v3 P1(Gate B)** | 08-19 | PASS | same-root 反事实分叉(物体位姿 0.0795) |
| **v3 P2(Gate C)** | 08-19 | PASS | oracle 未来排序 0.9998 |
| **v3 P3(Gate D)** | 08-19 | FAIL | WM 摘要瓶颈 B2≈B1 |
| **W1(未来信息价值)** | 08-20 | PASS | 真实未来 2/3 折 +3.4~3.8pp |
| **W2(learned 未来)** | 08-20 | FAIL | 两次改进均无增益(RMSE 0.15 仍不够) |
| **闭环 v2(风险驱动仲裁)** | 08-20 | **FAIL** | 49.2%,0 切换,全部 abstain |

---

## 2. 历史阶段(摘要,详见旧版 PROGRESS)

- K3(8 tasks,432 slots):same-root pairwise 0.750;K5(48 tasks,864 slots):0.787;
- Semantic Selector:0.871/0.778;Deployment FAIL(fallback 非最优 0.7%);
- π0.5 ceiling(97.2%)、Dynamic route(确定性无 diversity)、libero_90 all-fail;
- Applicability Boundary IDENTIFIED(Goldilocks regime 条件量化);
- OFT 机会窗口:oft_spatial/oft_object 互补 100%/0%,headroom +50pp;
- OPD 蒸馏:校准 0.7→97.6%;查询表闭环 99.2%(明确为任务记忆,非 RASE)。

---

## 3. v3 实验链:Gate A(零样本跨 VLA,冻结 v3 模型)

### 3.1 设置
- 训练域:oft_spatial + oft_object(冻结风险模型 v3,特征 proprio+chunk+bigram+prior)
- 测试 VLA(全部未见):oft_goal、oft_10(同架构)、π0-fast(异构架构)
- 采集:20 任务 × 3 episodes,决策点行(与训练域同格式)

### 3.2 结果
| 测试 VLA | 行数 | 成功率 | AUROC | ECE | 判定 |
|---|---|---|---|---|---|
| oft_goal | 3900 | 0% | NaN(全负) | 0.432 | FAIL(不可测) |
| oft_10 | 3800 | 4.2% | 0.456 | 0.451 | FAIL |
| **π0-fast** | 802 | 87% | **0.837** | **0.066** | **PASS** |
| VLA-ID probe | — | — | **0.894**(随机 0.33) | — | 指纹泄漏 |

### 3.3 结论
- 跨架构(π0-fast)transfer 意外存在:discrimination + 校准保留;
- goal/oft10 不可测(近全败域);**注意:π0-fast 87% 高分域,信号可能部分来自
  "成功率水平"而非动作细节**;
- representation 保留 policy fingerprint(0.894)→ 身份无关未完全达成。

---

## 4. v3 实验链:P1/P2(P1 same-root, Gate B;P2 oracle future, Gate C)

### 4.1 Gate B:same-root 反事实机会(648 行,216 roots)
- 采集:同一 snapshot → oft_spatial/oft_object/oft_goal 各执行 64 步 → 未来轨迹
- 结果:动作多样性 0.83;proprio 未来分叉 0.42;物体位姿分叉 **0.0795**;
  within-state advantage 100%;h_within=0(异质在任务级)→ **PASS**
- 工程:horizon=8 不够(物体不动)→ 64 步;snapshot restore 需复位
  robosuite done/timestep

### 4.2 Gate C:oracle future 排序上限
- 真实未来(progress/drift/recoverability 三维)→ 排序候选
- **AUROC 0.9998**、pairwise 一致性 0.984、exact rank 0.953 → **PASS**
- 结论:完整未来含充足选择信号(未来路线的信息上限)

---

## 5. v3 实验链:P3(Gate D:WM MVP)+ W1/W2(未来信息)

### 5.1 Gate D:WM 摘要瓶颈 → FAIL
| 留出 VLA | B0 动作统计 | B1 直接 | B2 摘要未来 | B2>B1? |
|---|---|---|---|---|
| oft_goal | **0.968** | 0.944 | 0.946 | 否 |
| oft_object | 0.939 | **0.956** | 0.942 | 否 |
| oft_spatial | **0.968** | 0.951 | 0.961 | 否 |
- **跨 VLA 风险转移强(LOVO 0.94+,retention 0.90+)— 但均为 OFT 架构内**;
- 摘要版未来无增益 → 表示丢信息(非未来无用)。

### 5.2 W1:真实未来信息价值(选项 C)→ PASS
| 留出 VLA | B1 | B2_real(真实未来) | Δ |
|---|---|---|---|
| oft_goal | 0.944 | 0.982 | +3.8pp |
| oft_object | 0.956 | 0.990 | +3.4pp |
| oft_spatial | 0.951 | 0.881 | -7.1pp |
- 2/3 折显著增益 → 未来信息有价值,但增量温和(+3pp)且非全域

### 5.3 W2:learned 未来预测器 → FAIL
- 两次改进:22 维摘要、70 维轨迹采样;WM(MLP 128 隐层,2000 步)
- LOVO:learned 全面 ≤ B1(-0.4~-6pp);WM RMSE 0.15 仍破坏排序信号
- **结论:信息在真实未来里,但 learned 无法保留(排序信号对误差极敏感);
  视频 WM 预测误差更大,大概率更差;增量上限仅 +3pp 配不上成本**

---

## 6. 闭环 v2:风险驱动仲裁(2026-08-20)→ FAIL

### 6.1 设置
- risk:same-root 全数据训练 B1(状态+动作+bigram,身份无关),AUROC 0.98
- 闭环:每决策点(8 步)双候选(oft_spatial/oft_object)生成 → risk 打分 →
  LCB/UCB + abstain(阈值 0.05)→ 执行;20 任务 × 6 trials

### 6.2 结果
| 指标 | 值 |
|---|---|
| 总成功率 | **49.2%(59/120)** |
| spatial 任务 | 59/60(默认 oft_spatial 主场) |
| object 任务 | **0/60**(从未切换到 oft_object) |
| 切换次数 | **0**;abstain 1455 次 |

### 6.3 诊断(结构性)
- 同状态双候选生成的早期动作 chunk 特征**无法区分适配性**:
  risk 训练时见过"各自轨迹上的 chunk"(状态不同),部署时是"同状态双生成"
  (分布 gap)→ 分数几乎相同 → 全部 abstain
- 与查询表(99.2%,任务记忆)对比:**风险驱动选择在 LIBERO 同状态场景不可行**

### 6.4 对比表
| 方法 | 成功率 | 机制 |
|---|---|---|
| best-fixed(单模型) | 50% | — |
| risk 驱动闭环 | 49.2% | 风险打分(失败) |
| 查询表 | 99.2% | 任务记忆(非 idea) |
| oracle | ~100% | 上限 |

---

## 7. 数据与代码资产

| 资产 | 位置 |
|---|---|
| same-root 数据集(648 行,含 future 轨迹+物体位姿) | `runs/oft_opportunity/same_root_w1.jsonl` |
| 零样本分析 | `runs/oft_opportunity/zero_shot_vla_analysis.json` |
| Gate B/C/D/W1/W2 报告 | `runs/oft_opportunity/future_divergence_big_report.json`、`oracle_future_report.json`、`wm_mvp_v4_lovo_*.json`、`w1_diagnostic.json`、`w2_report_v2.json` |
| 闭环 v2 | `runs/oft_opportunity/closed_loop_v2.json` |
| same-root risk 模型 | `runs/oft_opportunity/same_root_risk.npz`(+LOVO 变体) |
| 脚本 | `scripts/`(服务器)+ `RASE_analysis/`(本地) |

---

## 8. 当前科学状态(诚实总结)

| 主张 | 状态 |
|---|---|
| same-root 反事实协议(候选未来分叉) | ✅ 成立(Gate B/C) |
| 风险预测 OFT 架构内跨 VLA 转移 | ✅ 成立(LOVO 0.94+) |
| 风险预测跨架构转移(π0/SmolVLA) | ⚠️ **仅一个证据**(π0-fast 0.837),未确认 |
| 未来信息有边际价值 | ✅ +3pp(2/3 折,W1) |
| learned 未来(WM)捕获增益 | ❌ 失败(W2) |
| **风险驱动闭环选择** | ❌ **失败**(49.2%,同状态特征无法区分候选) |
| 身份无关(无指纹) | ⚠️ probe 0.894,未达成 |
| 任务记忆(查询表) | ✅ 99.2%(非 idea,仅对照) |
