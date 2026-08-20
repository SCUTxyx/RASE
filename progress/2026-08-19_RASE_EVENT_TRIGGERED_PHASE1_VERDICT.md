# RASE Event-Triggered Phase 1 判定:Detector 诊断 FAIL(2026-08-19)

## 一句话结论

> **Steering FAIL(决定性:detector 无稳定触发工作点)**。动态边界的 opportunity
> 真实存在(正控 `object_001041` 在动态边界 heterogeneous:fallback 0/3、
> requery 2/3),但基于动作/状态的停滞-相位检测在 π0-fast 上**不存在可稳定
> 校准的信号**:stagnation 阈值在 0.006(永不触发)与 0.012(过早误触发)之间
> 无工作点。按计划:停止扩大;剩余唯一未测信号为 disagreement(M=4 proposals,
> 高成本),另行决策。

## 1. Phase 0 完成项

- `rase/vnext/schema.py`:`BoundaryTriggerProvenance` + `TimingComponents`
  (rule/scores/threshold/trigger_step/timestamps;timing 分量求和容差校验);
- `rase/vnext/boundary.py`:低容量 causal detector(phase / stagnation /
  disagreement,combined 预注册规则,物理位移 + proprio 差分);
- `scripts/collect_rase_vnext_discovery.py`:`prefix_to_decision` 泛化为
  detector 模式(decision_step=None),复用 RNG/queue isolation、snapshot
  restore、native-chunk 对齐;prefix timing 分解
  (`source_prefix_{inference,env,total}_wall_s`);
- `scripts/audit_rase_vnext_timing_contract.py`(缺失率/非负/单调/分量和/
  prefix hash 一致性);
- 修复:proprio 嵌套 dict 提取(`robot_state.joints.pos + gripper.qpos`);
  正控/对照 roots 冻结(goal_000625、object_001041 + 3 个 fb-dominant)。

## 2. Detector 校准诊断(三次迭代,预注册允许的一次诊断)

| 迭代 | 信号 | 结果 |
|---|---|---|
| 1 | 归一化动作 norm | π0-fast 动作 norm≈1.0、从不 <0.2 → **无停滞语义** |
| 2 | 物理位移(×0.05m/0.5rad) | 位移 0.04-0.05m/步,仍无停滞 |
| 3 | proprio 差分(校准分布:p10≈0.01、median≈0.03、最小 5 步窗≈0.007-0.010) | 阈值 0.012 → **step5 过早误触发**(3/5 roots);0.006 → **永不触发**;区间内无稳定工作点 |

**关键发现**:π0-fast 的 proprio 变化是连续分布,不存在"停滞 vs 正常"的
清晰分隔——**该 policy 在该 benchmark 上没有可检测的在线停滞/相位模式**。

## 3. Smoke 结果(v3 触发版 / v4 收紧版)

### v3(阈值 0.012)

| root | 边界 | rule | verdict | 候选 success |
|---|---|---|---|---|
| **object_001041(正控)** | **5** | combined | **heterogeneous** | cont 1/3、**fb 0/3**、req 2/3 |
| goal_000625(正控) | 54 | combined | fb-dominates | cont 3/3、fb 2/3(非最优)、req 2/3 |
| goal_001075(对照) | 5 | combined | all_fail | 全 0/3(过早触发) |
| spatial_000610(对照) | 5 | combined | all_fail | 全 0/3(过早触发) |
| spatial_000365(对照) | 56 | combined | fb-dominates | — |

### v4(阈值 0.006)

全部 rule=none、boundary=80(goal/object 已确认,spatial 未完成即会话退出)。

## 4. Steering 判定(预注册条件)

| 条件 | 结果 | 判定 |
|---|---|---|
| dynamic heterogeneity ≥8% 且 ≥2× static | v3:1/5=20%(但依赖误触发边界) | ⚠️ |
| 正控重发现 | object_001041 ✓(1/2);goal 部分(fb 非最优) | ⚠️ |
| all-fail 不显著抬高 | v3:2/5=40%(early trigger) | ❌ |
| **detector 稳定触发** | v4 全部无触发;阈值无工作点 | ❌ **决定性** |

**判定:FAIL**——不是 opportunity 缺失(正控 heterogeneous 已复现),而是
**在线触发信号在该 policy/benchmark 上不可用**。

## 5. 按计划的处理

- 停止 Phase 2 confirmation(不扩大、不训练 selector、不加 K);
- 剩余唯一未测信号:**disagreement(M=4 source proposals)**——若未来执行,
  需接受 4× 推理成本;预注册其单独 Gate(触发率 ≥X% 且边界 heterogeneous
  率 ≥8%);
- 或转向 harder/contact-rich/long-horizon domain(outcome-blind 两阶段冻结);
- 动态边界协议代码(Phase 0)保留为可复用基础设施。

## 6. 保留结论(不变)

- 静态域结论不变:Opportunity Gate FAIL(0.9%-2.1% heterogeneous);
- 动作信号可学习性不变(π0-fast 0.871/0.778);
- **新增**:正控 opportunity 在动态边界可复现(非噪声),但需要**其他在线
  信号**(disagreement / 视觉 / 更晚边界)才能被检测器捕获。
