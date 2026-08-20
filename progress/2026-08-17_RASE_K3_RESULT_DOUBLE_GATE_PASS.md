# RASE K3 正式结果:双 Gate PASS(2026-08-17)

## 1. 一句话结论

> K3 独立动作信号 pilot **双 Gate PASS**(K3-SIGNAL-PASS + K3-UTILITY-PASS):
> 在 8 tasks × 3 roots × 6 operators × K3 = 432 slots(360 口径)的冻结 cohort 上,
> same-root pairwise ranking 中 **动作特征相对 state-only 增益 +0.55**(4/4 folds 同向、
> task bootstrap 下界 > 0),离线 selector 相对 continue 基线 utility +0.24、
> oracle regret 下降 48%。这为"同一物理状态下动作轨迹携带可学习风险信息"提供了
> 第一个 task-held-out 的正向证据,解锁小规模 semantic selector 设计阶段。

## 2. 冻结产物(全部 sha256 存档)

| 产物 | 路径 | sha256 |
|---|---|---|
| K3 cohort manifest | runs/rase_vnext/frozen/k3_cohort_manifest_v1.json | 6971fc3d4461fb2061e09786249125785e54ad8d5af514bd28e1f67966150b4c |
| K3 收集 | runs/rase_vnext/k3_collect_v1/(72 groups、72 captures、branches.jsonl) | branches 2e7d3607ec069744067655933bd598a3d84b4376c72d41d7b51b8f54d19d7d7a |
| K3 分析 | runs/rase_vnext/k3_collect_v1/k3_analysis.json | — |
| E0 smoke v3 | runs/rase_vnext/k3_e0_native_capture_smoke_v3/E0_VERDICT.json | — |

## 3. K3-CAPTURE-PASS(工程审计)

- 432/432 jobs 完成,0 missing / 0 duplicate / 0 corrupt;
- 72 个 capture 文件(v2 schema)全部通过 `audit_candidate_capture`(raw→canonical→raw 误差 0、alignment 全 True);
- capability 分布(数据驱动,非事后改判):
  - continue.source / requery.source / fallback.persistent:72/72 executable(含完整
    inference-event provenance、queue cursor);
  - resample.source/candidate.0/.1:72/72 incapable_missing(π0-fast 两候选 bitwise
    相同,确定性 capability record,非普通 failure);
  - abort.safe:72/72 control_only_abort;
- 全程无 queue 反推(chunk_origin 全部为 inference_event / forced_inference /
  executed_trace / control_only / contract_mask);
- simulator executions = 216(continue+requery+fallback;resample 无 diversity 不执行,
  abort 不执行;capability mask 不从分母剔除)。

## 4. K3-SIGNAL-PASS(主证据)

same-root pairwise ranking(task-held-out ridge OOF,8 tasks 固定 4 folds):

| 特征 | OOF pairwise acc | gain vs state-only |
|---|---:|---:|
| state-only(proprio) | 0.026 | — |
| raw-action | 0.576 | **+0.550** |
| trace-only(CanonicalMotionTrace) | 0.548 | +0.522 |
| trace+semantic | 0.534 | +0.508 |

- **4/4 folds 方向一致**(≥3/4 门槛);
- task bootstrap mean gain +0.548,95% CI [0.358, 0.718](下界 > 0);
- 8/8 tasks 增益方向为正;
- action-swap(原 OOF 模型 + 仅交换动作特征):swap sensitivity 95.3%,pairwise
  accuracy 0.806→0.801(不退化,swap coverage 全 executable 候选);
- 校准相对 state-only 改善:ΔBrier -0.174、ΔECE -0.052(≤0.02/0.05 门槛)。

**解读**:同 root 内 proprio 完全相同,state-only 无区分力(0.026);加入动作特征后
ranking 跃升至 0.55+。这是"动作本身携带 outcome 风险信息"的直接证据。

## 5. K3-UTILITY-PASS(离线选择)

| 策略 | mean utility | 说明 |
|---|---:|---|
| continue baseline | 0.458 | 始终继续 source |
| 离线 selector(frozen 阈值) | 0.700 | +0.242 vs continue |
| always fallback | 0.858 | fallback 本身很强(高成功、高成本) |
| oracle | 0.964 | 上界 |

- selector gain vs continue +0.242 ≥ 0.05 ✓;
- oracle regret reduction 47.9% ≥ 10% ✓;
- **注意**:selector < always-fallback(0.700 < 0.858)。selector 的价值在成本敏感
  场景(fallback 成本权重 0.1),且当前 fallback 成功率极高(129/216 总 success 主要
  来自 fallback)。这不是 gate 失败,但意味着 selector 的净收益需在成本/成功率
  联合评估下成立——写入后续 semantic selector 设计。

## 6. 限制(口径边界,不得逾越)

- 单策略(pi0-fast)、8 tasks、单 decision point(source.step.8);不构成 multi-VLA
  泛化、real-time closed-loop 或 D-PASS 证据;
- resample 在 π0-fast 无 native diversity(已冻结 capability);
- swap 敏感性高但 ranking 变化小:模型捕获动作分布/operator 级差异为主,
  具体任务语义证据仍有限;
- K3 pilot gate 是"继续/停止"门槛,不是论文显著性标准;
- 历史 Phase-C 仍为 B-FAIL(1/288 不复现),与新 K3 cohort 独立。

## 7. 按协议解锁的下一步(semantic selector 设计,低容量 ladder)

```text
E0: prior(operator/状态先验)
E1: state/history
E2: raw action statistics
E3: CanonicalMotionTrace
E4: trace + frozen visual history
```

- 每级为独立信息 gate,linear/ridge + task-held-out,超参 inner-fold 选择;
- 目标:same-root ΔU(continue, requery, fallback)回归,tie margin 0.01 内 abstain;
- 仍需:至少 32 informative tasks(C2 gate 口径)再谈规模化;当前 8 tasks 只有
  阶段意义;
- **不启动**(直到更高 gate):OPD、RL、world model、multi-VLA pooled selector、
  大规模公开数据下载、实时闭环部署。

## 8. 建议的下一批执行

1. 将 K3 分析固化(已存档 k3_analysis.json),写一页 gate decision 至 progress;
2. 扩大 cohort 至 48 tasks(K5 协议,复用本修复的 capture 链路)——或先做
   8-task 的 E2/E3 信息 ladder 快速确认特征可迁移性;
3. π0.5 独立 opportunity challenge(不能迁移 π0-fast 的 D0/K3 证据);
4. 上述任一 PASS 后再冻结 semantic selector 训练协议。
