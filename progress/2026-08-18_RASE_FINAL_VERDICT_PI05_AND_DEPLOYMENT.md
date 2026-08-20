# RASE 部署诊断 + π0.5 Challenge 最终结论(2026-08-18)

## 一、四格结论(最终)

| 问题 | 判定 | 依据 |
|---|---|---|
| **动作信号跨 policy?** | **否(policy-dependent)** | π0-fast Signal PASS(0.871/0.778);π0.5 in-policy 0.421(19 pairs,CI 跨 0.5)→ **ceiling effect**:π0.5 在 K3 roots 上 continue/requery/fallback 成功率全为 97.2%,结局差异不可观测 |
| **模型跨 policy?** | **不成立(inconclusive 弱正)** | π0-fast 冻结模型在 π0.5 上 0.526(CI [0.491, 0.519] 跨 0.5);转移目标在 π0.5 上几乎无失败可区分 |
| **当前场景可部署?** | **否** | Deployment-v2 S0 FAIL:selector 0.889 < fallback 0.951;fallback 非最优仅 1/144(0.7%) |
| **受约束场景可部署?** | **否** | S1 无空间(oracle gap 0.7%)、S2 反事实 selector 0.556 < quota-fallback 0.604 |

## 二、B 线(π0.5 Challenge)详细结果

### B0 parity smoke:PI05_SMOKE_PASS
- checkpoint/loader/1-group/6-ops 全过;prefix_available=True;
- **resample.source/candidate.0/.1 = executable(有 native diversity,32/32 首动作不同)**
  ——与 π0-fast 的 incapable_missing 形成直接对比(工程层政策差异);
- capture v2 audit PASS。

### B1 冻结
- `runs/rase_vnext/frozen/pi05_challenge_manifest_v1.json`
  (sha256 `a48dd90f…`,432 jobs/24 roots/8 tasks,复用 K3 roots,配对比较)。

### B2 收集
- 72/72 groups COMPLETE;432 rows **全部 available**(π0.5 无 incapable);
- per-operator success:continue 97.2% / requery 97.2% / fallback 97.2%;
- resample:选中的 candidate 被执行(32×candidate.0 + 40×candidate.1)。

### B3 Gate
- **policy-signal:INCONCLUSIVE(weak/ceiling)**——0.421(19 informative pairs),
  task bootstrap CI [0.477, 0.505] 跨 0.5;非"信号缺失",而是 π0.5 高成功率
  饱和(97.2%)导致同 root 结局差异不可观测;
- **cross-policy transfer:INCONCLUSIVE(weak)**——0.526,CI [0.491, 0.519] 跨 0.5;
- 按规划:小样本 CI 跨 0.5 不强行 PASS/FAIL;
- 结论收窄:**动作信号为 policy-dependent**——π0-fast 有可学习信号,π0.5 在
  该 cohort 上无可观测的结局差异(其价值上限被 policy 自身成功率决定)。

## 三、A 线(部署诊断)最终判定

- **Deployment-v2 FAIL(全部预注册场景)**,详见
  `progress/2026-08-18_RASE_DEPLOYMENT_V2_VERDICT.md`;
- oracle 上界:真实改进空间 0.7%(fallback 非最优 1/144);
- **关闭 learned-selector 部署主线**;不再调 margin/model;不做 closed-loop。

## 四、科学结论总账(可写入口径)

1. **动作反事实信号可学习性成立(π0-fast)**:48-task task-held-out 0.871 +
   K3 外部确认 0.778,source-source 0.706/0.75;
2. **信号是 policy-dependent**:π0.5 上因成功率饱和不可观测(inconclusive);
3. **可学习 ≠ 可部署**:fallback 主导场景下 selector 无 break-even,全部预注册
   部署场景 FAIL;
4. **工程链路完整可复用**:E0 capture 协议(v2)、双模型 selector 模块、成本
   账本、场景化评估——任何未来 VLA/任务变化可直接重跑。

## 五、最终决定(按协议停机规则)

- 关闭 learned-selector 部署与 closed-loop;
- 不启动 OPD / RL / world-model / universal pooled selector;
- 可选后续(低优先):
  1. 在**高失败率 policy/任务**上(如 π0-fast 的 Object/Long)验证 selector 的
     条件价值——若改进空间仍 <1%,则彻底关闭;
  2. RASE 定位收敛为"动作反事实信号审计 + capture 协议"科学工具。

## 产物清单

- `progress/2026-08-18_RASE_DEPLOYMENT_V2_VERDICT.md`(sha256 b13fdd3e)
- `progress/2026-08-18_RASE_DEPLOYMENT_DIAG_AND_PI05_CHALLENGE_PLAN.md`
- `runs/rase_vnext/frozen/pi05_challenge_manifest_v1.json`(a48dd90f)
- `runs/rase_vnext/frozen/selector_dataset_pi05_manifest_v1.json`(86f1280d)
- `runs/rase_vnext/pi05_challenge_analysis_v1.json`
- `runs/rase_vnext/frozen/cost_ledger_v1.json`(28ee4ad4)
- `runs/rase_vnext/deployment_feasibility_v1.json`、`deployment_oof_v1.json`
- 代码:`scripts/{freeze_rase_vnext_pi05_cohort,run_rase_vnext_pi05_collect,analyze_rase_vnext_pi05,build_rase_vnext_cost_ledger,analyze_rase_vnext_deployment_feasibility,run_rase_vnext_deployment_oof}.py`
