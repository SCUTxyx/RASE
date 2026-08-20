# Experiment progress records

- [CVPR 2027 canonical document refresh and execution overlay (2026-08-13)](2026-08-13_cvpr2027_document_refresh_and_execution_overlay.md):
  新版 canonical idea、主执行计划和 RL/公开数据专题计划已替换旧入口；冻结实验记录不变。
  当前只解锁 R10-B per-chunk root-cause diagnostic，所有 learned stages 仍锁定。
- [R6-B0 multi-VLA OOF result (2026-08-10)](2026-08-10_r6b0_multivla_oof_results.md):
  R6-A found two valid policy-pair opportunities, but the 48-task one-boundary
  learner fails the strict 5-seed per-VLA gate. A deployable-language diagnostic
  improves Pi0Fast AUROC but not stability. R6-B1 must collect grouped dynamic
  source-trajectory boundaries before any world-model or validation claim.
- [R6-B1 dynamic collector parity smoke (2026-08-10)](2026-08-10_r6b1_dynamic_collector_parity.md):
  Pi0Fast and Pi0.5 each exactly reproduce their frozen R6-A trajectory before
  counterfactual labels are retained. A cross-suite outcome-balanced pilot is
  the next gate.

This directory is tracked. It contains concise, reviewable records of completed
or blocked experiments; large logs, videos, checkpoints, and raw outputs belong
under ignored artifact directories.

## Start here

- [Current RASE status](CURRENT.md):
  R10-B deterministic case/control reproducibility 正式 FAIL；full-action trace 已定位为
  closed-loop divergence。当前服务器正在运行冻结的 18-group/K3 per-chunk input/output
  根因诊断。risk model、selector、RL、world model、validation 与 test 均未解锁。
- [R5-B24 result and R6 direction (2026-08-09)](2026-08-09_r5_b24_results_and_r6_direction.md):
  B24 完成 24 states / 122 boundaries / 610 continuations，protocol **READY**，但
  opportunity **NOT READY**：仅 4 个 finite-safe，真正 recovery-created 仅 **1/24**，
  0 个 populated positive stopping bins；历史二元与 all-K 标签一致率 60.66%，11/24
  sampled non-monotonic。正式停止 safe-handback 模型、第二 VLA/world-model/test 继续
  关闭。QC71 source-risk privileged 上界保持 persistent 65/71 成功但仅节省 21.89%，
  需捕获 91.35% oracle savings 才达 20%，故先做多 policy-pair 的 model-free R6 screen。
- [R5-B24 paired probability opportunity lock (2026-08-09)](2026-08-09_r5_b24_paired_opportunity_lock.md):
  冻结24个train task各1个state（21 finite-safe + 3 persistent-failure support）、
  h={0,16,32,64,96,128}、K=5和跨boundary common random numbers。只有live
  finite-safe≥20、≥3 tasks、≥2 populated bins、conservative oracle savings≥25%
  才重新训练；否则第二VLA/world-model/test继续关闭。
- [R5-A16 probability labels and five-seed OOF result (2026-08-09)](2026-08-09_r5_a16_probability_and_oof_results.md):
  A16 协议 gate **READY**（16/16 parity、56 boundaries、280 repeats），但 conservative
  oracle savings 仅 **11.39%**；标签 13/56 非退化、4/16 sampled non-monotonic 但
  0 个 Wilson-separated drop。Beta-binomial 多头 5-seed OOF 为 **0/5 PASS**：gap
  −6.25 至 −16.25pp、false handback 6.25–16.25%、savings 9.19–27.85%。第二 VLA、
  world-model、validation/test 全部保持关闭，下一步回到独立 opportunity-enriched
  development cohort。
- [R5 probabilistic controller execution plan (2026-08-09)](2026-08-09_r5_probabilistic_controller_execution_plan.md):
  冻结 A16→Beta-binomial 多头→固定 task folds 的 5-seed OOF→第二 VLA→世界模型
  Pareto 消融→独立 validation/test 的执行顺序；补充单类 head fail-closed、旧 val 已转为
  development、世界模型 AUC-only 不得入主模型等约束。
- [R5-A16 probability-label protocol lock (2026-08-09)](2026-08-09_r5_a16_probability_protocol_lock.md):
  冻结 16 个 fresh development states、4 boundaries、K=5、最多 320 continuation；A16
  只测标签熵/非单调性和协议稳定性。persistent parity、repeat 完整性、四 suite/八 task
  覆盖全部通过后才训练 Beta-binomial 多头模型；5-seed OOF 必须至少 4/5 同时满足
  success gap≥−5pp、false handback≤5%、savings≥20%，否则不开第二 VLA。
- [R5 probabilistic smoke4 and RASE idea rewrite (2026-08-09)](2026-08-09_r5_probabilistic_smoke4_and_idea_rewrite.md):
  主 idea 已改写为多 VLA、策略条件、概率风险/恢复/交还控制。四套件 same-snapshot
  K=5 pilot 的 protocol gate 通过：4 states、8 boundaries、40 continuations，3/8
  boundaries 为非退化概率；Spatial h0/h64 为 **4/5→0/5**，Goal 为 **3/5→4/5**，
  Object/Long 均 **0/5→0/5**，全部 LCB<0.5。故推进 16-state 概率标签 pilot，模型训练、
  frozen test 和大规模闭环继续关闭。
- [R5 seed stability and validation-boundary repeatability (2026-08-09)](2026-08-09_r5_seed_stability_and_val_repeatability.md):
  修正协议 5-seed 为 **0/5 gates pass**；AUC 0.922--0.932 但 success 仅
  **81.69--85.92% vs persistent 91.55%**，38.0% states 的交还时刻随 seed 变化。
  真 val smoke persistent parity 4/4，但历史/新标签仅 22/27 一致；新协议重测为
  25/26 一致，Goal 仍出现标签/轨迹漂移；same-snapshot K=5 又得到 h0 3/5、h64
  5/5（LCB 仅 0.649）。故暂停单次二元标签的 full-val，转概率标签、扩充独立
  calibration cohort，test 继续封存。
- [R4-D protocol re-audit and corrected delta results (2026-08-09)](2026-08-09_r4d_protocol_reaudit_and_delta_results.md):
  原 0.6M checkpoint 的工程导出仍有效，但 OOF fold 复用、state-level stopping 计算、M4
  全量重叠、M5 success/bootstrap 公式及教师缓存配对均存在协议错误。修正版 hard-label
  baseline 为 AUC **0.9221**、success **84.51% vs persistent 91.55%**、savings
  **21.09%**，安全 gate **FAIL**；精确 V-JEPA delta 也未改善 Pareto，主线转为扩数据与
  conservative calibration。
- [PRE-C0-R2 gate scaling result (2026-08-08)](2026-08-08_pre_c0_r2_gate_scaling_results.md):
  62 task-disjoint spatial-train snapshots give `H_activation=4.84pp`; grouped OOF gate has
  no threshold satisfying rescue/harm constraints, so learned-gate dev eval is stopped.
  Envelope-only is 31/40 vs B0 27/40 and legacy bounded F0 32/40; gate branch is `NO-GO`.
- [RASE full project narrative (2026-07-31)](2026-07-31_rase_full_project_narrative.md):
  **单文件总览**：原计划、各阶段实验、计划转化、W9C/W10 结论、可/不可主张、后续选项。
  新读者应先读此文，再下钻各次实验记录。
- [PRE-C1.2 kickoff (2026-08-05)](2026-08-05_pre_c1_2_kickoff.md):
  C1.1 zero recovery → successor / same-H sweep / student-query DAgger / native prefix flow；
  protocol `artifacts/pre_c1/pre_c1_2_protocol_lock.yaml`。

## Recorded baselines

- [Frozen SmolVLA clean LIBERO, seed 0, `n_action_steps=10`](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md):
  70.0% mean success over four suites and 2,000 episodes.
- [ForkableEnv W1 gate, 2026-07-17 11:36 CST](2026-07-17_forkable_env_w1_gate.md):
  4/4 fork integration tests passed (snapshot/restore deterministic).
- [W1 dry-run gates, 2026-07-17 11:46 CST](2026-07-17_w1_dry_run_gates.md):
  state-pool dry-run (26 states, idempotent resume) + collapse smoke dry-run (40 pending tasks).
- [SmolVLA LIBERO-Plus collapse smoke, seed 0, nas10](2026-07-17_smolvla_libero_plus_collapse_smoke_nas10.md):
  40 tasks × 1 ep → **2.5%** mean (camera 5% / robot 0%); qualitative only, not the paper curve.
- [NGC Step-1 real collect preflight + pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md):
  SmolVLA+Plus adapter; preflight 2 ep + pilot 20 ep (1/20 success, **367** states); camera/robot only.
- [SmolVLA LIBERO-Plus collapse full, seed 0, nas10](2026-07-18_smolvla_libero_plus_collapse_full_nas10.md):
  3142 completed + 7 skipped → **0.38%** mean (camera **0%** / robot **0.78%**).
- [NGC Step-1 scale200 camera-heavy pool](2026-07-18_ngc_step1_scale200_camera_heavy.md):
  199 ep / 3666 states；pool fork gate 已过。
- [NGC W2 K=8 candidates pilot](2026-07-18_ngc_w2_candidates_pilot.md):
  2 states → `[8,10,7]` artifacts；mean endpoint L2 **1.69**（多样性通过；未测续完成功率）。
- [NGC W3 research-grade continuation pipeline](2026-07-19_ngc_w3_pipeline.md):
  旧 16-state SmolVLA **16×Set C / 0/768**；OFT spatial **8/32** / libero_10 **0/32**；`min_remaining_steps` 协议已落地。
- [NGC W3 continuation-temperature ablation](2026-07-19_ngc_w3_cont_ablation.md):
  OFT 阳性态上 SmolVLA 四温 **0/192** vs OFT **8/8** → 续完能力缺口，非温度/判据假阴性。
- [NGC W3 ADEQUATE-only pilot](2026-07-19_ngc_w3_pilot_adequate.md):
  ADEQUATE 16 态 SmolVLA 仍 **16×C / 0/768**（0 NARROW）；OFT 四 suite：spatial **7/32**，object/goal/10 **0/32**。
- [NGC W4 ADEQUATE scale + v2 dual-oracle](2026-07-26_ngc_w4_adequate_scale.md):
  ADEQUATE 32 态 SmolVLA **32×C / 0/1536**；OFT portfolio **17/32**（候选命中 **65/256**）；交叉全为 `oft_only`/`both_fail`；采样全 `t0=0`。
- [NGC W5 SmolVLA screen t=0.7](2026-07-27_ngc_w5_smol_screen_t07.md):
  OFT-recovered smoke **0/136**、failure frontier **0/192**；**Confirm 未跑**；0.3/1.0 与 L1–L2 仍可选。
- [NGC W5 proposal-temperature sweep](2026-07-27_ngc_w5_temperature_sweep.md):
  t=0.3/0.7/1.0 在同 24-state frontier 上合计 **0/576**；候选温度与多样性已核验，
  proposal-temperature 诊断线关闭，下一步转 L1–L2 paired policy pilot。
- [NGC W6 L1–L2 collection and coverage](2026-07-27_ngc_w6_l1_l2_coverage.md):
  40 episodes / 762 states，SmolVLA outcome 为 0 success / 40 failure；冻结
  8-state episode-distinct failure-challenge cohort，clean-success control 后续单列。
- [NGC W6 L1–L2 paired policy matrix](2026-07-28_ngc_w6_l1_l2_policy_matrix.md):
  最终矩阵为 Smol **0/64 candidates、0/8 states**，OFT **10/64、2/8**；
  exact McNemar `p=0.5`，只支持方向性 pilot 结论。
- [W7 causal attribution and held-out validation](2026-07-29_w7_causal_heldout.md):
  W6 OFT-only 的 candidate-specific rescue 为 **0/2**；冻结 episode-disjoint
  24-state W7，Smol held-out 完整为 **0/192、0/24 states**，OFT/归因后台流水线运行。
- [W8 lightweight selector + direct escalation arm](2026-07-29_w8_lightweight_selector_direct_arm.md):
  三动作 ridge baseline、严格 readiness gate 与 direct-OFT 24-rollout 已完成。
- [W8 direct escalation result](2026-07-29_w8_direct_escalation_results.md):
  W7 Smol **0/24** vs prefix+OFT **8/24**（McNemar `p=0.0078125`）；W8
  direct OFT **9/24**，成功集中于 Goal/Long；下一 gate 为逐状态配对与 clean controls。
- [W9 clean controls + direct-policy selector gate](2026-07-29_w9_clean_selector_pipeline.md):
  W7/W8 overlap 为 both 7 / prefix-only 1 / direct-only 2 / neither 14；修正
  utility、direct Smol 与特征泄漏后，冻结 32-state clean-control pilot。
- [W9 clean-control coverage gate](2026-07-29_w9_clean_control_coverage_gate.md):
  预注册 140 ep 后 coverage **失败**（exit 2）；pool **10/138≈7.2%** success，
  几乎全在 Spatial；Object/Goal/Long early+mid 全空；selector **未训**。

- [W9C clean task-identity fix](2026-07-31_w9c_clean_task_identity_fix.md):
  Plus 0–9 与 official clean-10 不同；改用 exact-name vanilla BDDL/init 后 probe **PASS**
  （mean SR 0.6125），并完成 clean32 coverage 与 selector readiness。
- [W9C selector gate result](2026-07-31_w9c_selector_gate_result.md):
  episode/task readiness 均 **PASS**，但 ridge vs action-matched random 在两个 held-out
  split 上 Δutility 均为 **0**；触发 `kill_method_branch`，不上 MLP/RL。
- [Paper claim freeze](2026-07-31_paper_claim_freeze.md):
  主投姿态冻结为 **benchmark / diagnosis**；允许 policy-relative recoverability、direct
  escalation 与 candidate-specific rescue 否证，禁止 learned-selector claim。
- [Idea evolution and next questions](2026-07-31_idea_evolution_and_next_questions.md):
  汇总证据演化、已否证假设、claims 边界、task-held-out 8-clean/0-failure/0-escalation
  限制，以及下一阶段 benchmark 问题。
- [W9C benchmark reanalysis result](2026-07-31_w9c_benchmark_reanalysis.md):
  **COMPLETED（CPU-only）**；56-state composition/oracle/shortcut/LOSO 审计完成。
  Overall oracle-minus-fixed gaps 为 abstain **0.4729**、continue **0.2071**、escalate
  **0.0907**；分析 JSON 无 learned-action annotation，结果不撤销 W9C ridge kill。
- [W10 Object/Spatial failure benchmark](2026-07-31_w10_object_spatial_benchmark.md):
  独立 Object/Spatial failure cohort 完成；collect **80/80 failure**，direct Smol
  **0/16**，direct OFT **1/16**（Spatial 0/8）；escalate oracle 仅 1 态，episode split
  **`NOT_READY`**。结论：该 regime 下几乎无 policy-relative recovery，不开放 selector。
- [RASE UI Phase 0B 48-state opportunity screen](2026-08-01_rase_ui_phase0b_opportunity12_results.md):
  三算子完整矩阵 C/R/O = **9/13/30 of 48**，OFT 覆盖全部 Smol success，
  success-only oracle gap **0.0**（cluster CI `[0,0]`），因此 selector/world model 仍不开闸；
  探索性 cost sweep 定位到 utility-aware benchmark 的下一验证方向。
- [RASE UI Phase 0C kickoff](2026-08-01_rase_ui_phase0c_kickoff.md):
  冻结“物理成本校准 → 弱扰动边界 calibration → 独立 96-state screen”的执行顺序与
  stop rules；在新 gate 通过前继续禁止 selector/world-model training。
- [RASE UI Phase 0C balanced factorial16 result](2026-08-01_rase_ui_phase0c_factorial16_results.md):
  64-state 完整矩阵 C/R/O = **19/14/38**，success oracle **39/64**，相对 always-OFT
  仅 **+1/64**（episode bootstrap CI `[0, 0.046875]`）；REPLAN-only 为 0，gate
  **NOT_READY**。已补物理推理计时与 success-supported cost-winner 审计。
- [RASE UI Phase 0D no-world-model + timing kickoff](2026-08-01_rase_ui_phase0d_no_world_model_timing_kickoff.md):
  冻结当前阶段**不训练生成式 world model**；先用 episode-independent 16-state cohort
  补齐 Smol action-selection 计时，并与已有 OFT RPC 计时分范围报告。selector/outcome
  model/world-model gate 继续关闭。
- [RASE UI Phase 0D timing16 result](2026-08-01_rase_ui_phase0d_timing16_results.md):
  16/16 同状态计时覆盖通过；OFT 单次 RPC 较慢，但 action-chunk 摊销后 policy time
  为 **16.48 ms/env-step**，低于 CONTINUE **29.39** 和 REPLAN **30.09**。因此不能用
  latency 为 post-hoc `OFT cost=0.10` 背书；下一步转 deferred-switch operator screen。
- [RASE UI Phase 0E deferred-switch kickoff](2026-08-01_rase_ui_phase0e_deferred_switch_kickoff.md):
  代码审计确认旧 `full` prefix 来自 candidate artifacts，不能冒充 active suffix；冻结
  新 `decision-suffix` 合约、prefix SHA/步数 parity gate 与 16-state 语义校准，继续禁止
  selector/outcome-model/world-model training。
- [RASE UI Phase 0E/0F deferred-switch result](2026-08-01_rase_ui_phase0e_phase0f_deferred_switch_results.md):
  立即/延迟 OFT 为 **10/16 vs 9/16**，同状态 oracle **11/16**（相对最佳固定策略
  **+1/16**）；出现 2 个 immediate-only 与 1 个 deferred-only，3/3 确定性重放及
  active-suffix SHA 完全一致。已证实切换时机异质性，但独立 opportunity gate 未通过，
  下一步是 task/episode-disjoint Phase 0G screen，仍不训练 selector/world model。
- [RASE UI Phase 0G independent48 kickoff](2026-08-01_rase_ui_phase0g_independent48_kickoff.md):
  预注册 48-state task/episode-disjoint timing screen；冻结 metadata-only seed、48 个唯一且
  与 Phase 0C–0F 零重叠的 task、三臂 exact join、双向 timing-specific task 支持与
  `oracle gap >= 0.05` gate。screen 前后身份不一致即停止，仍不训练模型。
- [RASE UI Phase 0G independent48 result](2026-08-01_rase_ui_phase0g_independent48_results.md):
  独立 48-task screen 得到 immediate/deferred **37/48 vs 35/48**，双向 timing-only
  task 为 **4 vs 2**，oracle **39/48**；但相对最佳固定时机仅 **+2/48=4.17pp**，
  bootstrap CI `[0,8.33pp]`，低于冻结的 5pp gate，故 **NOT READY**。不降阈值、
  不开 96-state confirmation、不训练模型；下一步仅允许六个分歧态的机制审计。
- [RASE UI Phase 0H suffix-prefix6 kickoff](2026-08-01_rase_ui_phase0h_suffix_prefix6_kickoff.md):
  冻结 Phase 0G 的 4 个 immediate-only 与 2 个 deferred-only 状态，执行真实 active
  suffix 的 `k=0..5` 全前缀机制扫描；`k=0/5` 必须与 Phase 0G 成功和 SHA 双端点一致。
  本阶段仅为 outcome-selected 探索，不作总体或 selector claim，仍不训练模型。
- [RASE UI Phase 0H suffix-prefix6 result](2026-08-01_rase_ui_phase0h_suffix_prefix6_results.md):
  36/36 rollout 完成且 Phase 0G 双端点成功/SHA **6/6 PASS**；曲线中 4/6 单次翻转，
  2/6 三次非单调翻转，单次边界位于 `k={1,1,4,5}`，六种 pattern 全不同。共享标量
  timing mechanism **FAIL**，依冻结 stop rule 关闭 timing selector，固定 immediate OFT，
  不做 post-hoc targeted screen、不训练 selector/world model。
- [RASE Phase 1A replacement48 kickoff](2026-08-02_rase_phase1a_replacement48_kickoff.md):
  依旗舰计划首先执行“为什么不一直使用 OFT”审计；在 Phase 0G 的 48 个开发任务上重跑
  full-horizon SOURCE-ONLY，冻结 reset 后、首个 source action 前的 snapshot 运行
  OFT-ONLY，并与既有 env-step-25 source→OFT 严格 join。LIBERO reset 内部初始化使
  simulator timestep=10，但 policy step=0、source action=0；Phase 0 任务明确排除于
  flagship hidden test，不训练模型。
- [RASE Phase 1A replacement48 result](2026-08-02_rase_phase1a_replacement48_results.md):
  48-task exact join 完成：SOURCE-ONLY / OFT-ONLY / source→OFT 为 **10/42/37**；
  OFT-only 相对 source **+66.67pp**（95% CI `[+50.00,+81.25]pp`，McNemar
  `p=1.94e-8`），clean 也为 **15/16 vs 9/16**。仅 2 个 source-unique 例外跨两个
  suite，预注册 pilot gate 给出 `recovery_framing_signal`，但整体是**高 replacement
  risk + 稀疏待复现例外**；下一步只做互补性/真实成本 confirmation，不训练 selector/world model。
- [RASE-PRE PRE-A0 candidate opportunity](2026-08-03_rase_pre_a0_candidate_opportunity.md):
  12-state、K=4 严格同 profile 重采样得到 first / oracle@4 均为 **3/12**，
  **0 strict rescue、0 mixed-outcome state**；OFT 为 **11/12**，portfolio
  strict-only/fallback-only/both/neither = **0/8/3/1**。异构增益完全来自 fallback，
  所以状态为 **NOT READY**；下一步先做真实异构、同预算 PRE-A1 generator assay，
  不扩展当前重采样、不训练 critic/world model。
- [RASE-PRE PRE-A1 replan mechanism](2026-08-03_rase_pre_a1_replan_mechanism.md):
  在同一 12-state cohort 上用 OFT expert prefix 做 `h={0,1,4,8}` 配对 handback；
  四个长度均为 **3/12**，short-prefix rescue **0/9**，而 persistent OFT **11/12**、
  direct-only rescue **8**。结论是系统性的 Smol closed-loop 能力缺口，而非首 chunk
  采样问题；方法转为 temporally extended recovery option / persistent fallback，
  不训练短 replan head、critic 或 world model。
- [RASE-PRE PRE-A2 recovery duration](2026-08-03_rase_pre_a2_recovery_duration.md):
  12-state 闭环 OFT 轨迹的 `h={0,8,16,32,64}` handback 扫描得到
  **3/12, 3/12, 3/12, 2/12, 5/12**；两个跨 suite 的 camera failure 仅在
  64 步被救回，但 32 步误伤一个 base success，persistent OFT 仍为 **11/12**。
  结论升级为 recovery duration / safe handback 的机制信号；先做 task-disjoint
  replication，再决定是否训练 conservative termination model，world model 继续关闭。
- [PRE-A3 protocol freeze](2026-08-04_rase_pre_a3_protocol_freeze.md):
  冻结 120-state task-disjoint live closed-loop duration 协议、72/24/24 split、
  预注册 gate 与 runner/analysis；confirmatory 结果待采集执行。
- [PRE-A3 smoke4 kickoff](2026-08-04_rase_pre_a3_smoke4_kickoff.md) /
  [results](2026-08-04_rase_pre_a3_smoke4_results.md):
  4-state live closed-loop plumbing **PASS**；finite rescue 0、persistent 4/4，
  h=32 误伤 1 个 base success；method gate 保持关闭。
- [Method gate and paper track](2026-08-04_rase_method_gate_and_paper_track.md):
  在 PRE-A3 hidden pass 前冻结为 **benchmark/diagnosis**；termination/world-model
  gate 关闭。
- [Paper claim freeze update](2026-08-04_rase_paper_claim_freeze_update.md):
  更新可/不可主张边界与正文结构；条件性 safe-handback method 仍 gated。
- [PRE-B safe-handback scaffold](2026-08-04_rase_pre_b_safe_handback_scaffold.md):
  PRE-B 数据与 calibrated baselines 已实现但 gate 关闭；仅允许 scaffold，禁止 claim。
- [Benchmark release strengthening](2026-08-04_rase_benchmark_release_strengthening.md):
  release manifest、seed1/2 配置与第二 policy-pair 协议已就绪；多 seed/第二 backbone
  执行待 GPU。
- [RASE full project narrative](2026-07-31_rase_full_project_narrative.md):
  从立项到 W10 的完整流程叙事（与上方 Start here 同一文件）。

## Record requirements

Name new records `YYYY-MM-DD_<short-experiment-name>.md` and include:

1. status and research question;
2. checkpoint identity and frozen/trainable state;
3. resolved config path or embedded parameter summary;
4. seeds, task count, and episode count;
5. Python and dependency pins, upstream commits, Git SHA, and
   `env.lock.md` SHA-256;
6. aggregate and per-suite/task results;
7. artifact locations under `runs/`, `pool/`, or `results/`;
8. known deviations, failures, and follow-up work.

Never overwrite a historical result to reflect a new configuration. Add a new
record and link the superseded record instead.
