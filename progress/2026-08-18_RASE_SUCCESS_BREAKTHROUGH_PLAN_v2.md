# RASE 原路线保留与成功率突破计划(修订版 v2,2026-08-18)

## 〇、可行性总评

**可行,方向正确,批准执行;但基于现有数据的预注册预期是:Opportunity Gate 大概率 FAIL,
计划必须同时准备好"诚实收尾"路径(已包含)。** 关键事实核查(2026-08-18):

- `phase_a_audit.json` 存在(21KB,含 full/non-abort opportunity + integrity);
- protocol 已定义 **step 8 / step 16 两个 decision points**;
- **confirmation_v1(4800 rows,48 tasks × 2 points × 5 ops × 5 reps)已有两个边界的
  完整分支数据**,可零成本构建 domain atlas;
- **heterogeneous-winner 区域在现有数据中极稀疏**:
  - step 8:hetero 3 units(全部来自 1 个 root `sp1_d988a6d360f`);
  - step 16:hetero 5 units(2 个 roots:`sp1_2604b0af7f9`×3、`sp1_d988a6d360f`×2);
  - 其余为 fallback-dominates(222-231)或 all-fail(6-13);
- 独立 roots(有完整 outcome):confirmation 48 + K5 48 + K3 24 = **120 ≥ 100 门槛** ✓;
- fallback-not-optimal CI 下界 >5% 门槛:**现有数据不可能满足**(0.7%-2%)。

**推论**:LIBERO 域(48 tasks、π0-fast/π0.5、step 8/16)的 fallback 统治是结构性的;
本计划的价值 = ①用严格预注册把"无空间"变成最终结论;②若域 miner 意外发现
heterogeneous 子域(如动态临界边界),按 Gate 有条件重开成功率主线;③为更难
benchmark 预留完整方法管线。

---

## 一、冻结不变项与可调整项(保留原计划 §1)

- **冻结**:source VLA、轻量 model-agnostic runtime controller、same-root 多候选
  反事实价值比较、stochastic recoverability、intervention urgency、在线选择
  (continue/requery/resample/corrective takeover);方法定义
  `outputs/RASE_CANONICAL_IDEA_stochastic_multi_vla_risk_control_2026-08-13.md`;
- **主 endpoint**:closed-loop success improvement;成本仅约束与次指标,不改成
  "省 fallback 调用"主论文;
- **可调整**:evaluation domain、causal decision time、候选组成、采样密度;
- **禁止**:事后按 outcome 选任务、移除强 baseline、调高人为成本权重、只与裸
  continue 比较、把审计工具包装成 selector 成功;
- **结论限定**:当前 K5 边界的"免费 OFT fallback 统治"不外推为 RASE idea 无效;
  正证据保留(pairwise K5 0.871 / K3 0.778),可学习性 ≠ 部署收益。

## 二、Domain Atlas 第 0 步(零新收集,立即执行,~1 天)

**修正**:原计划 §2 的"筛选特征只能使用 branch rollout 前信息"与"fallback-
not-optimal rate"存在表述矛盾(domain 分类本身需要 outcome)。修订为**双层
outcome 政策**:

```text
候选 domain 特征(预注册,pre-rollout):source 历史失败率、progress/time、
状态/动作表征、policy/suite/task 元数据、预定义 decision boundary
domain 分类标签(允许使用):仅限内层开发数据(confirmation + K5 + K3 的
outcome),外层验证集 outcome 完全冻结 —— 这正是 nested split 的意义
```

- 输入:confirmation_v1(step 8/16,注意**分 policy 报告**——确认是否含 pi05 +
  pi0fast 两 policy)+ K5 + K3 + phase_a_audit + frozen manifests;
- 重建 root × task × decision-time atlas,每格输出:支持量、source success、
  fallback success、fallback-not-optimal rate、winner entropy、
  oracle-minus-best-fixed、all-fail rate、task-bootstrap CI;
- 三分类:**all-candidates-fail / fallback-dominates / heterogeneous-winner**;
  只有第三类支持 selector;
- 不做"Object/Long = 困难域"的事前假设(数据:Object 有 2 个 all-fail task,
  Long informative support 薄——用数据分类,不贴标签);
- **新增:hetero roots 个案诊断**——`sp1_d988a6d360f`、`sp1_2604b0af7f9`
  (fallback 在这些状态失败)做定性检查(observation/state 特征、OFT 失败模式),
  先于大规模收集,成本 ≈ 0;
- 输出:机器可读 `domain_atlas_v1.json` + 一页分类汇总。

## 三、训练前 Opportunity Gate(保留原计划 §3,附数据预期)

预注册门槛(同时满足):

| 门槛 | 阈值 | 现有数据预期 |
|---|---|---|
| 独立 tasks | ≥24(每 suite ≥6) | ✅ 48 |
| 独立 roots | ≥100 | ✅ 120(confirmation+K5+K3) |
| ≥2 operators 各自在 ≥10% roots unique/co-best | — | ❌ 大概率不满足(fallback 20/24) |
| fallback-not-optimal CI 下界 | >5% | ❌ 0.7%-2% |
| oracle-minus-best-fixed | ≥5pp 且 CI 下界 >0 | ❌ ~0.7pp |
| all-fail rate 不主导 headroom | — | ⚠️ 部分 task 全败 |

- multiplicity-aware 报告,预冻结唯一主 domain;
- **Gate FAIL(预期)**:不训练新 selector,形成"现有 benchmark 无可识别成功率
  空间"最终结果(这是预注册的科学结论,不是失败);
- **Gate PASS(意外)**:仅对通过的唯一主 domain 继续 §4-§6;
- baseline 必须为该 domain 的 best-fixed operator,always-fallback 单独报告;
  禁止用 continue 弱表现替代强对照。

## 四、小规模预注册补采:时机而非 idea 作为变量(保留原计划 §4,修订 decision time)

- **decision time 修订**:现有 protocol 只有静态 step 8/16。建议:
  - step 8/16 直接用**现有数据**做 miner(零收集);
  - 新收集只做**一个"临界前"动态点**:定义 = source 首次 progress 停滞/低
    progress(如 post_decision progress < 20%)或首次碰撞时触发;这是 canonical
    idea 的 urgency 检验,而非继续沿用一个 fallback 仍近完美的静态边界;
- 同一 root 预注册 3 个 decision times(step 8、step 16、临界前动态点)用于
  urgency 曲线;若动态点实现复杂,退化为 step 8/16/32 三静态点(预注册其一);
- 每个 state 同步采集完整 continue/requery/native resample/persistent fallback;
  候选部署时真实可执行、共享前缀、固定生成预算、禁止事后重推或按 outcome 选;
- **K≥5** 估计 p_success + 区间;保存完整 candidate action、seed、真实增量
  wall-time/query/fallback 成本与 provenance;
- **修复 `source_prefix_wall_s` 缺失**(小代码修改:prefix_to_decision 落盘
  wall),使 S1 延迟场景可解释;
- Pilot 规模:24 tasks × 3 times × K5(仅当 miner 显示该域最接近 Gate 时);
  只有 pilot 仍满足 Opportunity Gate 才扩 48 tasks,否则停止。

## 五、非退化域训练现有 RASE selector(保留原计划 §5)

- 复用 `rase/vnext/selector.py` explicit pairwise `x_a - x_b → Δp_success`;
  不启动大模型/RL/world-model;
- **target 修订**:改为 K-count 软成功概率差 + 置信权重(避免小 K 硬标签)——需
  实现 `soft_delta` target(小改动);
- 输入显式加入 decision-time/urgency、policy descriptor、候选 action 差异;
  low-capacity shared core;policy adapter 仅作预注册消融;
- 输出候选排序 + calibrated margin;margin 不足回退训练集内冻结的 best-fixed;
  所有阈值只在内层 train/validation 确定;
- 对照:continue、requery、resample、always-fallback、best-fixed、
  risk-threshold→fallback、operator-prior、RASE pairwise selector、oracle;
  always-fallback 与 best-fixed 是必须击败的主对照。

## 六、独立 closed-loop 验证与论文判定(保留原计划 §6)

- 完全 held-out tasks/roots 上执行;主统计单位 task/root(K rollouts 不作独立样本);
- **Primary**:RASE vs best-fixed/always-fallback 的 success gain,paired
  task-bootstrap 95% CI;**CI 下界 >0 且绝对提升 ≥3pp** 才 PASS;
- Secondary:harm UCB 不恶化、延迟与 fallback 调用完整披露;
- 三个必要分解:oracle headroom→selector captured fraction、各 operator
  winner coverage、按 decision time 的 urgency 曲线(区分失败来源:无空间/
  信号不可观测/selector 没学好);
- 论文主表并列 K5 ceiling 域与新 high-risk 域;不隐藏 always-fallback 负结果。

## 七、停机与分支决策(保留原计划 §7)

- Gate FAIL:不补数据、不调大模型;结论 = 所检查 domain 均缺 success headroom;
- Gate PASS + selector FAIL:opportunity 成立但 observability/learning 失败;
  只允许一次预注册 feature/urgency 消融;
- Gate PASS + offline PASS + closed-loop FAIL:定位 distribution shift 或候选
  执行偏差;禁止仅以 offline 结果宣称方法有效;
- closed-loop PASS:扩展第二 policy/benchmark;每个新域独立过 Opportunity Gate。

## 八、执行顺序与预估

| # | 工作 | 输入 | 预估 |
|---|---|---|---|
| 0 | domain atlas(双层 outcome 政策)+ hetero roots 个案 | 现有 confirmation/K5/K3 数据 | 1 天(CPU) |
| 1 | Opportunity Gate 判定(multiplicity-aware) | atlas | 0.5 天 |
| 2 | 【若 PASS】pilot 冻结 + 补采(24×3×K5) | 新收集 | 2-3 天 GPU |
| 3 | 【若 PASS】selector 软 target 训练 + offline OOF | pilot 数据 | 1-2 天 |
| 4 | 【若 PASS】closed-loop + 论文判定 | held-out | 1-2 天 |
| 5 | 【若 FAIL】最终收尾:现有 benchmark 无空间结论 + 一致性审计 | — | 0.5 天 |

## 九、产物

- `domain_atlas_v1.json`(机器可读)+ 分类汇总;
- 预注册文档(`protocols/domain_mining_preregistration_v1.json`):mining 规则、
  split、Gate、pilot 预算、主指标、停止规则;
- hetero roots 个案诊断报告;
- (条件性)pilot manifest、软 target selector、closed-loop 报告;
- 更新 `progress/2026-08-18_RASE_FINAL_VERDICT_PI05_AND_DEPLOYMENT.md`:
  "停止整个 selector 主线"修正为"停止当前退化 domain,按预注册 Gate 有条件
  重开成功率主线";
- 旧收尾计划 `/root/.cursor/plans/rase-paper-closeout_92bf077a.plan.md` 标记
  为被本计划取代,保留一致性审计与复现冻结任务。

## 十、对原计划的修改清单

| # | 原计划 | 修订 |
|---|---|---|
| 1 | "outcome-blind domain miner" | 双层 outcome 政策:pre-rollout 特征 + 内层 outcome 分类,外层冻结 |
| 2 | ≥100 roots 需新收集 | 现有 confirmation+K5+K3 = 120 独立 roots,门槛已可满足 |
| 3 | 3 个静态 decision times | step 8/16 用现有数据;新收集只做"临界前"动态点(或 step 32 备选) |
| 4 | 未提异构 roots | 新增 hetero roots 个案诊断(零成本前置) |
| 5 | 未给 Gate 预期 | 基于数据给出"大概率 FAIL"预期,FAIL 即最终科学结论(预注册) |
| 6 | K≥5 硬标签 | 软 Δp + 置信权重(§5 target 修订) |
| 7 | — | 新增 §0 可行性核查与 §八 执行顺序/预估 |
