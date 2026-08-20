# RASE Semantic Selector 下一阶段规划(修订版 v2,2026-08-18)

> 基于 v1 规划评审结论修订:方向批准执行;两处硬伤修正(source-source 硬 Gate
> 统计功效不足;E2+E3 融合方式);四项补充(部署推理时序成本、continue 候选口径、
> calibration 样本量、P0 bug 修复)。

## 0. 修订说明(v1 → v2)

| 项 | v1 原文 | v2 修订 |
|---|---|---|
| Signal Gate | all-pairs **与 source-source 均**高于随机基线 | source-source **不作硬 Gate**(K5:0.548,p=0.139 不显著),降为分层报告 + shuffle 对照;Gate 主指标 = all-pairs + task bootstrap CI + informative tasks |
| E2+E3 | 独立消融,默认 E2 主模型 | 保留;消融增加**三种融合变体**(concat / 标准化 concat / PCA 降维 concat),防止把"简单 concat 维度膨胀"(0.703)误判为"融合路线失败" |
| 新增 P0 | 无 | 先修 analyze_rase_vnext_k3.py 特征错位 bug(records.append 在 try 外)+ 移除 offline selector 的 oracle 逻辑 |
| 部署时序 | 未覆盖 | 显式建模"候选预查询成本"(requery/fallback 的特征需预先推理才有,有真实延迟) |
| continue 口径 | 未覆盖 | 训练与部署统一用**完整 10 步 chunk**(生成时已知),禁止部署时误用剩余 1 步 |
| calibration split | 折内 task-level split | 48 tasks、4 folds × 12,折内 9 train + 3 calib;提示小样本(108 行/折)用 pooling 校准器或重复校准 |

## 1. 证据基线(已冻结,不再重跑)

- K3(8 tasks):same-root raw-action acc **0.750**(state-only 0.000),8/8 tasks 正向;
- K5(48 tasks):raw-action acc **0.787**,4/4 folds 同向,task bootstrap 95% CI
  **[0.359, 0.625]**,**28/48 tasks 正向**,536 informative pairs;
- source-source(同策略 continue vs requery):
  - K3:0.625(72 pairs,p=0.022,显著);
  - K5:0.548(144 pairs,p=0.139,**不显著**;informative pairs 仅 34/144=23.6%);
  - 解读:同策略动作语义信号存在且方向一致,但效应中等、功效不足,只作方向证据;
- fallback 分布先验贡献最大(req-vs-fb 0.787 / cont-vs-fb ~0.6),selector 必须
  显式处理 operator 先验(报告 operator-stripped 指标,防止分数被先验掩盖)。

## 2. 目标与边界

- **主目标**:冻结可部署的 same-root 候选选择器(低容量:ridge/logistic),
  在**成本敏感区间**内优于 continue / requery / always-fallback 基线;
- **非目标**(保持锁定):RL / OPD / world-model / multi-VLA pooled 声明 /
  实时闭环部署声明 / semantic pretraining gain 声明。

## 3. P0:代码修复与审计(先于一切,~0.5 天)

1. **修复特征错位 bug**:`scripts/analyze_rase_vnext_k3.py::load_features_and_targets`
   `records.append` 移至 `try` 块内;失败样本同步过滤;新增审计断言
   `len(records) == len(state) == len(raw) == len(trace) == len(labels) == len(tasks)`;
2. **移除 oracle 逻辑**:`offline_utility` 的 selector 改为纯预测驱动
   (margin 选择 + 冻结阈值),禁止回退到 realized-utility max 选择;
3. 全量测试 + 重算 K5 基线,确认修复后指标无回归(结果一致性记录)。

## 4. 数据冻结与无泄漏 split(~0.5 天)

- 从 `runs/rase_vnext/k5_collect_v1/` 生成**不可变 selector dataset manifest**:
  capture/feature schema、全体文件 hash、task/root/operator/replica 索引、
  success 标签、fold 分配;manifest sha256 冻结;
- **K5 = 开发集**:4 folds × 12 tasks(task-held-out);折内 calibration split
  (task-level:9 train + 3 calib,禁止 row/root 随机切分);
- **K3 = 一次性外部确认集**:在模型、α、特征、abstain 阈值、λ_cost 全部冻结前
  **不读取**其 selector 指标;
- **标签**:二元 success;同 root 内 success 相同的 pair 不参与 ranking
  loss/accuracy(保留用于概率校准);**成本不进标签**,U_λ = P(success) − λ_cost·cost
  仅在部署决策层构造。

## 5. 特征预注册(冻结,不因单次 fold 临时增删)

| 特征组 | 内容 | 角色 |
|---|---|---|
| E2 | raw-action stats(mean/std/min/max/last/len × 7 dim) | **主模型** |
| E3 | CanonicalMotionTrace(速度/加速度/jerk/路径/反转/gripper) | 独立消融 |
| E2+E3-a | 直接 concat | 消融 |
| E2+E3-b | 标准化后 concat | 消融 |
| E2+E3-c | PCA 降维后 concat | 消融 |

- 禁止特征:operator ID、realized outcome/utility、task/suite 身份、
  未来状态;动作可计算的运动学/时序特征允许(trace+semantic 定义);
- **continue 候选统一用完整 10 步 chunk**(与部署一致,不用剩余 1 步)。

## 6. 模型(低容量双路线,超参仅 K5 nested OOF 选择)

- **A:候选级** ridge/logistic(E2 特征,absolute success 回归/分类);
- **B:显式 pairwise** ridge/logistic(同 root 候选对输入,直接建模 Δsuccess);
- 比较并冻结其一;报告 all-pairs / source-source / cont-fb / req-fb 分层指标。

## 7. 可部署 abstain 与成本决策

- 预测时以 **top-1/top-2 分数差(margin)** 作 confidence;margin < 冻结阈值 →
  abstain,默认执行 continue(同时报告 abstain→fallback 敏感性);
- 0.01 仅作真实 utility tie 的评估容差,**不充当预测 margin**;预测阈值在
  K5 calibration folds 上按 risk-coverage 冻结(不允许事后调);
- 输出覆盖率 25/50/75/100% 的 success、regret、错误率、operator 分布,
  画完整 risk-coverage 曲线;
- **成本曲线**:U_λ 扫描(λ_cost ∈ {0.05, 0.1, 0.2, 0.5}),比较 selector /
  continue / requery / always-fallback / oracle upper bound,明确 selector
  在什么 λ_cost 区间优于 always-fallback(break-even 报告);
- **部署时序成本**:requery/fallback 的特征需预先查询(现推理/OFT 调用),
  其延迟计入成本层;报告"预查询总延迟"与 coverage 的权衡。

## 8. Gates(冻结后按序执行)

### Protocol Gate
- P0 修复完成、dataset manifest/split/schema/label/threshold 全部冻结;
- 全测试 + provenance 审计(capture v2、无 queue 反推、无 outcome 依赖)通过。

### Signal Gate(K5 nested OOF)
- all-pairs pairwise acc 显著高于随机(0.5),task bootstrap 95% CI 下界 > 0;
- informative tasks ≥ 24/48(当前 28,留裕度);
- source-source 分层报告 + shuffle 对照(**不作 PASS/FAIL**);
- 报告完整(不只看 pooled pair 数)。

### Deployment Gate
- 在**预声明 λ_cost 区间**内优于 continue;
- 给出相对 always-fallback 的 break-even 区间与覆盖率;
- **若仅靠 operator prior 获益,不宣称 semantic selector 成功**(报告
  operator-stripped 后的真实增益)。

### K3 一次性外部确认
- 冻结后仅运行一次;PASS → 并行推进 π0.5 challenge 与 closed-loop;
- FAIL → 诊断(calibration / distribution shift / cost policy),不直接停止主线。

## 9. 工程产物

- `rase/vnext/selector/` 模块(从 phase_c_pilot 抽取):train / save / load /
  predict / select 接口,含 feature schema、task folds、ridge、grouped metrics;
- 模型 artifact 元数据:feature schema/version、normalizer、weights/calibrator、
  abstain threshold、cost policy、训练 manifest hash、代码版本;
- 端到端测试:特征与 metadata 对齐、same-root pairing、task 泄漏防护、
  序列化复现、abstain 行为、无 realized outcome/utility 推理依赖、
  operator-stripped 评估。

## 10. 执行顺序与预估

| # | 工作 | 预估 |
|---|---|---|
| 1 | P0 修复 + 审计 + 回归确认 | 0.5 天 |
| 2 | dataset manifest + split 冻结 | 0.5 天 |
| 3 | selector 模块 + 双模型 + 端到端测试 | 1-2 天 |
| 4 | K5 nested OOF + abstain/risk-coverage/成本分析 | 1 天 |
| 5 | Protocol/Signal/Deployment Gate + K3 一次性确认 | 0.5-1 天 |
| 6 | π0.5 8-task × K3 challenge + closed-loop 验证(并行) | 1-2 天 GPU |

## 11. 预期产物

- 冻结协议文档 + selector dataset manifest(sha256);
- 可复现训练/评估/推理入口 + 版本化模型 artifact;
- K5 nested-OOF 报告、K3 confirmation 报告、risk-coverage 与成本 break-even;
- 一页 PASS/FAIL 决策记录(是否进入 π0.5 与 closed-loop)。

## 12. 不启动(直到更高 gate)

RL / OPD / world-model / multi-VLA pooled selector / 大规模公开数据下载 /
实时 closed-loop 部署。
