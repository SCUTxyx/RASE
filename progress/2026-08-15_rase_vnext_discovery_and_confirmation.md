# RASE vNext discovery 结果与 confirmation 启动记录

**日期：** 2026-08-15  
**结论：** discovery 工程 feasibility PASS；非 abort 科学诊断支持进入独立 confirmation，但尚未通过正式 opportunity gate，learned model、RL 与 OPD 全部继续锁定。

## 1. Discovery 完整性

- manifest：16 roots × 2 policies × 2 decision points × 5 operators × K3；
- 计划 jobs：960；完成：960；
- available：960；masked：0；invalid：0；
- success trials：709/960；
- completion fraction：1.0；
- operator coverage：1.0；
- branches SHA256：`aee69532817fd7c744038c4e43f54cf8de54c03679d25f9b88257dbee6af58c0`；
- feasibility audit SHA256：`10b3076aa142f2fa6e9de6b2fa40069b3334f7e9eccbcffd20c95386e77b4a8c`；
- 正式 frozen feasibility：PASS。

正式 feasibility 的 64/64 nondegenerate cells 包含 `abort.safe`，因此不能单独作为科学机会证据。

## 2. 排除 abort 的 shadow diagnostic

排除 `abort.safe` 后：

- nondegenerate cells：13/64 = 20.31%；
- Pi0.5：3/32 = 9.38%；
- Pi0Fast：10/32 = 31.25%；
- Goal：5/16 = 31.25%；
- Long：6/16 = 37.50%；
- Object：0/16 = 0%；
- Spatial：2/16 = 12.50%。

因此差异不是完全由 abort 机械制造，但强烈不均匀：主要来自 Pi0Fast、Goal/Long，Object 完全没有 outcome diversity。

## 3. Operator 总体结果

| Operator | Success | Mean utility | Paired harm | Mean query cost | Mean fallback cost | Mean latency cost |
|---|---:|---:|---:|---:|---:|---:|
| continue.source | 86.46% | 0.8646 | 0% | 0 | 0 | 0 |
| requery.source | 90.63% | 0.8743 | 3.13% | 0.004 | 0 | 0.0572 |
| resample.source | 92.19% | 0.9048 | 1.56% | 0.012 | 0 | 0.1234 |
| fallback.persistent | 100% | 0.9427 | 0% | 0.0715 | 0.5585 | 0 |
| abort.safe | 0% | -0.8646 | 86.46% | 0 | 0 | 0 |

Persistent fallback 是总体 best-fixed：成功率最高且 mean utility 最高。其代价是平均 fallback cost 0.5585。状态依赖 selector 的唯一潜在价值，是在 source/requery 已能成功的 roots 上避免 fallback 成本，同时在 source 会失败的 roots 上切入 fallback。

## 4. 探索性 opportunity（不是正式 gate）

- oracle minus best-fixed utility：0.05096；
- task/root bootstrap 95% CI：`[0.0000, 0.06518]`；
- G-O1 mean effect 通过，但 lower bound 不通过；
- Pi0.5 gap：0.0000；
- Pi0Fast gap：0.04550；
- root-level winners：continue 11/16、requery 3/16、persistent fallback 2/16；
- 三个 winner operators 均达到 10% root threshold；
- 探索性 G-O3 双 policy opportunity 不通过。

这说明 discovery 的最佳解释是“Pi0Fast 可能有 cost-aware requery/fallback selection 机会，Pi0.5 在这批 roots 上处于 source ceiling”。不能据此训练 pooled multi-VLA selector；需要 48-task 独立 confirmation 判断该结构能否泛化。

## 5. 重复稳定性

- 320 个 operator branches 中 316 个 K3 outcome 完全稳定：98.75%；
- continue：62/64 稳定；
- resample：62/64 稳定；
- requery、persistent fallback、abort：64/64 稳定；
- 无缺失 replica 或 K contract violation。

稳定性足以进入 K5 confirmation，但仍需保留 trial-level 概率结果，不得硬化为确定标签。

## 6. Resample capability 结论

- Pi0.5：96/96 trials 的两个 candidate 首动作不同；45/96 选择 candidate 1；
- Pi0Fast：96/96 trials 的两个 candidate 首动作完全相同；全部选择 candidate 0；selected action 与 requery 也完全相同。

Pi0Fast 当前 API 下没有真实 native resample diversity。confirmation 中将 `pi0fast.libero × resample.source` 设为显式 contract mask，而不是继续消耗三次 query 并把闭环噪声误认为独立 operator。

## 7. Confirmation 冻结设计

- 48 tasks，四 suites 各 12；
- 每 task 1 个 outcome-independent physical root；
- 与 discovery roots 重叠：0；
- 两个 source policies；
- source steps 8、16；
- 固定 K5；
- schedule jobs：4800；
- expected available：4320；
- expected masked：480，全部是 Pi0Fast resample；
- manifest SHA256：`ad3d55f28eb9da3aea7936e3b240aa8e1173dc372aaf4175d5c3a16570816739`。

选择一个 root/task 是有限算力下的预冻结折中：最大化 task breadth，保持 physical-root 独立性，并避免把同 task 多 roots 当成额外 task 支持。roots 使用 metadata-only SHA256 排序，未读取 confirmation outcome。

## 8. 正式 confirmation gates

### G-O1

- oracle minus best-fixed utility ≥0.03；
- task/root bootstrap 95% lower bound >0。

### G-O2

- 至少两个语义 operator 各自在 ≥10% physical roots 上最优；
- 至少 8 tasks、3 suites；
- decision points、policies、replicas 不增加 root 分母。

### G-O3

- Pi0.5 与 Pi0Fast 分别 gap ≥0.03。

同时运行含 abort 和排除 abort 的两份 audit。只有两份均 PASS 才解锁 information gate。

## 9. OPD 决策

当前不运行 OPD，原因不是保守偏好，而是父 gate 尚未满足：

1. discovery 的 bootstrap lower bound 为 0；
2. Pi0.5 opportunity 为 0；
3. persistent fallback 已是 100% best-fixed，必须先证明 selector 能可靠节省成本；
4. Pi/OpenVLA-OFT 的 token/log-prob 兼容性仍未建立；
5. 完整 VLA OPD 在当前单卡显存预算下风险高。

若 confirmation、low-capacity information 和独立闭环均 PASS，优先做 controller-OPD 小试验；若 confirmation FAIL，OPD/RL/world model 均不解锁。

## 10. 后续决策

- confirmation 全 PASS：优先做 T3 low-capacity operator-value information gate；
- 只有 Pi0Fast PASS：严格 multi-VLA 主线 FAIL；可另行冻结 Pi0Fast-only scope，但不能事后改写当前 gate；
- G-O1 或非 abort G-O2 FAIL：停止 learned selector，转为 stochastic operator opportunity atlas；
- opportunity PASS、information FAIL：停止模型升级，不上 world model/OPD；
- information PASS：实现轻量 MVP、冻结 OOF prediction，再做独立 closed-loop；
- closed-loop PASS 后：第三 VLA、第二 benchmark、IQL 或 controller-OPD 一次只扩展一个轴。

## 11. 运行状态

- tmux：`vnext_confirmation`；
- output：`runs/rase_vnext/confirmation_v1`；
- runner：`scripts/run_rase_vnext_confirmation_resume.sh`；
- 两个正式 smoke groups 已通过并被完整 runner 复用；
- runner 自动 summarize、运行含/不含 abort 两份 opportunity audit，并写入 `COMPLETE.json`。

