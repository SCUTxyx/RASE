# RASE 实验进展与结果（截至 2026-08-21）

> 服务器：AutoDL bjb2:28921
> 本文档为唯一进展主表。idea 见 `RASE_IDEA.md`，规划见 `RASE_PLAN.md`。
> 所有实验均带 git 快照与 gate 纪律；legacy 数据已标记，不得用于真实 claim。

---

## 1. 阶段总览（十二环证据链）

| 阶段 | 时间 | 结果 | 一句话 |
|---|---|---|---|
| 历史 R 系列 / K3 / K5 | ~08-18 | 完成 | 动作信号可学习（0.787/0.778）；可学习≠可部署 |
| vFinal 冻结 | 08-19 | 完成 | 三文档 + Eligibility Screen + sha256 |
| OFT 机会窗口 | 08-19 | PASS | 异构模型对互补 100%/0%，headroom +50pp |
| OPD 蒸馏链 | 08-19 | 部分 PASS | 风险可预测（校准 97.6%）；查询表闭环 99.2%（=记忆） |
| v3 Gate A/B/C/D | 08-19 | 见 §2 | 零样本部分 PASS；反事实协议成立；WM 失败 |
| W1/W2（未来信息） | 08-20 | PASS/FAIL | 真实未来 +3.4~3.8pp；learned 未来失败 |
| 闭环 v2 | 08-20 | FAIL | 49.2% = best-fixed，0 切换全 abstain |
| **G1 多尺度重生成** | 08-20 | FAIL | 1/8 rescue，主结构 = task difficulty |
| **G2a/b 跨策略** | 08-20 | FAIL | π0-fast Long 86.25%（过强）；Spatial 弱支配 |
| **E3/E3-B 残差** | 08-20/21 | FAIL | 离线可学（MSE 改善 37%/28%）；闭环 0 candidate-only |
| **E4-0 候选池** | 08-21 | FAIL | oracle@8 = best-of-1（0/24 rescue） |
| **R0 执行期验证** | 08-21 | FAIL | proprio 演化 99.3% 可预测，偏差与结局无关 |
| **R0b 重规划频率** | 08-21 | FAIL | k4 50% vs k10 67%；固定频率无增益（= BCP 前提） |
| **BCP 对照** | 08-21 | 外部证据 | 学习自适应 horizon 在扰动域有增益（LIBERO-PRO +6.8%） |

---

## 2. 历史阶段（摘要）

- K3（8 tasks, 432 slots）：same-root pairwise 0.750；K5（48 tasks, 864 slots）：0.787；
- Semantic Selector 0.871/0.778；Deployment FAIL（fallback 非最优 0.7%）；
- π0.5 ceiling（97.2%）、Dynamic route（确定性无 diversity）、libero_90 all-fail；
- v3 Gate A：π0-fast 零样本 0.837（高分域）；VLA-ID probe 0.894（指纹泄漏）；
- v3 Gate B/C：same-root 反事实分叉成立、oracle 未来排序 0.9998（**注：Gate C 曾
  有循环性，已修正并标记 legacy**）；
- v3 Gate D / W1 / W2：WM 摘要无增益（B2≈B1）；真实未来 2/3 折 +3.4~3.8pp；
  learned 未来两次改进均失败 → **WM 路线搁置**；
- OFT 机会窗口：oft_spatial/oft_object 互补 100%/0%，headroom +50pp；
- OPD 蒸馏：校准 0.7→97.6%；查询表闭环 99.2%（明确为任务记忆，非 RASE）；
- P0 审计：offline/deploy 特征等价（max diff=0）、root 一致性、normalizer 一致 → PASS。

---

## 3. 九环排除链（2026-08-20/21，核心科学产出）

| # | 形态 | 实验 | 结果 | 失败机制 |
|---|---|---|---|---|
| 1 | 策略间切换 | 96-state Goal/Long（OFT vs SmolVLA）；G2a/b（π0-fast vs SmolVLA） | 0 continue-only；H_within=0；oracle gain 0pp | 嵌套失败模式（[Experts-in-an-MDP](https://proceedings.nips.cc/paper_files/paper/2004/hash/421b3ac5c24ee992edd6087611c60dbb-Abstract.html) 定理级） |
| 2 | 同策略噪声重采样 | G1：SmolVLA 多尺度温度（8 roots × K=8） | 1/8 rescue（<2 gate）；T=0.9 有孤立机制证据 | 噪声轴多样性不携带任务语义（BOKBO 实证版） |
| 3 | 跨架构候选 | G2a：π0-fast clean Long 80 eps | 69/80 = 86.25% > 70% 上界 | π0-fast 过强（ceiling 无 headroom） |
| 4 | 跨架构同状态 | G2b：Spatial16 | 0 continue-only；H_within=0%；oracle gain 0pp | 能力嵌套（fallback 弱支配） |
| 5 | BC 残差 | E3：OFT recovery 轨迹监督（560 参数） | 离线 MSE 改善 37.2%；在线 0 residual-only | teacher-forcing 分布偏移 |
| 6 | DAgger 残差 | E3-B：on-policy relabel 两轮 | 离线 28.1%；B2：0 candidate-only（16/24 neither） | 教师天花板 + 剂量/泛化 |
| 7 | 候选池 best-of-K | E4-0：π0-fast T=0.7（24 states × K=8，真实终局） | **oracle@8 = best-of-1 = 62.5%（0/24 rescue）** | 生成侧多样性不产生结局级互补 |
| 8 | 执行期偏差检测 | R0：predictor (s_t,chunk)→s_{t+8} vs 终局 | 可预测性 99.3%；偏差-结局 AUROC 0.51 | 确定性仿真偏差恒≈0（验证类方法无信号） |
| 9 | 重规划频率 | R0b：σ=0.05 噪声 × k∈{4,10} | k4 50% vs k10 67%（-16.7pp） | 固定频率重规划无增益（= BCP 前提验证） |

**九环结论**：clean 确定性 LIBERO 上，运行时纠正的全部形态（选择/验证/生成/
频率）均被机制级证伪。**扰动/不确定性是仲裁价值出现的必要条件（H2）**——
BCP 在扰动域（LIBERO-PRO/RoboTwin Randomized/真机）的自适应 horizon 增益
（+1.7~40pp）是外部正面证据。

**R0b 副产品**：动作噪声本身救回无噪必败状态（t01-i00 无噪 0/8 → 带噪成功；
t09-i00 1/8 → 成功）——执行侧随机性有探索价值（环境效应，不可学），与
"生成侧随机性无互补"（G1/E4-0）形成不对称观察。

---

## 4. BCP 对照（2026-08-21 关键外部锚点）

[BCP：Continue or Replan? Bernoulli-Continuation Policy Learning for
Adaptive Horizon Execution](https://arxiv.org/abs/2608.03483)（北大+微软亚研，
2026-08）：

| 域 | BCP 增益 | 说明 |
|---|---|---|
| LIBERO（π0.5） | +1.7% | 97.0→98.7% |
| **LIBERO-PRO（物体位置扰动）** | **+6.8%** | 30.9→37.7%；增益随扰动幅度增大 |
| RoboTwin 2.0（LingBot-VLA） | +4.06%（50 任务）；低成功任务 +11.08% | Clean 训练 → Randomized 泛化 +4.06% |
| 真机 AGIBOT G1 | 74→92%、44→84% | 两个操作任务 |

**对我们的意义**：
1. 我们的 R0b（固定频率 k4 vs k10 FAIL）恰好验证了 BCP 的核心前提——
   "fixed-horizon replanning is a task-agnostic periodic schedule"的局限；
2. 正确形态是**学习**自适应 continue-or-replan（BCP），而非固定频率；
3. **LIBERO-PRO（物体位置扰动）是我们之前没跑过的域，且是这类方法增益最大的域**；
4. BCP 未覆盖：跨策略 switch 臂、same-root 反事实校准、统计保证、zero-shot 跨 VLA
   ——即 RASE 的差异化层。

---

## 5. 数据与代码资产（服务器）

| 资产 | 位置 | 状态 |
|---|---|---|
| same-root 数据集（648 行 legacy） | `runs/oft_opportunity/same_root_w1.jsonl` | legacy（重复 chunk，仅诊断） |
| 修正 recovery pilot | `runs/oft_opportunity/crr_recovery_pilot64_v1.jsonl` | 可用（single native chunk） |
| E4-0 候选池审计（24×8 真实终局） | `runs/e4_candidate_pool_audit_v1/` | 可用（核心负面证据） |
| R0 执行期验证（192 行 branch-end） | `runs/e5_ev_probe_v1/` | 可用 |
| R0b 重规划频率 | `runs/e6_replan_freq_v2/` | 可用 |
| G0-G2 全部 artifact | `runs/` + git 历史 | 已提交（`a256e83`…`25930cd`） |
| 模型 ckpts | `ckpts/{smolvla_libero, pi0fast_libero, pi05_libero, oft_*, SmolVLM2-500M}` | 全部在位 |
| 环境 | conda `smolvla`（LeRobot 0.5.1）/ `oft` | 在位 |

Git 关键提交：`a256e83`（CRR 快照）→ `6ab1fc9`（修正协议）→ `2d73b38`（96-state）
→ `c2dc942`（G1）→ `f330510`（G2）→ `9d3559e`（E3）→ `df54eed`（E3-B）
→ `a76a663`（E4-0）→ `579af37`（R0）→ `25930cd`（R0b）。

---

## 6. 主张状态表（2026-08-21）

| 主张 | 状态 | 证据 |
|---|---|---|
| same-root 反事实协议 | ✅ PASS | P0 审计 + Gate B |
| 风险预测 OFT 架构内转移 | ✅ PASS | LOVO 0.94+ |
| 风险预测跨架构转移 | ⚠️ 未确证 | π0-fast 0.837（高分域） |
| clean 确定性域仲裁无空间 | ✅ PASS | 九环一致（闭环 v2/G1/E4-0/R0/R0b） |
| 扰动域自适应 horizon 有增益 | ✅ 外部证据 | BCP：LIBERO-PRO +6.8%（待复现） |
| **扰动域跨策略仲裁（RASE 主贡献）** | ⏳ 未验证 | R2 计划（见 RASE_PLAN.md） |
| 身份无关（无指纹） | ⚠️ 未达成 | VLA-ID probe 0.894 |
| 跨平台 | ⏳ 未验证 | 特征含 LIBERO 特定量 |
| WM 路线 | 搁置 | Kill 条件满足（除非 B1 低 + oracle headroom >10pp） |
