# RASE E4-0 执行报告：π0-fast best-of-K 候选池信息量审计（2026-08-21）

> 目的：验证文献主流形态（RoboMonkey 式 "同策略多采样 + verifier 选"）在
> LIBERO-Long + π0-fast 上是否有候选池级机会。E4-1（SmolVLM2 verifier）按
> 纪律未执行——oracle gain=0 时 verifier 无可用信息。

## 结论（预注册 Gate：FAIL）

**π0-fast temperature=0.7 采样的 K=8 候选池存在真实 chunk 多样性（L2 0.26–1.13），
存在 outcome 分叉状态（12/24），但 oracle@8 = best-of-1 = 62.5%，oracle gain = 0pp——
没有任何 "candidate 0 失败 → 后续候选成功" 的 rescue 状态。**

即使拥有完美 verifier（oracle 选择），best-of-K 相对单次采样**零增益**。
RoboMonkey 式路线在 LIBERO-Long 的候选池前提不成立。

## 协议

- 策略：`ckpts/pi0fast_libero`（自回归离散 token；`config.temperature=0.7` 启用
  token 级采样——注意：`predict_action_chunk` 读取 `config.temperature` 而非
  kwargs，`policy.temperature` 属性不生效，已修）；
- 状态：clean LIBERO-10，tasks 1/2/9（G2a 成功率 8/8、5/8、4/8）× 8 init states =
  24 个决策状态（t=10，greedy 前缀）；每个状态 restore 同一 snapshot 后执行 K=8
  个独立采样 chunk（每候选独立 seed），随后 greedy（T=0）续跑到真实终局；
- 成本：192 条真实终局 rollouts（~6h GPU）；
- Gate：≥2 个 mixed 状态 且 oracle@8 − best-of-1 ≥ 3pp。

## 结果

| 指标 | 值 |
|---|---|
| best-of-1（candidate 0） | 15/24 = 62.5% |
| oracle@8 | 15/24 = 62.5% |
| **oracle gain** | **0.0pp** |
| mixed 状态（K 内 outcome 分叉） | 12/24 |
| **rescue 状态（c0 败、k>0 成）** | **0/24** |
| chunk L2（mean，全状态） | 0.26 – 1.13 |
| Gate | FAIL |

### 分叉状态模式（12 个 mixed 全部为 "c0 成功 + 部分冗余成功"）

```text
SS......  SS......  SSSS....  SSSS....  SSSSS...  ...（candidate 0 恒为成功侧）
```

**模式：成功与否由状态难度主导**——可成功状态产生 2–8 个成功候选（冗余多样性），
不可成功状态产生 0 个。T=0.7 采样的 chunk 差异（真实、非零）不改变终局归属。

## 机制解读（与既有证据链的衔接）

1. **BOKBO 警告的终局版**：候选差异（chunk L2）真实存在但不携带"可仲裁的结局差异"；
   G1（SmolVLA 多尺度噪声）与此同构，本轮用 π0-fast + token 级采样 + 真实终局
   标签再次确认——**生成侧多样性与结局级互补之间没有桥梁**。
2. **不是 verifier 问题**：oracle gain=0 排除 verifier 判别力；是候选池结构问题
   （成功集合嵌套于状态难度）。
3. **不是温度问题**：0.3/0.5/0.7/0.9 多尺度已在 G1 覆盖；本轮 0.7 是 FAST 论文
   验证值。候选质量差异（chunk 层面）存在但结局不变。
4. **与 RoboMonkey 的差异**：其 ID +9pp/OOD +25pp 建立在 base 成功率 ~50–60%
   且候选质量差异可改变结局的域；π0-fast 在 Long 86%（且任务难度主导）时
   候选池无结局级互补空间。

## 对路线的裁决

- **同策略 best-of-K + verifier（文献主流形态）在 LIBERO-Long 结构性无机会**
  （oracle 上界 = 默认采样）；
- 不执行 E4-1（SmolVLM2 verifier）——oracle gain=0 时 verifier 训练/评估无意义
  （延续"不在 oracle gain=0 时训练 verifier"纪律）；
- 候选池信息量的可测量前提（rescue 状态 > 0）成为后续所有域筛选的**前置 gate**：
  先测 oracle@K > best-of-1，再谈 verifier。

## 产物

- `runs/e4_candidate_pool_audit_v1/summary.json` + 24 个 per-episode JSON
- 脚本：`scripts/e4_candidate_pool_audit.py`（temperature 采样修复记录在脚本 docstring）
- Git：待提交（见 commit 消息）

## 下一步

按规划 Track A/B：
- 论文轨：本结果 + G1 合并为"生成侧多样性不产生结局级互补"的完整证据（两个
  策略族 × 两种采样机制 × 真实终局标签）；
- 实验轨：候选池 oracle 前置 gate 并入 Eligibility Screen（E0'）；若后续域
  （新策略对/新任务集）oracle@K > best-of-1 且 rescue ≥2，再恢复 verifier 路线。
