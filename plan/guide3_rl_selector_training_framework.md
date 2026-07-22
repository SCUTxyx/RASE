# 施工指南三：RL Selector 训练框架搭建（超详细版）

配套文档：RASE-Lite 设计报告 v3.1 §4、§6。覆盖：系统架构、特征管线、动作空间与 fallback 执行器实现、offline warm-start（BC+CQL）、online DQN、并行采样 infra、奖励与预算记账、稳定性预案、评测与日志。

> 实施事实（2026-07-17）：当前仓库根目录为 `RASE/`；SmolVLA 环境使用
> Python 3.12，OFT 环境使用 Python 3.10，LIBERO-Plus 固定 commit
> `4976dc3`。本指南属于 W10 之后的设计，任何示例版本或硬件参数均由
> 届时的 `env.lock.md` 与 `configs/` 覆盖。

---

## 0. 系统总览

```
                    ┌────────────────────────────────────────────┐
                    │              Selector (3.5M, DQN)           │
   特征向量 x_t ───▶│  Q(x, u)  →  argmax / ε-greedy              │──▶ u_t ∈ U(s)
                    └────────────────────────────────────────────┘
                          ▲                                │
        ┌─────────────────┴──────────┐        ┌────────────▼─────────────┐
        │  Feature Pipeline (frozen) │        │   Fallback Executor       │
        │  DINOv2-B / V-JEPA 2 ViT-L │        │  ROLLBACK/RESAMPLE/REPLAN │
        │  ACC / value-probe / SAFE  │        │  WAIT/ABSTAIN + 候选执行   │
        └─────────────▲──────────────┘        └────────────┬─────────────┘
                      │              LIBERO-Plus env        │
                      └────────────────◀────────────────────┘
```

决策粒度：每个 action chunk 边界做一次 selector 决策。VLA、DINOv2、V-JEPA 2 全程冻结；唯一可训练参数在 selector。

---

## 1. 特征管线（frozen，先建后训）

### 1.1 七路信号与维度预算

| 信号 | 计算 | 维度 | 延迟 |
|------|------|------|------|
| \(\varphi_t\) | DINOv2-B CLS：当前帧 vs 上一 chunk 末帧余弦距离 + 当前帧 embedding 的 PCA-64 投影 | 1+64 | ~10ms |
| \(d_t^{\text{pre}}\) | V-JEPA 2 对 K 个候选各做短程（chunk 长度）latent rollout：候选间终态两两散度的 {mean, max}、每候选终态与近期成功轨迹参考库最近邻距离 | 2+K | ~40ms（批量） |
| \(d_t^{\text{post}}\) | 上一 chunk 的 WM 预测终态 latent vs 实际观测 latent 的 L2（滞后一拍） | 1 | 复用 |
| \(\sigma_t\) | \(d^{\text{pre}}\) rollout 的多噪声采样方差 | 1 | 复用 |
| \(c_t^{\text{ACC}}\) | 相邻决策点候选分布的动作不一致度（重叠时段动作差的 RMS） | 1 | ~0 |
| \(v_t^{\text{probe}}\) | frozen 特征（DINOv2 或 SmolVLA 内部）上线性 probe 的 MC-outcome 读数：当前状态值 + 每候选执行首步后的预估值 | 1+K | ~0（线性层） |
| \(p_t^{\text{fail}}\) | SAFE 式失败概率（VLA 内部特征 + 轻量头，用 NGC-Plus 采集轨迹训练） | 1 | ~0 |
| 上下文 | episode 进度 t/H、上次 fallback one-hot、连续 fallback 计数、任务 suite one-hot | ~12 | 0 |

拼接后 \(x_t\in\mathbb{R}^{\approx 90+2K}\)（K=8 时约 106 维）+ 每候选特征 \([d^{\text{pre}}_i, v^{\text{probe}}_i, \text{action 统计}]\)。

**部署一致性红线**：训练与部署的特征集合完全一致；state-fork 仅用于 reward / 标注，绝不进特征（设计报告 §4.1）。

### 1.2 value probe 与 SAFE 头的预训练（W10 第一件事）

两者都是廉价监督学习，用 NGC-Plus 采集副产品（8 万条带 outcome 的续完轨迹）训练：

```python
# value probe: 岭回归 / 单线性层，目标 = 该状态出发的 MC outcome
probe = Ridge(alpha=10).fit(feats_frozen, mc_returns)
# SAFE 头: 2 层 MLP, 目标 = episode 最终失败的逐步标签（时间加权）
```

按任务留出验证：报 held-out suite 上的 AUROC，确认非任务记忆。

### 1.3 成功轨迹参考库

从 clean LIBERO 成功 episode 抽 V-JEPA 2 latent 关键帧（每任务 ~50 帧），FAISS 建索引供 \(d^{\text{pre}}\) 的最近邻距离查询。这是 optimism-bias 的部分对冲：不问 WM"会不会成功"，而问"预测终态离成功流形多远"。

---

## 2. 动作空间与 fallback 执行器

### 2.1 统一动作空间

\(\mathcal{U}(s)=\{a_1..a_K\}\cup\{\text{ROLLBACK},\text{RESAMPLE},\text{REPLAN},\text{WAIT},\text{ABSTAIN}\}\)，K=8 时 13 维离散。

### 2.2 各 fallback 的确切实现（可测试的单元）

| fallback | 实现 | 终止/成本语义 |
|----------|------|--------------|
| 执行候选 \(a_i\) | 执行该 chunk | 正常推进 |
| ROLLBACK | 回放最近 \(m\le 2\) 个 chunk 的逆动作（增量动作系取负、gripper 特殊处理：若期间发生抓/放则先恢复 gripper 状态）；仅对 \(\rho=\text{reversible}\) 有语义保证 | cost 中；连续 ROLLBACK ≤ 2 次强制转 RESAMPLE |
| RESAMPLE | 换噪声种子重采 K 个候选（温度 +0.2），本步不执行 | cost 低；连续 ≤ 2 次 |
| REPLAN-goal（主） | 从预生成 milestone 库（B2FF 式，episode 开始时由 clean 初始观测想象/检索）选一个中间视觉目标，切换 VLA 条件重采候选 | cost 中 |
| REPLAN-text（对照） | 规则/小 LLM 改写指令（简化、加空间限定词），重采候选 | cost 中 |
| WAIT | 空动作 1 个 chunk 时长（等待动态扰动稳定） | cost 低；连续 ≤ 3 |
| ABSTAIN | 终止 episode，判 abstain（不计成功不计破坏） | 终止 |

每个 fallback 写成独立可单测的类（给定 env snapshot → 断言行为），W10 用 20 个手工挑选状态做 fallback 单元验收。

### 2.3 动作屏蔽（action masking）

不可用动作在 Q 值上置 \(-\infty\)：ROLLBACK 在 episode 前 2 chunk 不可用；连续计数超限屏蔽对应 fallback；ABSTAIN 在 episode 前 25% 进度屏蔽（防早退刷分）。masking 在训练与部署一致。

---

## 3. Selector 网络（3.5M）

```python
class Selector(nn.Module):
    def __init__(self, d_ctx=106, d_cand=16, K=8, n_fb=5, d=256):
        self.ctx_enc  = MLP(d_ctx, [512, d])            # 上下文塔
        self.cand_enc = MLP(d_cand, [64, d])            # 候选塔（K 个共享权重）
        self.attn     = nn.MultiheadAttention(d, 4)     # ctx 对候选做一层 cross-attn
        # Dueling: V(x) + A(x,u)
        self.V   = MLP(d, [128, 1])
        self.A_c = MLP(2*d, [128, 1])                   # 每候选 advantage（共享）
        self.A_f = MLP(d, [128, n_fb])                  # fallback advantages
    def forward(self, ctx, cands):                     # → Q ∈ R^{K+5}
        h = self.ctx_enc(ctx); c = self.cand_enc(cands)
        h = h + self.attn(h, c, c)
        A = torch.cat([self.A_c(cat(h, c_i)) for c_i], self.A_f(h))
        return self.V(h) + A - A.mean(-1, keepdim=True)
```

候选塔共享权重使 K 可变（cross-backbone 时不同 K 直接复用）。参数量核算控制在 3.5M ± 0.5M。

---

## 4. 训练配比 D5：offline warm-start → online DQN

### 4.1 Offline warm-start（W10–W11）

**数据**：NGC-Plus 采集副产品直接转换为 selector 决策数据集：

- 每个已标注状态天然给出"oracle 最优决策"：Set A/B → 执行 \(\arg\max_i \hat r_i\)；Set C → 逐个 fallback 用 state-fork 实测其后续成功率（每 fallback 3–5 次续完；这部分是 warm-start 的增量采集，计入 \(B_{\text{selector}}\)），取最优 fallback 为标签；全 fallback 皆 < 0.2 → ABSTAIN。
- 产出 \((x, u^\*, \{\hat q_u\})\) 三元组，约 4,000 状态 × 有效决策 ≈ 1–2 万条。

**目标函数**：BC 交叉熵 + CQL 保守项：

\[
\mathcal{L} = \text{CE}(\pi_\theta(x), u^\*) + \lambda_{\text{CQL}}\Big(\log\sum_u e^{Q(x,u)} - Q(x, u^\*)\Big),\qquad \lambda_{\text{CQL}}=1.0 \text{ 起点}.
\]

**验收**：held-out 状态上 top-1 决策一致率 ≥ 60%、Set C 上"选了 fallback（而非候选）"的召回 ≥ 80%，达标才进 online 阶段。

### 4.2 Online DQN（W11–W13）

| 项 | 设定 |
|----|------|
| 算法 | Double + Dueling DQN，n-step=3 |
| replay | 优先经验回放（PER，α=0.6, β 0.4→1.0），容量 200K transitions，offline 数据以固定 20% 混采 |
| target | 软更新 τ=0.005 |
| 探索 | ε 0.3 → 0.05（前 3K episodes 线性），加 fallback 侧 ε-boost（fallback 动作探索概率下限 0.02，防塌缩到只执行候选） |
| 优化 | Adam 3e-4，Huber loss，grad clip 10，reward clip [-2, 2] |
| batch | 256，每环境步 1 次更新（update-to-data 比按吞吐调） |
| 课程 | 前 30% 训练只喂 Set B 富集的初始状态分布（可救的失败），之后混入全分布——直接从 Set C 学会导致 ABSTAIN 崩溃 |

### 4.3 奖励（设计报告 §6.4 落地）

\[
R = R_{\text{task}} \;-\; \alpha_b\,\mathbb{1}[\text{broken-success}] \;-\; \alpha_c \sum \text{cost}(u) \;-\; \alpha_a\,\mathbb{1}[\text{ABSTAIN}\land\text{oracle 判可救}],
\]

- \(R_{\text{task}}\in\{0,1\}\) episode 末；broken-success 判定用训练时的 state-fork 反事实（该状态 \(\max_i \hat r_i \ge \tau\) 但 fallback 后失败）——**只在训练 reward 中使用 fork，部署不需要**；
- cost：RESAMPLE/WAIT 0.02、ROLLBACK/REPLAN 0.05、ABSTAIN 0.1（基准值，E10 对 \(\alpha_b\in\{0.1,0.3,1,3\}\) sweep）；
- \(\alpha_a\) 项与 ABSTAIN 屏蔽共同构成 abstain 崩溃的双保险。

### 4.4 预算记账（§6.7 落地）

计数器写进采样器：`B_selector = B_warmstart_fallback_forks + B_online_episodes`。每 500 episodes 落一次 checkpoint + 记账快照，E3 的 2K/4K/8K/16K 检查点评测直接从这些 checkpoint 取。

---

## 5. 并行采样 infra（借鉴 SimpleVLA-RL 训推渲一体思路，单卡版）

```
GPU (4090, 分时三租户)
 ├── SmolVLA 批量候选生成（主吞吐占用）
 ├── DINOv2 + V-JEPA 2 特征批推理
 └── selector 前向/训练（很小）
CPU
 ├── 8 × env worker（LIBERO-Plus，EGL offscreen 渲染）
 ├── replay buffer 进程 + 训练数据装载
 └── 记账/日志进程
```

- 三租户经同一个批处理调度器（收集各 worker 请求 → 按模型分组 batch → 一次前向）；SmolVLA 0.45B + 两个 ViT 合计 < 6GB，与训练共存无压力。
- 吞吐预估：8 env、每 chunk 决策一次（约 1s 模拟时长），期望 3–5 episodes/min → 16K episodes ≈ 3–4 天墙钟。W11 实测校正。
- 断点续训：optimizer/replay/记账三者同步 checkpoint。

---

## 6. 稳定性预案（症状 → 处置）

| 症状 | 检测指标 | 处置 |
|------|---------|------|
| Q 值发散 | mean|Q| 持续上行 | 已有 Huber+clip+软更新；再犯降 lr、n-step→1 |
| ABSTAIN 崩溃 | ABSTAIN 占比 > 30% | 提高 \(\alpha_a\)、检查课程是否过早喂 Set C |
| fallback 失活（退化成 reranker） | fallback 选择率 < 2% 且 Set C 上 FEB→1 | fallback ε-boost 提高；检查 warm-start 是否被 online 冲掉（提高 offline 混采比） |
| PER 过拟合少数状态 | 高优先级样本集中度 | β 加速退火、优先级上限截断 |
| 特征尺度漂移 | 各信号 running stats | 全信号进 RunningNorm，冻结于 warm-start 结束时 |
| 训练/评测 gap | eval 成功率 << 训练 | 检查 ε 未关、masking 不一致、RunningNorm 泄漏 |

---

## 7. 评测与日志协议

### 7.1 固定评测集

三个互斥集合，训练全程不碰：

1. **E1 主评测**：LIBERO-Plus camera/robot L3+ 与组合扰动的任务级 held-out 子集（按任务切，非按状态切，防状态级泄漏）；
2. **NGC-Plus Set C-consensus**：FEB / net-success / broken-success；
3. **Set A/B 评测**：clean-regret（fallback 在可救状态造成的损失）。

### 7.2 每次评测的标准输出（对接 NGC-Plus protocol/eval_feb.py）

```
run_id, ckpt, budget_at_eval,
E1: success@suite×dim×level (mean±std, 3 seeds)
SetC: FEB, net-success, fallback 分布直方图
SetAB: clean-regret, broken-success
决策日志: 每决策点 (state_key, Q values, chosen u, mask) → 供 per-fallback ablation 复分析
```

### 7.3 必做诊断图（论文素材）

1. 样本效率曲线（E3 主图：三臂 × 预算检查点）；
2. Set A/B/C 上七路信号的分离度（E6：AUROC 条形图，含 \(d^{\text{pre}}\) 的 optimism-bias 专项）；
3. fallback 使用率 vs 扰动维度热力图（哪类扰动触发哪类恢复——因果分析的 selector 侧呼应）；
4. \(\alpha_b\) sweep 的 net-success vs broken-success 帕累托前沿（E10，rule/agent 基线各是前沿外一个点）。

---

## 8. Baseline 实现清单（E8，与本框架共享 fallback 执行器）

| baseline | 决策器 | 复用本框架的部分 |
|----------|--------|------------------|
| VoLo-lite | VLM（如开源 7B 级）在同一 \(\mathcal{U}\) 上 deliberation 选择 | 全部 fallback 执行器、评测协议（最干净对照） |
| HELM-lite | rule：失败检测阈值触发 → 固定 rollback-to-checkpoint + replan | ROLLBACK/REPLAN 执行器 |
| B2FF-lite | milestone selection（其开源实现精神的复现） | REPLAN-goal 的 milestone 库 |
| CycleVLA-lite | VLM 失败检测 + backtrack/replan 二选一 rule | 执行器 |
| FOREWARN-lite | V-JEPA 2 rollout 打分在候选内强制选择 | 特征管线（其 FEB 恒为 1 的实测样本） |
| SAFE+Abstain / rerankers | 检测阈值 → 停 / 候选内重排 | 特征管线 |

统一原则：**所有 baseline 与我们共享同一 fallback 执行器与评测协议，只替换决策器**——这使 E8 的每一行差异都可归因于决策器本身，是 learned-vs-rule/agent 论证的方法论根基。

---

## 9. 里程碑验收（对齐 18 周计划）

- W10 末：特征管线全通（单决策点端到端延迟 < 150ms）、fallback 单元测试 20/20、probe/SAFE 头 AUROC 报告。
- W11 末：warm-start 验收达标（§4.1）；online 训练启动并稳定 1K episodes 无发散。
- W13 末：E1 首版数字 + E6 诊断图；决定是否触发降级序。
- W16 末：E3 记账表 + E8 五 baseline（或降级后三 baseline）完成。
