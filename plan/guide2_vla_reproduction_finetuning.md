# 施工指南二：VLA 复现与微调全流程（超详细版）

配套文档：RASE-Lite 设计报告 v3.1 §2、§8-E3。覆盖三条轨道：主线 SmolVLA-0.45B 复现与自测、副线 OpenVLA-OFT / OFT+ 推理部署、以及 E3 所需的 SmolVLA LoRA policy-RL 基线微调。

> **2026-07-27 优先级更新：** baseline/OFT 部署与版本锁定部分继续有效；LoRA
> policy-RL 暂停。当前先构建 Smol→Smol、Smol→OFT、OFT→OFT 的 paired
> recoverability matrix，并用 frozen policy 完成三臂 selector gate。只有简单
> selector 在 task-held-out 上优于 matched random-trigger 后，LoRA 才作为
> matched-budget training-side baseline 恢复优先级。当前计划见
> [`RASE_top_conference_execution_v4.md`](RASE_top_conference_execution_v4.md)。

> **W5 后决策：** SmolVLA proposal temperature `0.3/0.7/1.0` 在冻结 cohort
> 上合计 `0/576`，且候选差异/多样性核查通过。不得继续扩大温度或在 L3–L5
> 重复同类 rollout。当前 VLA 侧唯一新增实验是 L1–L2 paired policy matrix：
> 先固定 SmolVLA proposal，再分别用 SmolVLA 与 suite-matched OFT continuation。

> 实施事实（2026-07-17）：仓库使用 `RASE/`、Python 3.12、LeRobot 0.5.1，
> SmolVLA 主评测固定 `num_steps=10`、`n_action_steps=10`、`batch_size=2`，
> LIBERO-Plus 固定 `4976dc3`。下文旧路径、Python 3.10、RTX 4090 与
> batch size 8 仅为示例；执行时以 `env.lock.md` 与 `configs/` 为准。

**总原则**：本项目所有 VLA 均以"官方 checkpoint + 冻结推理"为主，唯一涉及训练的是 E3 的 LoRA 基线。复现的目标不是打平论文数字，而是**得到一个被完整记录、可被审稿人重跑的自测 baseline**。

---

## 1. 轨道一：SmolVLA-0.45B 复现与自测（W1，主线）

### 1.1 已知事实与预期管理

- 论文报告：Spatial/Object/Goal/Long ≈ 90/96/92/71，均值 87.3%。
- 官方 checkpoint `HuggingFaceVLA/smolvla_libero` 的社区复现：object ≈ 93%、goal ≈ 81%、spatial ≈ 63–73%、long ≈ 43–56%，总体 ≈ 73%。**复现不到论文数字是社区普遍现象，不是你的 bug**；我们的策略是自测、锁配置、报自测数。
- 结果对以下配置极度敏感（社区 issue 与 vla-eval 论文的合并教训）：
  1. `--policy.n_action_steps`（1 vs 10 差异巨大；有 issue 显示 n_action_steps=1 时 Long 仅 43%，调整后 51%）；
  2. lerobot 版本（0.5.x 系列间行为有变化）；
  3. 数据集 / 环境版本（`HuggingFaceVLA/libero` v3.0 数据集、mujoco 3.3.x）；
  4. flow-matching 推理步数 `--policy.num_steps`；
  5. 观测 key 命名与 proprio 来源、绝对/增量动作模式（vla-eval：错配可造成 0% 或 55pp 波动）。

### 1.2 安装与锁定

```bash
conda create -n smolvla python=3.10 -y && conda activate smolvla
pip install "lerobot[libero]==0.5.1"        # 版本以 W1 实测最稳者为准，锁定后写 env.lock.md
huggingface-cli download HuggingFaceVLA/smolvla_libero --local-dir ckpts/smolvla_libero
```

### 1.3 官方评测命令（基准配置，全项目统一）

```bash
lerobot-eval \
  --policy.path=ckpts/smolvla_libero \
  --policy.num_steps=10 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.n_episodes=50 \
  --eval.batch_size=8 \
  --policy.device=cuda \
  --output_dir=runs/smolvla_baseline_$(date +%m%d)
```

要点：

- **每任务 ≥ 50 episodes**（社区 issue 里 n_episodes=10 的数字方差极大，不可用作论文 baseline）。四 suite × 10 任务 × 50 = 2,000 episodes，4090 上约 1–2 天。
- 固定全套随机种子并跑 **3 个种子**，报均值 ± 标准差。
- 同一配置在 clean LIBERO 与 LIBERO-Plus 上各跑一遍（LIBERO-Plus 侧只需 pip 环境切换，任务名经 `task_classification.json` 选取子集：先跑 camera/robot × L1–L5 的分层子集，每格 ≥ 20 episodes，得到自己的"塌缩曲线"——这张图会进论文 §1）。

### 1.4 自测报告模板（每个 backbone 一份，进论文附录）

```
backbone: smolvla-0.45B  ckpt: HuggingFaceVLA/smolvla_libero@<hash>
lerobot==0.5.1  mujoco==3.3.2  LIBERO-plus@<commit>  GPU: RTX4090  seeds: {0,1,2}
n_action_steps=10  num_steps=10  n_episodes=50/task
结果表: suite × (clean, plus-camera-L1..L5, plus-robot-L1..L5, combo) 
已知与论文报告的差异及原因猜测: ...
```

### 1.5 排障决策树

1. Long suite 显著低于 43% → 检查 `n_action_steps`（应为 10）与推理步数；检查数据集版本是否 v3.0。
2. 全 suite 接近 0% → 动作模式错配（绝对 vs 增量）或观测 key 错位；打印一条轨迹的动作值域检查。
3. 单 suite 异常 → 环境资产版本；用官方 issue 里他人复现数字对表定位。
4. 数字与他人复现一致但低于论文 → 接受，报自测数（这正是预期路径）。

---

## 2. 轨道二：OpenVLA-OFT 7B / OFT+ 推理部署（W1–W2，副线 + oracle + 诊断轨）

OFT 在本项目承担三个角色：E2 副线评测、Step 3 续完 oracle、E5 诊断轨（OFT+）。三者共用同一套推理服务。

### 2.1 安装

```bash
conda create -n oft python=3.10 -y && conda activate oft
git clone https://github.com/moojink/openvla-oft && cd openvla-oft
git checkout <COMMIT> && pip install -e .
pip install flash-attn --no-build-isolation     # 4090 支持；失败则退回 sdpa
# 官方 LIBERO checkpoints（四 suite 各一个，或统一版，以 repo README 为准）
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-<suite> --local-dir ckpts/oft_<suite>
# OFT+ / 变体 checkpoint 从 LIBERO-plus 官方仓库 README 指引获取
```

显存预算：7B bf16 权重约 14GB + KV/激活，4090 24GB 可承载 batch 4–8 的图像推理；OFT 的并行解码（一次前向出整个 action chunk）对吞吐友好。

### 2.2 推理正确性核验（vla-eval 教训清单，逐条打钩）

- [ ] 四元数→轴角转换的 antipodal 规范化与训练侧一致（vla-eval：此单项错配使 Goal 97→83、Long 95→56）；
- [ ] 评测时 center crop 与训练增强匹配（OpenVLA 系：省略 crop 约损失 3pp）；
- [ ] 图像归一化 / 分辨率 / 相机名与 checkpoint 卡片一致；
- [ ] unnorm_key（动作反归一化统计）选对 suite；
- [ ] 温度：E2 评测用官方默认（贪心/低温）；**oracle 续完用 temp=0.5**（设计报告 D6），两种模式的配置分开固化。

核验方式：clean LIBERO 每 suite 跑 20 episodes，对齐官方报告数字（97% 级别）±3pp 以内即通过；不通过先查上面清单再查版本。

### 2.3 Oracle 推理服务化

Step 3 的续完需求是"多个 CPU env worker 持续请求动作"。将 OFT 包装为常驻服务：

```python
# oracle_server.py — zmq REP 服务，批量聚合
while True:
    batch = collect_requests(max_batch=8, timeout_ms=15)   # [(env_id, obs, instr)]
    actions = oft.predict_chunk_batch([b.obs for b in batch], [b.instr for b in batch],
                                      temperature=0.5)
    for b, a in zip(batch, actions): reply(b.env_id, a)
```

吞吐目标：batch-8 时单次前向 ≤ 300ms（含预处理），折合每 env 有效 40ms/chunk 级别。W2 实测并记录，此数字直接决定 §3.6 成本重估。

---

## 3. 轨道三：SmolVLA LoRA policy-RL 基线微调（W14–W15，E3 用）

E3 需要一个"直接 RL 微调 VLA"的对照臂。诚实声明：这是 SmolVLA 上的单卡 LoRA 实现，非 SimpleVLA-RL 原版（8×A800 + OFT）复现；实现思路借鉴其开源码。

### 3.1 阶段 A：LoRA-SFT 冷启动（可选但推荐）

SimpleVLA-RL 的经验是极少量 SFT 冷启动 + RL 主体。我们对应做法：

```bash
lerobot-train \
  --policy.path=ckpts/smolvla_libero \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --steps=5000 --batch_size=32 \
  --policy.optimizer_lr=1e-4 \
  --output_dir=runs/lora_sft_warmup
```

注意：lerobot 若无原生 LoRA 开关，用 peft 包 monkey-patch action expert 的 attention/MLP 线性层（rank=16, alpha=32, dropout=0.05），**冻结 VLM backbone，只训 action expert 的 LoRA**——这既省显存也符合 SmolVLA 官方微调惯例（backbone 冻结）。可训练参数量记录进 E3 记账表（预计 ~8–15M，与 selector 3.5M 同量级可比）。

### 3.2 阶段 B：在线 RL（GRPO 风格，适配 flow-matching 的注意事项）

**关键技术风险（提前知晓）**：PPO/GRPO 的 log-prob 策略梯度不能直接套到 flow-matching 头上；且"压低坏动作概率"类更新会把预测推离动作流形（LeHome 2026 冠军方案的实证教训）。可行路线按优先级：

1. **Reward-weighted / advantage-weighted 回归（AWR 风格）**：采样 rollout → 以 episode return 加权做 flow-matching 目标的加权 SFT。实现最简单、最稳，作为 E3 的默认 policy-RL 臂。
2. GRPO-on-chunks：以 chunk 为决策单元、用高斯化的 flow 输出近似 log-prob——不稳定，只作附录尝试。

训练循环（AWR 版伪代码）：

```
for iter in range(N):
    rollouts = collect(policy, envs, n_episodes=64, temp=0.7)   # 记入 B_policy-RL
    adv = normalize([ep.return for ep in rollouts])             # 组内标准化（GRPO 精神）
    for batch in make_batches(rollouts):
        loss = (w(adv) * flow_matching_loss(policy, batch)).mean()   # w = exp(adv/β) 截断
        loss.backward(); opt.step()
    eval_every(5)
```

超参起点：\(\beta=1.0\)、权重截断 \(w\le 20\)、lr 5e-5、每 iter 64 episodes、总预算与 selector 严格对齐（10–16K episodes，含 SFT 冷启动折算，见设计报告 §6.7 记账规则）。

### 3.3 E3 记账表模板

| 臂 | 可训练参数 | 环境交互（episodes） | 其中 warm-start | GPU·h | LIBERO-Plus 子集成功率 @ {2K, 4K, 8K, 16K} |
|----|-----------|---------------------|-----------------|-------|--------------------------------------------|
| selector-RL（我们） | 3.5M | | oracle rollouts 折算 | | |
| LoRA policy-RL | ~8–15M | | SFT 冷启动折算 | | |
| residual-RL（PLD-lite） | | | | | |

每臂在 2K/4K/8K/16K episodes 四个检查点各评一次同一测试子集——**样本效率曲线是 E3 的主图**，不要只报终点。

---

## 4. 第三 backbone（E11，可裁剪）

若时间允许，选一个 <1B 且 LIBERO 生态成熟的模型（如 VLA-0-Smol 线）走 §1 同样流程：官方 ckpt → 自测 → frozen 接入 selector（不重训 selector，直接测跨 backbone gain）。降级序中最先砍，W16 前不投入。

---

## 5. 全轨道通用纪律

1. **一处配置，处处引用**：所有评测入口读同一个 `configs/eval_base.yaml`，禁止命令行散落覆盖；每次实验落一份 config 快照到输出目录。
2. **三个环境隔离**：smolvla / oft / rl 三个 conda 环境互不污染（lerobot 与 openvla-oft 的依赖树有冲突风险）。
3. **每个数字可溯源**：论文里出现的每个成功率都能指回一个 `runs/<name>/` 目录（含 config、seed、日志、每 episode 结果 CSV）。
4. **W1 结束时的验收物**：SmolVLA 四 suite 自测表（3 seeds）、OFT clean 对齐核验记录、LIBERO-Plus 塌缩曲线初版、`env.lock.md`。
