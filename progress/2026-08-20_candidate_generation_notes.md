# RASE 候选动作生成机制说明（与默认动作的关系）

> 2026-08-20 · 基于服务器最新代码（commit 6ab1fc9 修正后的
> `collect_same_root.py` / `rase_selector_loop.py` / `rase/collect/forked_rollout.py`）
> 与 `RASE_IDEA.md §3` 的候选语义定义。

## 1. 候选的五种语义（RASE_IDEA §3，不变定义）

| 候选类型 | 生成方式 | 与"默认动作"的关系 |
|---|---|---|
| `continue.source` | **不生成**：直接继续执行当前 VLA 上一次推理产出的动作块中尚未执行的部分 | **这就是"默认动作"** |
| `requery.source` | 同一边界、同一 VLA **重新推理**（独立 seed / 独立采样）→ 得到一个新动作块 | 默认动作的**同分布重采样**：默认动作（greedy/温度0）只是该分布的一个实现，requery 是另一个实现 |
| `resample.source/candidate.{0,1}` | 同一次推理（同一上下文）的**两个独立采样** | 同上：同一分布内的多个实现 |
| `fallback.persistent` | 另一个（纠正）策略的完整动作块，生成后**持续接管**直到终局 | 默认动作的**异分布候选**：来自不同策略的分布 |
| `abort.safe` | 安全中止（control event，不是零动作伪装） | — |

## 2. 关键关系：默认动作 = 分布的一个实现

```text
同一物理状态 s_t、同一 VLA 的策略分布 π(a | s_t, task)

  temperature = 0（greedy / 众数）
       └──► 默认动作 a*（确定性）
  temperature > 0 或原生随机采样（flow matching 噪声等）
       ├──► 候选 a_1（seed 1）
       ├──► 候选 a_2（seed 2）     ← requery / resample
       └──► ...

不同 VLA 的分布 π_other(a | s_t, task)
       └──► fallback 候选（异分布）
```

- **同策略候选**（requery/resample）与默认动作的关系 = **同一分布的不同实现**。默认动作无特殊地位——它只是"第一个/贪心的实现"。
- **多样性来源**：同策略候选的多样性完全来自推理随机性。确定性策略（temperature=0，如 OFT 的 L1 回归、π0-fast 默认 config `temperature: 0.0`）下，requery 会得到**完全相同的 chunk** → 同策略候选集退化 → 记录 `capability mask`（IDEA §4 规则），不得当普通失败处理。
- **跨策略候选**（fallback）与默认动作的关系 = **异分布**。它不需要源策略有随机性，代价是需要第二个策略。

## 3. 部署时刻（closed-loop）怎么生成候选

`rase_selector_loop.py`（闭环 v2 协议，决策点每 8 步）：

```text
决策点 t：
  snapshot = env.get_sim_state()            # 冻结同一物理状态
  chunks = {m: dual.act(m, observation, task) for m in models}
                                            # 每个模型从同一状态生成一个 8 步 chunk
  X = stack(feats_of(observation, chunks[m]))   # 同状态特征
  mu/sigma = risk_model(X)                  # 风险打分
  chosen = decide(mu, sigma, cfg, state)    # LCB/UCB + abstain（保守规则）
  queue = chunks[chosen]                    # 只执行被选中候选的动作块
```

要点：**所有候选在同一个物理状态上生成（restore 同一 snapshot）**；被选中的候选 = 接下来执行的"默认动作"；风险模型比较的是同根候选的反事实价值。

## 4. 训练数据时刻（same-root 采集）怎么生成候选

`collect_same_root.py`（6ab1fc9 修正后）：

```text
决策点 t：
  snapshot = env.get_sim_state()            # 冻结
  for name in models:                       # 每个候选模型
      restore_env(env, snapshot)            # 恢复同一物理状态
      chunk = act(model, observation, task) # 生成候选 chunk（冻结）
      # candidate_rollout_mode=single_chunk（默认）：
      #   只执行 chunk 的 native 长度（8 步），禁止重复复用冻结 chunk
      #   记录 branch_snapshot = env.get_sim_state()   ← branch-end 快照
      #   （修正：recovery evaluator 从 branch-end 起，而非从 s_t 起）
      # 每个 native chunk 耗尽后 requery 生成新 chunk 继续执行
      记录 s_t / s_{t+H} / chunk_raw / future / consequence_label / recovery
```

**冻结纪律**：候选 chunk 在执行前生成、执行前冻结；保存的数据就是部署 scorer 实际会看到的 chunk；restore 后不重新生成候选（P0 审计已 PASS）。

## 5. 同策略采样候选的现有实现（随机续跑路径）

`rase/collect/forked_rollout.py`：

- `InProcessSmolVLAContinuation`：**stochastic SmolVLA continuation**，`temperature=0.5`（可配置），通过 LeRobot `select_action(temperature=...)` 采样；可选 `seed`，配合 `seed_everything` 可复现；
- `InProcessLeRobotContinuation`（π0-fast / π0.5）：**原生采样**（flow matching 噪声），`temperature=None`（不传），注释原文："Pi0Fast/Pi0.5 sample natively"；
- `rollout_seed(state_key, candidate, rollout, salt)`：每个 (状态, 候选, rollout) 派生确定性 seed → 同状态多候选可复现。

**结论：同策略 requery/resample 的代码路径已存在且可用于 SmolVLA（temperature>0）与 π0-fast（原生采样）；E0 探针只需实测"同状态 2-3 次推理的 chunk 差异"，即可确认候选多样性是否真实存在。**

## 6. 与"默认动作"关联的三个实践含义

1. **默认动作就是 continue 臂**：风险模型比较的是"继续执行当前 chunk" vs "重新生成" vs "fallback"，默认动作是 baseline 臂，不是特权臂。
2. **确定性策略下 continue ≡ requery**：若源策略 deterministic，同策略仲裁无意义（两个候选完全相同）→ 只能靠跨策略候选（fallback 层，已测 FAIL）或换随机源（E0 待测）。
3. **候选质量 ≠ 候选多样性**：temperature 过高会破坏动作质量；需要 E0 实测 diversity-quality 权衡（L2 差异 vs 终局成功率）。
