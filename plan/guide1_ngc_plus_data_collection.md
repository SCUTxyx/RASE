# 施工指南一：NGC-Plus 数据采集全流程（超详细版）

配套文档：RASE-Lite 设计报告 v3.1 §3。本指南覆盖从环境搭建到 benchmark 发布物打包的每一步，含目录结构、命令、代码骨架、数据 schema、QC 流程与排期。

---

## 0. 总览：产出物与流水线

**最终产出物（NGC-Plus 发布包）**：

1. `states/`：约 4,000 个扰动状态（Set B ≈ 2,000 + Set C ≈ 2,000），每个含 MuJoCo 快照 + 观测帧 + 扰动元数据。
2. `annotations/`：per-state、per-candidate 的可恢复性标注 \(\hat{r}(s,a_i)\)、Wilson 置信区间、Set A/B/C 判定、物理可逆性标签 \(\rho(s)\)、cross-oracle 判定与 \(\kappa\) 统计。
3. `candidates/`：每状态 \(K=8\) 个候选 action chunk（主 oracle 用 SmolVLA 生成，副本用 OFT）。
4. `protocol/`：FEB 评测脚本 + 报告模板（FEB、broken-success、net-success、clean-regret）。
5. `analysis/`：扰动类型 → NGC 产率因果分析表与图。

**流水线六步**：

```
Step 1 扰动状态生成 → Step 2 候选生成 → Step 3 state-fork 续完（自适应采样）
→ Step 4 三分与筛选 → Step 5 因果分析 + 可逆性标注 → Step 6 cross-oracle + 人工核查 + 打包
```

---

## 1. 环境搭建与版本锁定（W1，约 2 天）

### 1.1 版本 pin 原则

复现敏感性是这个项目最大的隐性风险源（vla-eval 记录了单参数 55pp 波动的案例）。**所有版本号写入 `env.lock.md` 并进 git，此后任何升级需走变更记录。**

```bash
# 建议的 conda 环境
conda create -n ngcplus python=3.10 -y
conda activate ngcplus

# 固定核心依赖（示例版本，W1 第一天以实际可用的最新稳定版为准并锁定）
pip install "mujoco==3.3.2" "robosuite==1.4.1"
pip install "lerobot==0.5.1"          # SmolVLA 主线
# LIBERO-Plus：pip 直换原版 libero
git clone https://github.com/sylvestf/LIBERO-plus && cd LIBERO-plus
git checkout <COMMIT_HASH>            # 锁定 commit，写入 env.lock.md
pip install -e .
```

### 1.2 验收测试（必须全绿再往下走）

```bash
python -c "import libero; print(libero.__file__)"   # 确认指向 LIBERO-plus
python scripts/smoke_test.py   # 见 1.3
```

`smoke_test.py` 需要验证四件事：

1. 能按 `task_classification.json` 加载指定扰动维度/难度档的任务并 reset；
2. `env.sim.get_state()` / `env.sim.set_state()` 快照-恢复往返后观测逐像素一致（这是 Step 3 fork 的根基，**必测**；若 robosuite 封装层有额外缓存状态——如 controller 内部状态——需要一并序列化）；
3. SmolVLA checkpoint 能加载并对一帧观测出动作；
4. 一条完整 episode 能跑通并正确返回 success 标志。

### 1.3 fork 正确性专项测试（最容易踩坑处）

MuJoCo 的 `set_state` 不覆盖 controller 内部状态（如 OSC 的积分项）、随机数种子、以及 lerobot policy 侧的观测历史缓存 / action queue。**fork 必须重置三层状态**：

```python
class ForkableEnv:
    def snapshot(self):
        return {
            "sim_state": self.env.sim.get_state().flatten().copy(),
            "controller_state": copy.deepcopy(self.env.robots[0].controller.__dict___relevant()),
            "rng_state": self.env.np_random.bit_generator.state,
            # policy 侧缓存由调用方负责 reset（policy.reset()）
        }

    def restore(self, snap):
        self.env.sim.set_state_from_flattened(snap["sim_state"])
        self.env.sim.forward()
        self._restore_controller(snap["controller_state"])
        self.env.np_random.bit_generator.state = snap["rng_state"]
```

验收标准：同一 snapshot restore 两次、执行同一确定性动作序列 50 步，两次的观测序列逐像素一致、物体位姿差 < 1e-9。**不达标就不要开始采集**——fork 漂移会污染整个 \(\hat{r}\) 标注。

---

## 2. Step 1：扰动状态生成（W1–W2）

### 2.1 采样策略

目标：约 20,000 个 `(state, 扰动标签)` 候选池，向 NGC 富集区倾斜。

| 维度 | 配额 | 理由 |
|------|------|------|
| camera viewpoint × L3–L5 | 30% | LIBERO-Plus 证实最致命维度之一 |
| robot initial state × L3–L5 | 30% | 同上（OFT 该维度 31.9%） |
| camera × robot 组合扰动 | 20% | 负组合泛化 → NGC 密度最高的区域 |
| layout（confounder + 目标位移）× L3+ | 10% | 多样性 |
| 其余维度（light/background/noise）× L4+ | 10% | 覆盖完整性，防"cherry-pick 维度"攻击 |

任务 suite 分布：LIBERO-Long 40%（长时程 NGC 更丰富）、Goal 25%、Spatial 20%、Object 15%。

### 2.2 状态采集时机：不只采 t=0

**关键设计**：NGC 状态大量出现在执行中途（错误累积后），不只在初始帧。采集协议：

1. 对每个扰动任务，让主 VLA（SmolVLA）自主执行；
2. 每隔 \(\Delta=2\) 个 action chunk 打一个 snapshot；
3. episode 结束后，**失败 episode 的所有 snapshot 全部入池，成功 episode 随机保留 20% 的 snapshot**（作 Set A 对照与 clean-regret 评测用）；
4. 每个 snapshot 记录：sim state、三视角观测帧、proprio、任务指令、扰动标签（维度/子维/难度档）、episode 内时间步、episode 最终结局。

### 2.3 数据 schema（每个状态一个目录）

```
pool/{task_id}/{episode_id}/{step_id}/
├── sim_state.npz            # flatten 后的 MuJoCo 状态 + controller 状态 + rng
├── obs_agentview.png
├── obs_wrist.png
├── proprio.npy
├── meta.json                # {task, instruction, perturb_dim, perturb_sub, level,
│                            #  step, episode_outcome, snapshot_version}
```

### 2.4 产能估算与实测点

按每 episode 平均 300 步、渲染 + SmolVLA 推理约 15–25 ms/步估算，单 episode 5–8 秒纯执行 + 环境 reset 开销。20,000 状态约需 3,000–4,000 episodes（每 episode 出 5–7 个入池 snapshot），单卡 4090 + 8 个 CPU 并行 env 约 **1–2 天墙钟**。W2 第一天实测校正。

---

## 3. Step 2：候选生成（W2）

对池中每个状态生成 \(K=8\) 个候选 chunk：

```python
def generate_candidates(policy, obs, K=8, temp=0.7):
    cands = []
    for k in range(K):
        policy.reset()                       # 清空 action queue / obs 历史
        seed_everything(BASE_SEED + k)
        chunk = policy.sample_chunk(obs, temperature=temp)  # flow-matching:
        cands.append(chunk)                  # 对初始噪声采样即天然多样
    return np.stack(cands)                   # [K, chunk_len, action_dim]
```

要点：

- SmolVLA 是 flow-matching 头，多样性来自初始噪声采样；温度语义用噪声 scale 实现。**W2 需做一个候选多样性 sanity check**：K=8 候选的末端位移两两平均 L2 距离分布，若坍缩（多样性过低）调大噪声 scale——候选无多样性会使 Set A/B/C 划分失真（一切状态都趋向 all-good 或 all-bad）。
- 副本：同一状态用 OFT 也生成 8 候选，存 `candidates_oft/`，供 cross-oracle 与 E2。
- 存储：`candidates/{state_key}.npz`，含 `[K, T, 7]` 动作、生成温度、种子。

---

## 4. Step 3：state-fork 续完与自适应采样（W3–W7，最贵一步）

### 4.1 单元操作

```python
def evaluate_candidate(env, snap, candidate_chunk, cont_policy, horizon):
    env.restore(snap)
    cont_policy.reset()
    for a in candidate_chunk:                # 先执行候选
        obs, _, done, info = env.step(a)
        if done: return info["success"]
    while not done and t < horizon:          # 再交 oracle 续完
        a = cont_policy.act(obs)             # OFT, temp=0.5
        obs, _, done, info = env.step(a)
    return info["success"]
```

### 4.2 两阶段自适应采样（v3.1 判据）

```python
from statsmodels.stats.proportion import proportion_confint

def adaptive_r_hat(env, snap, cand, oracle, tau=0.5, n1=3, n2=10):
    succ = [evaluate_candidate(env, snap, cand, oracle, H) for _ in range(n1)]
    lo, hi = proportion_confint(sum(succ), n1, alpha=0.05, method="wilson")
    if hi < tau or lo > tau:                 # 区间整体在阈值一侧 → 早停
        return sum(succ)/n1, n1, (lo, hi)
    for _ in range(n2 - n1):                 # 边界状态补采
        succ.append(evaluate_candidate(env, snap, cand, oracle, H))
    lo, hi = proportion_confint(sum(succ), n2, alpha=0.05, method="wilson")
    return sum(succ)/n2, n2, (lo, hi)
```

**Set C 判据（保守）**：状态 \(s\) 判为 Set C 当且仅当所有候选的 Wilson 95% 上界 \(<\tau\)。Set A：存在 \(\ge 3\) 个候选下界 \(>\tau\)。Set B：其余存在可接受候选的状态。三类之外（区间全跨界、无法定案）标 `uncertain`，不进发布集但保留数据。

### 4.3 并行执行架构

```
主进程（调度 + GPU 推理服务）
 ├── GPU worker: OFT 7B 批量推理（batch 所有并行 env 的当前 obs）
 └── N=8 个 CPU env worker（各持一份 ForkableEnv），经 zmq/multiprocessing 队列交换 obs/action
```

- OFT 推理 batch 化是吞吐关键：8 env 同步 step、集中一次前向。
- 4090 24GB 放 OFT 7B（bf16 约 14GB）+ 渲染，余量放 batch。SmolVLA cross-oracle 阶段吞吐约快 3 倍。
- 断点续采：每完成一个 `(state, candidate)` 立即落盘 `results/{state_key}/{cand_idx}.json`，调度器启动时扫描已完成项跳过。**采集会跑一周以上，断点续采不是可选项。**

### 4.4 成本核算与在线监控

- 期望 rollout：\(\approx 2{,}500\)（候选 Set C 目标超采）\(\times 8 \times \mathbb{E}[N] \approx 82{,}000\) 条 + Set B 部分。
- W3 第一天实测 20 条 rollout 的真实墙钟（含 fork 恢复、渲染、推理、续完），据此重估总时长并回填 v3.1 §3.6。
- 在线看板（简单 CSV + 每日脚本即可）：累计 rollout 数、Set A/B/C/uncertain 计数、各扰动维度 NGC 产率、平均 \(N\)。**若 NGC 产率 < 8%，立即上调采样池中 L4–L5 与组合扰动的比例**，不要等采完再发现 Set C 不够。

---

## 5. Step 4–5：三分、因果分析、物理可逆性标注（W7–W8）

### 5.1 三分与去偏检查

- 按 4.2 判据切 Set A/B/C。
- 检查 Set C 的任务分布：若某单一任务贡献 > 15% 的 Set C，检查是否环境 bug / oracle 系统性盲区，人工抽看 10 条视频。
- 检查 Set C 的 episode 时间步分布：若全部集中在 episode 末段，说明大量是"已经不可逆的残局"，信息量低——用 §5.3 可逆性标签分层报告。

### 5.2 扰动 → NGC 产率因果分析

产出一张主表 + 两张图：

- 表：`(扰动维度 × 难度档) → {状态数, NGC 数, NGC 产率, 95% CI}`；组合扰动单列，报告交互效应（组合产率 vs 独立产率之积的偏离，呼应 LIBERO-Plus 的负组合泛化）。
- 图 1：产率热力图（维度 × 档位）；图 2：NGC 产率 vs 成功率塌缩幅度的散点（回答"掉下去的部分里多少是结构性死局"——这是 benchmark 章节的灵魂图）。

### 5.3 物理可逆性标注 \(\rho(s)\)

自动规则（对每个 Set B/C 状态回看其 episode 前缀）：

1. `task-irreversible`：任务对象已进入不可逆终态（掉出工作区、目标容器倾覆等，按任务白名单谓词判定）；
2. `contact-irreversible`：episode 前缀中检测到非抓取接触事件且相关物体位姿相对初始扰动布局偏移 > 阈值（如 3cm/15°）；
3. `reversible`：其余。

人工抽查 100 个状态校准阈值，报告自动标注与人工的一致率。

---

## 6. Step 6：cross-oracle、人工核查与打包（W8–W9）

### 6.1 cross-oracle

- 对全部 Set C 与 20% 抽样的 Set B，用 SmolVLA（temp 0.5）作第二 oracle 重跑判定（自适应采样同 4.2，预算约为主 oracle 的 1/3——只复核判定而非重估全部 \(\hat r\)：对每个状态只需复核 \(\max_i r\) 与 \(\tau\) 的关系，可从主 oracle 判定最优的 2 个候选开始，若均判不可恢复即可定案）。
- 报告 Cohen's \(\kappa\)；两 oracle 一致的进 `Set C-consensus`（主表），不一致的进 `Set C-oracle-only`（附录单列）。

### 6.2 人工核查（300–500 状态）

- 分层抽样：每个扰动维度 × 每档 ≥ 10 个；Set C-consensus 与 Set C-oracle-only 各占一半。
- 工具：写一个极简回放页（HTML + 视频），标注者看"状态帧 + 8 条候选执行续完视频"，回答两问：(a) 该状态是否存在人类可见的可行恢复路径（三选：明显可救 / 边缘 / 确实死局）；(b) 8 条候选中是否有明显被 oracle 低估的。
- 双人标注 20% 重叠样本，报标注者间一致率；人工 vs oracle 一致率进论文正文。

### 6.3 发布物打包

```
NGC-Plus/
├── README.md               # 版本、许可（建议 CC-BY-4.0 数据 + Apache-2.0 代码）、引用格式
├── env.lock.md
├── states/ candidates/ annotations/
├── protocol/
│   ├── eval_feb.py          # 输入：任意方法在 Set C 上的决策日志；输出：FEB / broken-success / net-success / clean-regret
│   └── report_template.md
├── analysis/                # 因果分析表与图 + 生成脚本
└── splits.json              # consensus / oracle-only / uncertain 划分 + per-state 置信度
```

托管：HuggingFace Datasets（状态 + 标注）+ GitHub（代码 + protocol）。预印本挂出时同步公开 protocol 与一个 100 状态的 preview 子集，全量随论文发布。

---

## 7. QC 检查清单（每周执行）

- [ ] fork 往返一致性抽测（每周 5 个状态重跑）
- [ ] 断点续采完整性：results 目录与调度清单 diff
- [ ] 候选多样性分布未漂移
- [ ] NGC 产率周报 vs 目标（Set C ≥ 2,000 的进度线）
- [ ] 磁盘用量（观测帧是大头；PNG 可换 JPEG-95 + 保留 3 个关键帧全量）
- [ ] 随机抽 10 条续完视频人眼看一遍（oracle 行为是否退化，如原地抖动）

## 8. 常见坑速查

| 症状 | 根因 | 处置 |
|------|------|------|
| 同一状态两次 fork 结果发散 | controller/rng/policy 缓存未重置 | §1.3 三层重置 |
| Set C 产率诡异地高 | oracle 续完 horizon 太短 / oracle 配置错（`n_action_steps` 等） | 先跑 clean 状态标定 oracle 自身成功率 |
| Set C 产率诡异地低 | 候选温度太低（8 候选近乎相同且恰好可行）/ 扰动档位太低 | 候选多样性 check + 上调档位 |
| \(\hat{r}\) 全 0 或全 1 双峰、边界状态极少 | 正常现象（好消息：自适应采样早停率高、成本低于预算） | 无需处置，如实报告分布 |
| 采集速度只有估算 1/3 | 渲染在 CPU 上串行 / OFT 未 batch | 检查 offscreen 渲染后端（EGL）与 batch 聚合 |
