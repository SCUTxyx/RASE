# RASE-Lite 代码库搭建策略：复用 vs 自研的逐组件分析

配套文档：RASE-Lite 设计报告 v3.1、施工指南一/二/三。
本文回答一个问题：**哪些代码用现成的（base code），哪些必须自己写，整个 repo 怎么组织。**

---

## 0. 结论先行

**不要二选一，采用"官方代码作 frozen 依赖 + 自研薄胶水层"的混合策略：**

1. **所有底座模型与环境（SmolVLA/lerobot、OpenVLA-OFT、LIBERO-Plus、V-JEPA 2、DINOv2）一律用官方代码 + 官方 checkpoint，pin 版本，以"依赖"方式引入，绝不 fork 魔改源码。**
2. **所有项目核心逻辑（ForkableEnv、自适应采样、fallback 执行器、selector、FEB 评测协议、记账系统）100% 自研**——这些正是论文的贡献所在，没有任何现成实现，且必须可被审稿人审计。
3. **RL 训练框架不引入 stable-baselines3 / tianshou 等大框架，自写 cleanrl 风格的单模块 DQN**（理由见 §2.5）。

一句话判据：**"是不是论文贡献/是不是发布物的一部分"决定自研，"是不是被评测对象/被冻结组件"决定复用。**

这个划分同时服务三个硬约束：
- **复现敏感性**（vla-eval 记录单参数 55pp 波动）→ 底座代码不改一行，只 pin 版本，出问题时可与社区数字对表；
- **审稿人可重跑**（设计报告 §2.4 纪律）→ 自研部分必须小、干净、每个数字可溯源；
- **18 周时限** → 不在别人代码里做考古式重构，胶水层薄到一个人能维护。

---

## 1. 逐组件决策总表

| 组件 | 决策 | 引入方式 | 自研内容 | 理由 |
|------|------|---------|---------|------|
| SmolVLA 推理/采样 | **复用** lerobot | `pip install lerobot==<pin>`，不 fork | 温度/噪声 scale 的候选采样 wrapper（`policy.reset()` + 种子控制） | 官方实现是"被评测对象"，改动即失去与社区数字的可比性 |
| SmolVLA LoRA 微调（E3） | **复用 + 外挂** | lerobot 训练入口 + `peft` monkey-patch | patch 脚本（rank/alpha/冻结逻辑）+ AWR 训练循环 | 指南二 §3.1 已定：不改 lerobot 源码，用 peft 从外部注入 |
| OpenVLA-OFT 推理 | **复用**官方 repo | `git clone + pip install -e .`，pin commit | `oracle_server.py`（zmq 批量服务化） | 7B 推理正确性核验清单（四元数规范化、crop、unnorm_key）依赖官方实现原样 |
| LIBERO-Plus 环境 | **复用**，pin commit | pip 直换原版 libero | **ForkableEnv wrapper（自研，全项目最关键的一段代码）** | 快照/恢复三层状态（sim + controller + rng）官方不提供，但以 wrapper 实现，不动其源码 |
| V-JEPA 2 / DINOv2 | **复用**官方权重 | torch hub / 官方 repo | 特征管线（latent rollout、参考库距离、批处理调度） | frozen 特征提取器，零训练 |
| Wilson 区间/统计 | **复用** | `statsmodels.stats.proportion` | 两阶段自适应采样调度逻辑 | 标准库级可靠性，自己写反而有 bug 风险 |
| 参考库检索 | **复用** | `faiss-cpu` | 建库脚本 + 查询封装 | 同上 |
| RL selector（DQN） | **自研** | — | 网络、PER、masking、训练循环、记账（约 1,500–2,500 行） | 见 §2.5 专项分析 |
| Fallback 执行器 | **自研** | — | 5 个 fallback 类 + 单元测试 | 论文贡献本体，无先例实现 |
| 自适应采样 + Set A/B/C 三分 | **自研** | — | `adaptive_r_hat`、判据、断点续采调度器 | benchmark 核心，发布物的一部分 |
| FEB 评测协议 | **自研** | — | `protocol/eval_feb.py` + 报告模板 | 随 NGC-Plus 发布，必须自有、干净、文档化 |
| Baselines（VoLo/HELM/B2FF/CycleVLA/FOREWARN-lite） | **自研**（决策器）+ 共享自研 fallback 执行器 | — | 每个 lite 决策器 100–300 行 | E8 方法论根基：只换决策器，其余全共享 |
| 评测 harness | **半复用** | 考虑 vla-eval harness 校验用 | 主评测入口自写，读统一 `configs/eval_base.yaml` | vla-eval 用作对表工具而非主干，避免被其抽象绑架 |
| value probe / SAFE 头 | **自研（极小）** | sklearn `Ridge` + 2 层 MLP | 训练脚本 + AUROC 报告 | 廉价监督学习，几十行 |

**"绝不 fork 魔改"的三条红线**：
1. 不改 lerobot / openvla-oft / LIBERO-Plus 任何一行源码——需要扩展行为时一律用 wrapper、monkey-patch（集中在一个 `patches/` 目录、有单测）或上游 issue。
2. 若某个 bug 确实必须改上游：fork 到自己的 org、单 commit 修复、pin 到该 commit、`env.lock.md` 记录 diff——这是最后手段，且 diff 必须进论文附录的复现说明。
3. 任何"看起来只是小改"的源码修改都是复现事故源（vla-eval 教训），code review 时一票否决。

---

## 2. 关键决策的展开论证

### 2.1 为什么底座必须复用而非重写

- **可比性**：论文要引用"OFT robot 维度 31.9%（LIBERO-Plus CVPR 2026）"级别的数字，并声称自己的自测 baseline 与社区复现一致（SmolVLA 总体 ≈73%）。只有跑官方代码才具备这种可对表性；自实现的推理管线哪怕 1 个预处理细节不同（center crop 缺失即 −3pp）都会让整条叙事失去锚点。
- **排障成本**：指南二 §1.5 的排障决策树全部建立在"官方代码 + 已知 issue 数字"之上。自写实现出了问题连参照系都没有。
- **审稿防御**："我们对被评测的 VLA 不做任何修改"本身是防攻击话术的一部分（外挂式、不动 VLA 一个参数）。

### 2.2 为什么 ForkableEnv 必须自研（且是第一优先级）

state-fork 是整个 NGC-Plus 的地基：\(\hat{r}(s,a_i)\) 的每一条 rollout 都依赖快照-恢复的逐像素一致性。官方 MuJoCo `set_state` 不覆盖 controller 积分项、rng、policy 侧缓存三层状态（指南一 §1.3），没有任何现成库解决 robosuite 封装层的这个问题。这段代码：
- 必须自研，作为 LIBERO-Plus env 的外层 wrapper；
- 必须先于一切采集工作通过验收（同 snapshot 恢复两次 × 50 步确定性动作 → 观测逐像素一致、位姿差 \(< 10^{-9}\)）；
- 是发布物 `NGC-Plus/` 的一部分（使用者需要它来跑 protocol）。

### 2.3 为什么 fallback 执行器必须自研且独立成包

E8 的方法论根基是"**所有 baseline 与我们共享同一 fallback 执行器，只替换决策器**"。这要求执行器：
- 与任何决策器解耦（selector、VLM deliberation、rule 触发都调用同一接口）；
- 每个 fallback 是独立可单测的类（W10 验收 20/20）；
- masking 逻辑在训练与部署间字节级一致。

现成 RL 框架的"动作空间"抽象无法表达 ROLLBACK 的逆动作重放 + gripper 特殊处理、RESAMPLE 的"本步不执行"、REPLAN-goal 的 milestone 库切换这类富语义动作，硬套只会把逻辑打散到 callback 里。

### 2.4 为什么评测协议必须自研

`protocol/eval_feb.py` 是 benchmark 发布物：外部使用者拿任意方法的决策日志输入，得到 FEB / broken-success / net-success / clean-regret。它定义了论文提出的度量本身，必须：零重依赖（numpy + pandas 级）、有 schema 文档、有 golden test（固定输入 → 固定输出）。这类代码不存在"base code"可用。

### 2.5 RL 框架选型：为什么自写而不用 SB3 / tianshou / cleanrl 直接套

需求清单 vs 框架能力：

| 需求 | SB3 | tianshou | 自写（cleanrl 风格） |
|------|-----|----------|---------------------|
| 动作 masking（\(-\infty\) on Q） | 需魔改 | 支持但绕 | 原生 3 行 |
| 双塔 + cross-attn + Dueling 自定义网络 | 痛苦 | 可以 | 原生 |
| PER + offline 数据固定 20% 混采 | 需自写 buffer | 部分 | 自写 buffer 本来就要写 |
| BC+CQL warm-start → DQN 无缝衔接 | 不支持 | 不支持 | 同一套网络/优化器自然衔接 |
| 预算记账钩子（每 rollout 计数） | callback 地狱 | callback | 直接写进采样器 |
| 三租户 GPU 批处理调度采样 | 完全不匹配其 VecEnv 假设 | 同 | 自研采样器必须存在 |
| checkpoint 含 replay + 记账 + RunningNorm | 需魔改 | 需魔改 | 原生 |

结论：环境交互侧（三租户批处理、zmq env worker）无论如何都要自研，而 DQN 算法本体（Double+Dueling+PER+n-step）不过千行且成熟配方齐全。引入框架只共享了最不值钱的 20%，却让最关键的 80%（masking 一致性、记账、混采）去迁就框架抽象。**做法：以 cleanrl 的 DQN 实现为参考起点复制思想（非依赖），写成本项目的 `selector/` 模块。**

### 2.6 lite baselines 的复用边界

- VoLo-lite / CycleVLA-lite 的 VLM deliberation：**复用**一个开源 7B 级 VLM 的推理（如 transformers 直载），prompt 与解析逻辑自写；
- B2FF-lite 的 milestone 库：优先读其开源码理解精神后**自研简化版**（其代码风格/依赖未必兼容本 repo；且论文必须逐一声明 lite 与原版差异，自研版差异边界更清楚）；
- FOREWARN-lite：直接复用自家特征管线的 V-JEPA 2 rollout 打分，几乎零新增代码；
- HELM-lite：纯 rule，全自研（一两百行）。

---

## 3. 仓库组织：monorepo + 三 conda 环境

### 3.1 顶层结构

```
rase-lite/                          # 单一 git 仓库（monorepo）
├── env.lock.md                     # 三环境全部版本号 + upstream commit hash + 变更记录
├── envs/
│   ├── smolvla.yaml  oft.yaml  rl.yaml     # conda 环境定义（指南二 §5：三环境隔离）
├── configs/
│   ├── eval_base.yaml              # 唯一评测配置源（一处配置，处处引用）
│   ├── collect.yaml  selector.yaml  reward.yaml
├── third_party/                    # 只放 pin 说明与补丁，不 vendor 源码
│   ├── PINS.md                     # LIBERO-plus@<commit>、openvla-oft@<commit> 等
│   └── patches/                    # 万不得已的 monkey-patch，每个带单测
├── rase/                           # 自研核心包（pip install -e .）
│   ├── envs/
│   │   ├── forkable_env.py         # ★ 三层快照/恢复 wrapper
│   │   └── workers.py              # zmq env worker + 批处理调度
│   ├── collect/                    # 指南一 Step 1–6
│   │   ├── perturb_sampler.py      # 配额采样（camera/robot L3–L5 倾斜）
│   │   ├── candidates.py           # K=8 候选生成（SmolVLA/OFT 双 oracle）
│   │   ├── adaptive.py             # 两阶段自适应采样 + Wilson 判据
│   │   ├── scheduler.py            # 断点续采调度器（results/ 扫描跳过）
│   │   ├── triage.py               # Set A/B/C/uncertain 三分
│   │   ├── reversibility.py        # ρ(s) 自动标注规则
│   │   └── causal_analysis.py      # 扰动→NGC 产率表与图
│   ├── features/                   # 指南三 §1
│   │   ├── dino.py  vjepa.py  acc.py  probe.py  safe_head.py
│   │   ├── reference_bank.py       # FAISS 成功轨迹参考库
│   │   └── pipeline.py             # 七路信号拼接 + RunningNorm
│   ├── fallbacks/                  # ★ 与决策器解耦的执行器包
│   │   ├── base.py  rollback.py  resample.py  replan_goal.py
│   │   ├── replan_text.py  wait.py  abstain.py
│   │   └── masking.py              # 训练/部署共用的唯一 masking 实现
│   ├── selector/                   # 指南三 §3–4
│   │   ├── network.py              # 双塔 + cross-attn + Dueling（3.5M）
│   │   ├── replay.py               # PER + offline 固定混采
│   │   ├── warmstart.py            # BC + CQL
│   │   ├── dqn.py                  # Double+Dueling+n-step 训练循环
│   │   └── budget.py               # B_selector 记账（§6.7）
│   ├── oracle/
│   │   ├── server.py               # OFT zmq 批量推理服务
│   │   └── client.py
│   ├── baselines/                  # E8：只有决策器，全部调 rase.fallbacks
│   │   ├── volo_lite.py  helm_lite.py  b2ff_lite.py
│   │   ├── cyclevla_lite.py  forewarn_lite.py
│   ├── eval/
│   │   ├── run_eval.py             # 读 configs/eval_base.yaml 的唯一入口
│   │   └── diagnostics.py          # E6 分离度、fallback 热力图等论文图
│   └── utils/  (seeding, logging, io)
├── protocol/                       # ★ 随 NGC-Plus 发布的独立零依赖子包
│   ├── eval_feb.py  report_template.md  schema.md
├── scripts/                        # 一次性入口（smoke_test、建参考库、打包发布物）
├── tests/
│   ├── test_fork_roundtrip.py      # §1.3 验收（逐像素一致）
│   ├── test_fallbacks/             # 20 状态单元验收
│   ├── test_masking_parity.py      # 训练/部署 masking 一致性
│   └── test_feb_golden.py          # protocol golden test
└── runs/  pool/  results/          # 产出（gitignore，仅结构约定）
```

### 3.2 三环境与包边界的关系

- `rase/` 包设计为**在三个 conda 环境中都可 import**：重依赖（lerobot、openvla-oft）全部延迟导入（函数内 import），`rase.collect` 在 smolvla 环境跑、`rase.oracle` 在 oft 环境跑、`rase.selector` 在 rl 环境跑。
- 跨环境通信只走 zmq + 磁盘（npz/json），**绝不**试图在一个环境里同时装 lerobot 与 openvla-oft（依赖树冲突风险，指南二 §5.2）。
- `protocol/` 是刻意的第二个包：零重依赖、可单独 pip 安装、随 benchmark 发布——它的用户是外部研究者，不应被迫装你的全栈。

### 3.3 配置与溯源纪律（写进 CI）

1. 所有评测入口只接受 `--config configs/eval_base.yaml [+overrides file]`，禁止散落命令行覆盖；每次运行把 resolved config 快照 + git SHA + `env.lock.md` 哈希写入 `runs/<name>/`。
2. `env.lock.md` 变更必须单独 commit，message 带理由——它就是论文附录复现配置的单一事实源。
3. CI（哪怕只是本地 pre-push 钩子）跑：fork 往返测试、masking parity 测试、FEB golden test——这三个坏了会静默污染数据，比任何功能 bug 都贵。

---

## 4. 分阶段搭建顺序（对齐 18 周计划）

| 周 | 搭什么 | 复用/自研 | 验收 |
|----|--------|----------|------|
| W1 | 三 conda 环境 + pin；`scripts/smoke_test.py`；**ForkableEnv** | 底座全复用；ForkableEnv 自研 | fork 往返逐像素一致，不达标不进 W2 |
| W1–W2 | SmolVLA/OFT 自测 baseline；oracle server | 复用官方评测命令；server 自研 | OFT clean 对齐官方 ±3pp；batch-8 前向 ≤300ms |
| W2 | `collect/` 候选生成 + 多样性 sanity check | 自研 | 候选末端位移分布不坍缩 |
| W3 | `adaptive.py` + `scheduler.py` + 断点续采 | 自研（statsmodels 复用） | 20 条 rollout 实测墙钟 → 成本重估 |
| W4–W9 | 全量采集（跑，不是写）+ `triage/reversibility/causal_analysis` | 自研 | 周 QC 清单 |
| W9–W10 | `protocol/` 打包 + golden test | 自研 | 外部可安装、README 可跑通 |
| W10 | `features/` 七路信号 + probe/SAFE 头；`fallbacks/` 5 类 | V-JEPA/DINO 复用，其余自研 | 端到端延迟 <150ms；fallback 单测 20/20 |
| W10–W11 | `selector/warmstart.py`（BC+CQL） | 自研 | held-out top-1 ≥60%、Set C fallback 召回 ≥80% |
| W11–W13 | `selector/dqn.py` + 三租户采样 infra + 记账 | 自研 | 1K episodes 无发散；checkpoint 三件套同步 |
| W14–W16 | `baselines/` 五个 lite + E3 LoRA（peft 外挂） | 决策器自研、VLM/peft 复用 | E8/E3 记账表 |

**自研代码量的现实预算**（帮你判断可行性）：ForkableEnv + workers ≈ 800 行；collect 全套 ≈ 2,000 行；features ≈ 1,200 行；fallbacks ≈ 1,000 行 + 测试；selector ≈ 2,000 行；baselines ≈ 1,000 行；protocol + eval ≈ 800 行。合计约 **9,000–10,000 行核心 Python**，单人 18 周内与实验并行是紧但可行的量——前提正是把一切能复用的都复用掉，胶水保持薄。

---

## 5. 反模式清单（每一条都对应一个真实事故模式）

1. **Fork lerobot 加"小功能"** → 三个月后 upstream 修了 bug 你合不进来，自测数字与社区永久分叉。
2. **在采集脚本里内联 Wilson/三分逻辑** → 判据要进论文与发布物，散落即不可审计；一律收进 `rase/collect/`。
3. **selector 训练用框架、部署自写前向** → masking/归一化两套实现，训练-评测 gap 排查一周起步；网络与 masking 只允许一份实现两处 import。
4. **fork 状态进特征** → 部署一致性红线（设计报告 §4.1），代码层面用类型隔离：特征管线的输入 dataclass 里根本不存在 fork 字段。
5. **vendor 上游源码进 repo** → 版本漂移不可见；只 pin，不 vendor。
6. **protocol 依赖主包** → 外部用户装不上，benchmark 采用率归零；`protocol/` 保持零重依赖。

---

## 6. 一页决策卡（贴在显示器上）

- 底座模型/环境：**官方代码，pin 死，一行不改。**
- 快照恢复、采样判据、fallback、selector、FEB 协议：**全自研，这就是论文。**
- RL 框架：**不引入，cleanrl 风格自写 DQN。**
- 组织：**monorepo + 三 conda 环境 + zmq/磁盘通信 + protocol 独立零依赖子包。**
- 纪律：**一处配置、每数字可溯源、三个 CI 级测试（fork/masking/FEB golden）永远绿。**
