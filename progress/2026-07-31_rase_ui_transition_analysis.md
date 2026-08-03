# RASE → RASE-UI 转向分析、代码交付与后续执行方案

**日期：** 2026-07-31  
**检查对象：** `/root/autodl-tmp/RASE` 服务器工作树、W1–W10 进展记录、两份 RASE-UI 方案文档  
**本轮边界：** 已读代码、改代码、运行 CPU 单测与静态检查；未启动、重跑或续跑任何 VLA/GPU 实验

## 1. 结论

建议转向 RASE-UI，但不是把旧的三分类 selector 扩成六分类，也不是立即加入世界模型。

正确的新主线是：

> 从完全相同的 simulator snapshot 出发，对结构化、真实可执行的 operator 做重复 continuation，先证明 operator 之间存在状态依赖互补和足够的 oracle headroom，再学习 `V(history, operator)`；世界模型只在 history-only value model 已过门后作为增量预测证据加入。

这次转向有充分证据支持：旧的 ridge selector 已按预注册规则失败，继续堆 MLP/RL 不合理；但 W7/W8/W9C 又证明同态下不同 continuation policy 的结果确实不同，说明 same-state intervention allocation 是一个比旧 policy routing 更自然、更广的新问题。

不过，新项目应被视为一个新的预注册方法分支，不能写成“W9C selector 的继续优化”。旧论文主张仍冻结为 benchmark/diagnosis；新方法必须从新的 operator opportunity gate 重新开始。

## 2. 服务器上的真实现状

### 2.1 工程资产

当前仓库已经具备很强的 benchmark 基础：

- `ForkableEnv` 支持 simulator state、controller、observable cache 和 RNG 的 snapshot/restore；
- state pool 有 pickle-free、checksum、manifest、idempotent resume；
- 已有 restore parity、wrong-task rejection、episode/task split 防泄漏测试；
- direct Smol、direct OFT、candidate prefix、zero prefix 等 arm 已有 resumable runner；
- 已有 deployable feature、action-matched random、best fixed、oracle、clean regret 和 split support audit；
- W9C 修复了 official clean task identity，不能再使用 W9A/W9B 错误 control；
- 现有代码坚持 any-of-K portfolio 不是 deployable action，这一点应保留。

这些资产意味着不需要重写仓库，也不应新开一个与 RASE 完全割裂的 repo。最合理的是在现有仓库增加 `rase/interventions/`，让旧 state pool 和 runners 成为 RASE-UI 的底层执行基础。

### 2.2 科学进展

关键结果如下：

| 阶段 | 结果 | 新方向中的含义 |
|---|---:|---|
| clean SmolVLA | 约 70% | source policy 在干净任务上有能力 |
| LIBERO-Plus collapse | 约 0.38% | 形成强 failure frontier，但过度 failure-biased |
| W4 | Smol 0/1536 candidate outcomes；OFT portfolio 17/32 states | recoverability 主要是 policy-relative |
| W5 | temperature 0.3/0.7/1.0 合计 0/576 | “多采样即可恢复”不成立 |
| W7 | direct Smol 0/24；prefix+OFT 8/24 | alternate policy 在部分同态可恢复 |
| W8 | direct OFT 9/24 | SWITCH_POLICY 是真实可部署 operator |
| prefix 机制消融 | candidate-specific rescue 0 | candidate prefix 不应成为核心新 operator |
| W9C | clean32 和 readiness 通过；ridge 与 matched random 差值 0 | 数据可训练不等于方法有效；旧 selector 分支应终止 |
| W10 | Object/Spatial Smol 0/16、OFT 1/16 | recoverability 强烈 suite/regime dependent |

W9C 冻结 56-state 三臂数据的临时迁移审计显示：

- `replan_smol / switch_oft / abstain` 的 same-state oracle utility 为 `0.4729`；
- best fixed（`switch_oft`）utility 为 `0.3821`；
- oracle gap 为 `0.0907`；
- 三个最优动作的 state counts 为 16 / 12 / 28。

这证明旧数据中存在分配 headroom，但不证明方法有效：ridge 已经失败。更重要的是，这还不是新的 RASE-UI gate，因为旧数据没有 strict CONTINUE、LOCAL_CORRECT 或 REWIND。

## 3. 最重要的代码语义发现

旧数据中的 `continue_smol` 实际执行流程是：restore snapshot → `policy.reset()` → 从当前观察重新调用 SmolVLA。

因此它应被解释为：

```text
REPLAN(smolvla, current_observation)
```

而不是严格的：

```text
CONTINUE(smolvla, remaining_active_action_suffix)
```

旧 snapshot 又位于 action-chunk 边界，没有保存 active action suffix。若不修正这个定义，论文最核心的 `CONTINUE vs REPLAN` 标签会错。

已安装的 SmolVLA 确实使用 `policy._queues[ACTION]` 保存 action chunk 剩余动作，所以新 collector 可以在 mid-chunk snapshot 时保存可部署 suffix；但必须先实现并测试：

1. mid-chunk snapshot；
2. postprocessor 后的 env-space suffix；
3. suffix replay parity；
4. suffix 执行完后再恢复正常 source policy；
5. public action/RGB/proprio history 与 privileged restore state 的隔离。

在这五项完成前，任何旧 direct-Smol 数据都不得改名为 CONTINUE。

## 4. 本轮已完成的代码

服务器新增：

```text
rase/interventions/__init__.py
rase/interventions/base.py
rase/interventions/schema.py
rase/interventions/dataset.py
scripts/migrate_legacy_interventions.py
scripts/audit_intervention_opportunity.py
tests/test_intervention_schema.py
tests/test_intervention_dataset.py
configs/interventions_phase0.json
protocol/intervention_dataset_schema.md
docs/runbooks/rase_ui_phase0.md
```

实现内容：

- 六个核心 family：CONTINUE、REPLAN、LOCAL_CORRECT、REWIND、SWITCH_POLICY、ABSTAIN；
- 结构化 `OperatorSpec`，明确 executor、recovery target、parameters、requirements；
- 独立的 snapshot / outcome schema；
- feasibility reason、operator completion、seed、proxy、outcome semantics；
- physical cost vector 与 legacy scalar utility cost 分离；
- 旧 W9C/W10 三臂 JSONL 的保守迁移，默认 `continue_smol -> replan_smol`；
- same-state opportunity audit：complete coverage、repeat deficit、best fixed、oracle gap、winner/task coverage、harm、futility；
- strict CONTINUE 为默认硬门；旧三臂数据即使有 9.07pp oracle gap，也因缺 strict CONTINUE 返回 `not_ready`。

验证结果：

- 新增测试：`7 passed`；
- 兼容性测试（旧 selector、benchmark analysis、dataset export、forked rollout + 新测试）：`37 passed`；
- Ruff：`All checks passed`；
- 实际 W9C 迁移 smoke：56 snapshots、168 outcomes，正确返回 `not_ready`，唯一硬原因是 strict CONTINUE 缺失。

## 5. 旧模块到新问题的映射

| 旧资产 | 新语义 | 处置 |
|---|---|---|
| ForkableEnv restore | exact same-state branching | 直接保留 |
| direct OFT | SWITCH_POLICY(openvla_oft) | 直接保留，增加 handoff/latency 记录 |
| direct Smol + reset | REPLAN(smolvla) | 重命名语义，不重写历史原始文件 |
| candidate prefix | diagnostic/resample proxy | 不作为主 deployable operator |
| abstain | ABSTAIN | 保留，但单独报告 human/help/reset 语义 |
| W9C ridge | history-only 负基线 | 作为旧 policy-routing 负结果，不升级容量 |
| selector features | history-only baseline 输入 | 可复用，但需扩充短时 history |
| scalar costs 0.02/0.10/0 | frozen utility sensitivity | 不能宣称为物理 latency/compute |

## 6. 推荐的实施顺序

### Phase A：Decision-context v2 与四臂 pilot

先实现四个定义最干净的 operator：

```text
CONTINUE(smol active suffix)
REPLAN(smol from current observation)
SWITCH_POLICY(OFT from public handoff)
ABSTAIN
```

数据建议：

- 10 个任务，四个 suite 都保留；
- 至少 40 个 complete snapshots；
- clean / near-failure / natural-or-injected failure 平衡；
- 每个真实 stochastic arm 3 seeds，common-random-number 配对；
- snapshot 不能只取失败末端，要覆盖任务早中晚期；
- 先跑 Goal/Long 正 headroom regime，同时保留 Object/Spatial negative regime，不能只采容易成功的状态。

Go 条件：

- exact same-state restore/suffix replay parity 通过；
- oracle − best fixed ≥ 5pp；
- 至少三个 operator 在至少两个任务上成为最优；
- harm 与 costly futility 非零；
- task-ID-only shortcut 不能解释主要结果。

若此门失败，不训练 CIVR，也不做世界模型。

### Phase B：增加 LOCAL_CORRECT

首个 local profile 应简单、跨任务、物理可执行，例如：

```text
retreat → open/hold gripper rule → short Cartesian realign → source replan
```

必须明确：可行性、最大 20 steps、接触/碰撞停止条件、是否完成 correction、失败原因。不要一开始实现十几个 primitive，否则 profile 数量会爆炸且无法公平比较。

### Phase C：增加物理 REWIND

REWIND 不能用 simulator restore/teleport 充当执行动作。restore 只用于让不同 arm 从相同 state 起跑。

真实 REWIND 需要：

- collection 时保存 recent safe/contact-free public milestone；
- 实际 controller 沿可行路径返回；
- 记录 target selection、reachability、progress loss、collision、执行失败；
- 找不到安全 milestone 时 feasibility=false。

### Phase D：history-only value model

在 benchmark gate 通过后，先训练不含世界模型的 operator-conditioned value baseline：

```text
V(history, operator_spec)
→ success distribution
→ harm / feasibility / progress / physical cost
→ calibrated lower confidence bound
```

必须比较：best fixed、frequency/action-matched random、failure gate + fixed recovery、suite/task shortcut、same-state oracle。

方法门应保持严格：learned 同时超过 matched random 和 best fixed，paired 95% CI 下界为正，且 harm 不增加。否则停止方法分支。

### Phase E：世界模型

只有 history-only 方法过门后再做。世界模型只预测短时 operator-conditioned future evidence，不产生 outcome ground truth。建议先预测 object relation/contact/progress latent，而不是高分辨率长视频。

必须做：

- WM train、value train、calibration/test outcome 三者按 episode/task 隔离；
- imagined operator ranking 对真实 continuation ranking 的相关；
- full vs history-only；
- held-out perturbation；
- compute/latency-matched；
- OOD uncertainty 与错误 imagination safeguard。

若没有增量，WM 降为 baseline，不硬留在方法名中。

### Phase F：第二 simulator 与实机

首篇 pilot 不建议同时推进第二 simulator、世界模型和实机。顺序应为：LIBERO pilot → operator headroom → history-only method → WM → 第二平台/实机。

实机不声称 exact individual counterfactual，采用 randomized block design、initial-condition bins、重复 reset，验证闭环 gain、harm、latency 和 calibration。

## 7. 当前可直接运行的命令

### 7.1 代码验证（无实验）

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python

$PY -m pytest -q \
  tests/test_intervention_schema.py \
  tests/test_intervention_dataset.py

$PY -m pytest -q \
  tests/test_lightweight_selector.py \
  tests/test_benchmark_analysis.py \
  tests/test_dataset_export.py \
  tests/test_forked_rollout_contract.py \
  tests/test_intervention_schema.py \
  tests/test_intervention_dataset.py

$PY -m ruff check \
  rase/interventions \
  scripts/migrate_legacy_interventions.py \
  scripts/audit_intervention_opportunity.py \
  tests/test_intervention_schema.py \
  tests/test_intervention_dataset.py
```

### 7.2 迁移并审计 W9C 冻结数据（CPU，只分析已有结果）

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python

$PY scripts/migrate_legacy_interventions.py \
  --input runs/ngc_w9c_selector_dataset.jsonl \
  --output-dir runs/rase_ui_legacy_w9c

set +e
$PY scripts/audit_intervention_opportunity.py \
  --registry runs/rase_ui_legacy_w9c/operators.json \
  --snapshots runs/rase_ui_legacy_w9c/snapshots.jsonl \
  --outcomes runs/rase_ui_legacy_w9c/outcomes.jsonl \
  --output runs/rase_ui_legacy_w9c/opportunity_audit.json \
  --min-complete-snapshots 20 \
  --min-oracle-gap 0.05 \
  --min-winning-operators 3 \
  --min-tasks-per-winning-operator 2 \
  --allow-zero-harm
echo "audit_exit=$?"  # 预期为 2：旧数据缺 strict CONTINUE
set -e
```

不要为使它 PASS 而加 `--allow-missing-continue`；该参数只适合 CLI 调试。

### 7.3 复用已有 runner 生成 REPLAN 与 SWITCH 基线

对一个新的 frozen state-key manifest，direct Smol 命令只能标为 REPLAN：

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python

$PY scripts/rollout_direct_smol.py \
  --config <pilot-config.yaml> \
  --state-keys-json <pilot-state-keys.json> \
  --output-dir <output-replan-smol> \
  --fresh-run
```

OFT SWITCH_POLICY 可复用 suite-serial runner：

```bash
cd /root/autodl-tmp/RASE
SMOLVLA_ENV=smolvla \
OFT_ENV=oft \
OFT_SUITE_SHORTS=spatial,object,goal,10 \
OUTPUT_PREFIX=rase_ui_pilot_switch_oft \
STATE_KEYS_JSON=<pilot-state-keys.json> \
CANDIDATES_DIR=<pilot-state-keys.json> \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
FRESH_RUN=1 \
./scripts/run_oft_verify_suites.sh <pilot-config.yaml> ui_pilot
```

这些命令仍不能产生 strict CONTINUE；在 decision-context v2 collector 和 suffix runner 完成前，不应启动正式四臂 pilot。

## 8. 不建议做的事

- 不要恢复 W9C ridge → MLP → RL 的旧升级路线；
- 不要把 W10 的 1/16 当作 Object/Spatial 支持；
- 不要把 simulator restore 当作 REWIND；
- 不要把 candidate any-of-K 当 operator outcome；
- 不要只采 failure states；
- 不要把抽象 utility cost 写成真实 latency；
- 不要先训练世界模型再寻找 operator headroom；
- 不要一次实现十类 operator；
- 不要改 seed、top-up 或重分 split 去修复预注册失败结果。

## 9. 最终建议

短期论文策略应分成两条互不污染的轨道：

1. 现有 RASE：保持 benchmark + diagnosis，完整报告 ridge kill、suite heterogeneity 和负结果；
2. 新 RASE-UI：从 decision-context v2、strict CONTINUE、四臂 same-state pilot 和 opportunity gate 开始新的预注册。

若四臂 pilot 能稳定重现约 5–10pp 的 oracle-over-best-fixed gap，并出现 CONTINUE、REPLAN、SWITCH 至少三类互补赢家，这个新方向值得继续投入 LOCAL_CORRECT、REWIND 和 CIVR。若互补只来自 REPLAN/SWITCH/ABSTAIN，项目仍可作为更严格的 intervention benchmark，但不应提前承诺世界模型和实机旗舰版本。
