# RASE-UI 三算子 same-state 实验结果与下一阶段计划

日期：2026-08-01  
服务器项目：`/root/autodl-tmp/RASE`  
实验终端：`tmux 0`

## 1. 结论

本轮实验已经把严格 `CONTINUE`、`REPLAN` 和 `SWITCH_POLICY(OFT)` 放到完全相同的 8 个冻结状态上比较，实验与汇总均完整结束。

最重要的结论是：**当前状态池不存在成功率层面的算子互补性**。

| 算子 | 成功数 | 成功率 |
|---|---:|---:|
| `CONTINUE(smol active suffix)` | 6/8 | 75.0% |
| `REPLAN(smol)` | 5/8 | 62.5% |
| `SWITCH_POLICY(OFT)` | 6/8 | 75.0% |
| same-state oracle | 6/8 | 75.0% |

因此：

- best fixed success = 75.0%；
- same-state oracle success = 75.0%；
- oracle − best fixed = **0.0 pp**；
- 没有任何 success-only unique winner；
- OFT 与 CONTINUE 的 8 状态成功/失败向量完全一致；
- 当前数据不能支撑 selector、CIVR 或 world model 的训练主张。

不过，出现了一个值得下一轮验证的次级信号：在 6 个“至少有一个算子能成功”的状态上，以“先保证成功、再最小化环境步数”为目标，逐状态 oracle 的平均完成步数为 74.50，而最佳固定 CONTINUE 为 84.17，节省 9.67 步，约 11.5%。其中 5 个早期状态的最低步数算子是 OFT，1 个中期状态是 CONTINUE。

这说明当前 pool 可能有**效率路由机会**，但没有**成功率恢复机会**。顶会主结果必须仍以 terminal success 为主，不能用成本信号掩盖 0 pp 的 success oracle gap。

## 2. 逐状态结果

算子顺序为 `CONTINUE / REPLAN / SWITCH_OFT`。

| suite / task | snapshot step | 成功模式 | 环境步数 C / R / O | 解释 |
|---|---:|---|---|---|
| object / task 10 | 0 | 1 / 1 / 1 | 128 / 123 / 107 | 三者都能完成，OFT 更快 |
| spatial / task 4 | 0 | 1 / 1 / 1 | 80 / 81 / 78 | 三者都能完成，差异很小 |
| object / task 10 | 2 | 1 / 1 / 1 | 107 / 103 / 97 | 三者都能完成，OFT 更快 |
| spatial / task 4 | 2 | 1 / 1 / 1 | 58 / 57 / 52 | 三者都能完成，OFT 更快 |
| object / task 10 | 4 | 1 / 1 / 1 | 90 / 84 / 71 | 三者都能完成，OFT 更快 |
| spatial / task 4 | 4 | 1 / 0 / 1 | 42 / 225 / 180 | 保留 active suffix 很重要；REPLAN 失败，OFT 虽成功但很慢 |
| object / task 10 | 6 | 0 / 0 / 0 | 205 / 205 / 205 | 当前三种算子均无法恢复 |
| spatial / task 4 | 6 | 0 / 0 / 0 | 205 / 205 / 205 | 当前三种算子均无法恢复 |

成功模式计数：

- `111`：5 个状态；
- `101`：1 个状态；
- `000`：2 个状态。

这不是“不同失败需要不同恢复”的理想 benchmark 结构，而更像一个共同的时间/可逆性边界：早期都成功，中期 CONTINUE 有一次优势，晚期都失败。若直接训练模型，模型很可能只学到 step 或 task shortcut。

## 3. 成本结果应如何理解

全 8 状态的平均环境步数：

| 算子 | 平均环境步数 | 平均记录执行时间 |
|---|---:|---:|
| CONTINUE | 114.375 | 6.157 s |
| REPLAN | 135.375 | 7.223 s |
| SWITCH_OFT | 124.375 | 4.529 s |

OFT 的记录执行时间更短，但当前数字是 warm-server continuation 时间，不包含模型冷启动与服务加载，因此不能直接写成“OFT 计算更便宜”。环境步数更可比，也显示：

- 在普通早期状态，OFT 常更快；
- 在 spatial step 4，CONTINUE 只需 42 步，而 OFT 需 180 步；
- 所以即使成功标签相同，仍可能存在 cost-aware routing 的价值。

下一轮必须把冷启动、warm inference、GPU seconds、模型调用次数、环境步数和 wall-clock latency 分开记录，并预注册 utility 权重。目前 `utility_cost=0`，不能计算可信的 harmful intervention rate 或 matched-cost deployment。

## 4. 正式 Opportunity Audit

使用当前三算子矩阵运行 gate：

```text
status: not_ready
complete snapshots: 8 < 20
oracle gap: 0.0 < 0.05
no costly failed intervention was observed
```

第三条不是说真实系统“没有伤害”，而是当前 Phase 0 将 scalar `utility_cost` 冻结为 0，尚未建立物理成本标定。

按照研究计划中的 Go/No-Go 规则，本轮只能判定：

> 工程语义与执行链路已通过；benchmark opportunity 尚未通过；现在不能训练方法。

## 5. 本轮完善的代码

### 三算子执行与汇总

- `configs/rase_ui_phase0_oft_smoke.yaml`：冻结 OFT same-state smoke 配置；
- `scripts/assemble_intervention_matrix.py`：把 Smol strict CONTINUE/REPLAN 与 direct OFT 合并为统一 operator registry、snapshot 和 outcome matrix；
- 严格要求 OFT 状态覆盖与 Smol snapshot ID 完全一致，缺失、额外或跨 suite 重复状态都会失败；
- 输出 success oracle、best fixed、unique winner、无算子支持率、成功模式、相对 CONTINUE 的配对方向、平均成本和 success-then-steps oracle；
- `tests/test_assemble_intervention_matrix.py`：覆盖 success complementarity、缺失 arm 和成功向量相同但成本不同的情况。

### 配置与文档

- `configs/interventions_phase0.json`：strict CONTINUE 已在 suffix provenance 与 parity 通过后启用；
- `docs/runbooks/rase_ui_phase0.md`：记录本轮 canonical outputs、三算子模式与 not-ready 结论。

相关测试与 lint 均通过。

## 6. 下一阶段：先建立 Opportunity，再训练方法

### Phase 0B-1：统一 runner 与真实成本（代码优先）

应先完成一个统一 `rollout_intervention_matrix.py`，由冻结 registry 和 snapshot manifest 调度所有 arm，替代当前“Smol runner + OFT runner + assembler”的过渡结构。必须具备：

1. 同一 snapshot、同一 continuation seed、同一 horizon；
2. operator feasibility、proposal、execution、termination 的统一 contract；
3. 可恢复的幂等 resume、每 arm 独立 failure record、manifest hash；
4. warm/cold 模型成本、GPU seconds、policy calls、env steps、latency 分项记录；
5. 冻结 scalar utility 配置，不再使用 `utility_cost=0` 作为主实验标签。

### Phase 0B-2：改变状态来源，而不是重复放大当前 8 个状态

当前 pool 的最大问题不是样本量本身，而是只有两个 task、一个 source policy family、四个时间点，导致结果主要由时间决定。下一轮 screen 应覆盖：

- 至少 10 个 task，并跨 spatial、object、goal、long-horizon task；
- Smol 和 OFT 都作为 source policy 生成轨迹，避免只研究“从 Smol 切到 OFT”；
- clean success、natural failure、controlled perturbation、policy disagreement、contact 前后、progress stall、near-irreversible boundary；
- 固定 cadence 与事件触发 snapshot 混合，评估集不能只选择看起来有恢复价值的状态；
- 新增一个可执行的 `LOCAL_CORRECT(retreat_realign_v1)`，先不急着加入物理 REWIND。

推荐两阶段节省算力：

1. **Screen**：10 tasks × 20 episodes × 6 snapshots × 4 executable operators × 1 seed = 4,800 branches；
2. **Confirm**：从预注册规则选出的约 200 个边界状态 × 4 operators × 3 seeds = 2,400 branches。

Screen 只能用于发现状态区域；正式统计使用 confirm 集，并以 task / source episode / perturbation cluster 为独立统计单元。

### Phase 0B-3：提高进入方法阶段的门槛

建议下一轮必须同时满足：

1. 至少 200 个 complete same-state snapshots；
2. same-state success oracle 比 best fixed 高至少 5 pp；
3. 至少 3 个 operator 在至少 2 个 task 中有稳定 unique-win 区域；
4. 存在可复现的 beneficial、harmful 与 futile interventions；
5. operator ranking 不是只由 task ID、step 或 failure type 决定；
6. 3-seed confirm 后方向不消失。

未通过时不要训练 selector；应先修改 operator profile 或 source/perturbation sampling，并用新版本 preregistration，不能在同一测试集上反复挑状态。

### Phase 1：Gate 通过后再训练轻量方法

顺序应是：

1. task/episode/perturbation-cluster held-out split；
2. best fixed、frequency-matched random、risk-triggered fixed recovery；
3. history-only operator-value model；
4. operator embedding + executor fingerprint；
5. 只有 history-only 已超过 matched random、且能捕获至少 25%–35% oracle gap 后，才加入 operator-conditioned world model；
6. world model 必须证明超过 history-only，并在 unseen perturbation 上仍有增益，否则降为 optional baseline。

### Phase 2：顶会完整版

目标配置仍应是：两个 simulator、三个冻结 policy/executor、五个 core operators、matched-total-compute、完整 same-state oracle、至少一个真实机器人小规模验证。LIBERO 可以作为 compatibility platform，但不应是唯一证据。

## 7. 可复现实验命令

当前三算子 OFT continuation：

```bash
cd /root/autodl-tmp/RASE
PY=/root/autodl-tmp/envs/smolvla/bin/python

OUTPUT_PREFIX=rase_ui_phase0_switch_oft \
STATE_KEYS_JSON=runs/rase_ui_phase0_smoke_parity_keys.json \
CANDIDATES_DIR=runs/rase_ui_phase0_smoke_parity_keys.json \
OFT_RUNNER=prefix-ablation \
OFT_PREFIX_ARMS=direct \
OFT_SUITE_SHORTS=spatial,object \
FRESH_RUN=1 \
PREFLIGHT=1 \
./scripts/run_oft_verify_suites.sh \
  configs/rase_ui_phase0_oft_smoke.yaml parity8_v1
```

三算子矩阵汇总：

```bash
$PY scripts/assemble_intervention_matrix.py \
  --smol-run runs/rase_ui_phase0_smoke_parity_paired \
  --oft-summary runs/rase_ui_phase0_switch_oft_spatial_parity8_v1/summary.json \
  --oft-summary runs/rase_ui_phase0_switch_oft_object_parity8_v1/summary.json \
  --output-dir runs/rase_ui_phase0_matrix_parity8_v3 \
  --fresh-run
```

正式 gate：

```bash
$PY scripts/audit_intervention_opportunity.py \
  --registry runs/rase_ui_phase0_matrix_parity8_v3/operators.json \
  --snapshots runs/rase_ui_phase0_matrix_parity8_v3/snapshots.jsonl \
  --outcomes runs/rase_ui_phase0_matrix_parity8_v3/outcomes.jsonl \
  --output runs/rase_ui_phase0_matrix_parity8_v3/opportunity_audit.json \
  --min-complete-snapshots 20 \
  --min-oracle-gap 0.05 \
  --min-winning-operators 3 \
  --min-tasks-per-winning-operator 2 \
  --allow-zero-harm
```

注意：若输出目录已存在，不要重复使用 `--fresh-run`；使用新版本目录或显式 resume，避免覆盖已有结果。

## 8. Canonical artifacts

```text
runs/rase_ui_phase0_switch_oft_spatial_parity8_v1/summary.json
runs/rase_ui_phase0_switch_oft_object_parity8_v1/summary.json
runs/rase_ui_phase0_matrix_parity8_v3/summary.json
runs/rase_ui_phase0_matrix_parity8_v3/opportunity_audit.json
runs/rase_ui_phase0_matrix_parity8_v3_audit_tmux0.log
```

最终判断：**Phase 0 工程验证通过；Phase 0 benchmark opportunity 不通过；下一步应扩展 source/task/perturbation/operator 形成真实多算子互补，而不是立即训练世界模型。**
