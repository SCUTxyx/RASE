# RASE vNext 下一阶段实验结果与推进记录（2026-08-17）

## 结论先行

本阶段与两个 revised 文档的顺序基本一致，但必须保留一个边界：历史 Phase-C 数据仍然是 B-FAIL；本阶段只证明了“修复后的同步候选捕获路径”通过了新的 B2 parity 审计，并在 A-PARTIAL 的单策略 π0-fast pilot 中完成了 D0 feasibility。不能把这两个结果写成 pooled multi-VLA、D-PASS 或闭环增益。

当前状态：

| Gate | 结果 | 含义 |
|---|---|---|
| A | A-PARTIAL | 仅允许显式标注的单策略 π0-fast pilot |
| B（历史 Phase C cohort） | B-FAIL | 1/288 requery group 不可复现，旧数据不追溯修复 |
| B2（未来同步采集路径） | `B2_CAPTURE_PASS` | 新路径的结构、哈希、执行关联和 MotionTrace 通过 |
| C | `PILOT_SIGNAL_WEAK` | 旧 pilot 有弱 action signal，但 handcrafted semantic interaction 为负 |
| D0 | `D0_FEASIBILITY_PASS` | 20 个同根真实仿真 rollout 中，至少两个 root 出现成功/失败混合 |

## B2 同步捕获 smoke

冻结 manifest：`runs/rase_vnext/frozen/b2_capture_smoke_manifest_v1.json`；sha256 为 `c90b5b72c63425fc73a8b9c3d1e0fd331d04ee8c9bdd111f1a24447031eed82a`。

四个套件各一个 metadata-only 选择的 replica-0 group，共 20 个 branch jobs。重新实现了 immutable candidate capture：同一 boundary 同步保存 continue、requery、fallback 的完整数组、mask、instruction、proprio、seed ledger、first-action hash 和实际 fallback trace。

审计结果：

- 4/4 capture groups 通过；12 个动作块通过完整性检查；
- 三种可执行 operator 均被捕获；branch↔capture join 通过；
- raw→canonical→raw 往返最大绝对/相对误差均为 0；
- MotionTrace 12/12 通过，valid-step fraction 为 1；
- 结果是 `B2_CAPTURE_PASS`，但 scientific scope 明确为 `PARITY_ONLY_NOT_AN_EFFECT_RESULT`。

这解决的是“候选动作在同一边界被同步、可审计地冻结”的工程/可复现性问题，不是方法效果。

## D0 simulator-verified semantic feasibility

由于 A 仍是 PARTIAL，D0 被严格限定为单策略 feasibility smoke，不是 D-GATE。协议在任何 outcome 产生前冻结，使用 B2 的四个 capture root，固定 5 种动作变换：

1. identity
2. translation sign flip
3. rotation sign flip
4. temporal reverse
5. gripper phase shift

原始候选动作使用同一 boundary 实际捕获并真实执行的 `fallback.persistent` 10×7 chunk；`continue.source` 仅用于边界首动作 hash 的独立复现检查。这样避免了原先 continue/requery 仅有 1 步导致 temporal reverse 退化为 identity。

20/20 rollout 完成，具体结果：

| Suite | identity | translation | rotation | temporal | gripper | root diversity |
|---|---:|---:|---:|---:|---:|---:|
| Object | F | F | F | F | F | 否 |
| Long | F | F | F | F | F | 否 |
| Goal | T | T | F | F | F | 是 |
| Spatial | T | T | T | T | F | 是 |

总计 6/20 成功、14/20 失败；按扰动成功数为 identity=2、translation=2、rotation=1、temporal=1、gripper=0。D0 summary 为 `D0_FEASIBILITY_PASS`，4 个 root 中 2 个有 within-root outcome diversity。

这只说明预注册动作变化确实能在至少两个同根状态上改变 outcome 支持，足以解锁一个独立 K3 feasibility/pilot。它不支持：

- D-PASS 或 semantic pretraining gain；
- selector/regret/closed-loop gain；
- multi-VLA 泛化；
- 对 Object/Long 的正向语义结论；
- 用 K=1 的 D0 数值估计 effect size。

## 已落地的代码与验证

- `rase/vnext/candidate_capture.py`：不可变同步候选动作捕获、哈希和审计工具；
- `scripts/collect_rase_vnext_discovery.py`：接入同步 capture 和 fallback 实际动作轨迹；
- `scripts/freeze_rase_vnext_b2_capture_smoke.py`：B2 metadata-only manifest freezer；
- `scripts/audit_rase_vnext_b2_capture_smoke.py`：B2 parity 审计；
- `scripts/collect_rase_vnext_d0_semantic_feasibility.py`：D0 v3，含模型加载前 hash/finite/distinct/10×7 预检；
- `scripts/run_rase_vnext_b2_capture_smoke.sh`、`scripts/run_rase_vnext_d0_semantic_feasibility.sh`：可恢复运行封装；
- vNext 单元测试：`50 passed in 0.77s`。

服务器结果目录：

- `/root/autodl-tmp/RASE/runs/rase_vnext/b2_capture_smoke_v1`
- `/root/autodl-tmp/RASE/runs/rase_vnext/d0_semantic_feasibility_v3`

## 下一阶段（严格门控）

### 1. 先做独立、outcome-independent 的 K3 pilot

冻结新的 8-task×K3 cohort，不复用 B2/D0 的四个 root，不看 outcome 选任务；保持单策略 π0-fast 标签。建议四个 suite 各选两个 task，每 task 三个独立 physical roots/replicas，candidate operators 和 transform 参数在冻结文件中写死。

每个 root/operator 保存：success、progress、harm/cost、query/fallback steps、latency、capability mask。主分析是 same-root paired ranking、action-swap sensitivity、risk-coverage 和 calibration；不能只报告 episode success。

### 2. 修正 capture contract 再扩大

D0 暴露一个重要工程事实：当前 source-policy queue 在部分 boundary 只有 1 步，完整 10 步来自 fallback。K3 之前应在 inference-time 直接保存 source/requery 的完整 native output chunk，并明确每个 operator 的 chunk horizon；不能从执行后 queue 反推“完整 requery”。

### 3. K3 通过后再做 semantic target ladder

只允许低容量、冻结视觉 backbone 的 E0→E4：prior、state/history、raw action statistics、CanonicalMotionTrace、trace+frozen visual history。只有在 A/B 条件升级且 D-GATE 前置条件满足时，才考虑语义辅助训练；当前不启动 OPD、world model、RL、大规模 public dataset/teacher 或 backbone scaling。

### 4. π0.5 必须走独立 opportunity challenge

π0.5 仍属于 A-PARTIAL/failed-policy 路径，先做 outcome-independent 的 8-task×K3 opportunity smoke，不能把 π0-fast 的 D0 结果迁移成 pooled selector 证据。

### 5. 论文口径

当前最稳妥的表述是：RASE 提出一个多 VLA 风险控制框架；在单策略开发 pilot 中，先修复了同步候选捕获和边界复现协议，并验证 simulator-executed action counterfactual 具有可观测 outcome diversity。多 VLA、held-out policy、实时 selector 和闭环 Pareto 仍是后续未通过的 gates，不能提前宣称。

## 证据文件 sha256

- B2 verdict：`d88450381a5f98ac0013730026730f49089a9016d69ead05662a87f7e2ab6737`
- D0 summary：`8903a5c25231bd6cefc188be543b68ab33a1d5a63e59679abd46caeef4d99215`
- D0 frozen protocol：`52b9bb0fc7f208d442e6f8d9bc530c7cf227d77351f4e4d7c536d4b160b52753`

