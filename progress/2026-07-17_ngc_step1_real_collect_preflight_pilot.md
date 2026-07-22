# NGC Step-1 真采集：preflight + camera/robot pilot

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-17 13:32 CST** |
| 状态 | **已完成**（preflight 2 ep + pilot 20 ep；工程门通过） |
| 目的 | 验收真实 SmolVLA + LIBERO-Plus 状态池采集链路（非 dry-run） |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |
| Git SHA | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |
| LIBERO-Plus | `/data/data2/yuxuan/LIBERO-plus` @ `4976dc3` |
| Policy | 冻结 `ckpts/smolvla_libero`（`num_steps=10`，`n_action_steps=10`） |
| Tokenizer | `ckpts/SmolVLM2-500M-Instruct`（`RASE_TOKENIZER_PATH`） |
| Adapter | `rase.collect.lerobot_libero_plus_adapter:make_adapter` |

前置记录：

- [Clean LIBERO baseline 70.0%](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md)
- [W1 dry-run gates](2026-07-17_w1_dry_run_gates.md)
- [Collapse smoke 2.5%](2026-07-17_smolvla_libero_plus_collapse_smoke_nas10.md)

---

## 1. 实现与配置

| 项 | 路径 / 设定 |
|---|---|
| Adapter | `rase/collect/lerobot_libero_plus_adapter.py` |
| Preflight 配置 | `configs/collect_preflight.json`（2 ep） |
| Pilot 配置 | `configs/collect_pilot.json`（20 ep） |
| Runbook | `docs/runbooks/collect_state_pool.md` |
| 快照节奏 | 每 2 个 action-chunk（步号 0,2,4,…） |
| 成功保留 | 20%（失败 100%） |
| 采样维 | **仅 camera + robot**（combination 未开） |
| 设备 | `CUDA_VISIBLE_DEVICES=1`，`MUJOCO_GL=egl` |

Catalog 修复：上游 `task_classification.json` 中 121 条 Light Conditions 的 `difficulty_level` 为 `null`；`_load_catalog` 跳过 null / 非法 level，避免采集启动崩溃。

---

## 2. Preflight（2 ep）

| 指标 | 值 |
|---|---|
| 完成时间 | ~2026-07-17 12:36 CST |
| episodes | 2（success 0 / failure 2） |
| snapshots_seen / retained / created | 41 / 41 / 41 |
| 配额 | camera×Long + robot×Goal 各 1 |
| 产物 | `pool/ngc_step1_preflight/`（~38 MB） |

任务：`libero_10_000758`（Long, camera, L4）、`libero_goal_000664`（Goal, robot, L4）。  
结论：**链路可跑通**；0/2 成功在 Plus 硬扰动上不视为失败。

---

## 3. Pilot（20 ep）

| 指标 | 值 |
|---|---|
| 完成时间 | ~2026-07-17 13:25 CST |
| episodes | 20（success **1** / failure **19** → **5%**） |
| snapshots_seen / retained / created | 378 / 367 / 367 |
| 产物 | `pool/ngc_step1_pilot/manifest.json`（367 states，~383 MB） |

### 配额（与配置整数近似一致）

| 维度 | 数 | Suite | 数 |
|---|---|---|---|
| camera | 10 | Long | 8 |
| robot | 10 | Goal | 5 |
| combination | 0 | Spatial | 4 |
| layout / other | 0 | Object | 3 |

### 难度分布（按 episode）

| Level | 数 |
|---|---|
| L3 | 4 |
| L4 | 9 |
| L5 | 7 |

偏难（无 L1–L2），与低成功率一致。

### 唯一成功 episode

| 项 | 值 |
|---|---|
| episode | `ep-0135276d-00000006` |
| task | `libero_goal_000619` |
| suite / dim | Goal / robot（initial_state） |
| level | 4 |
| instruction | `put the bowl on the plate ... initstate 119` |
| 保留快照 | 1（成功 ep 的 20% 采样） |

### Bundle 抽查

单状态含：`sim_state.npz`、`obs_agentview.png`、`obs_wrist.png`、`proprio.npy`、`meta.json`、`checksums.json`。

---

## 4. 解读

1. **工程门：通过。** 真实 policy + Plus env + 配额采样 + 状态池原子写 / manifest 已验收。  
2. **成功率：仅定性。** 1/20 = 5%，与 collapse smoke 均值 2.5% 同量级；**不能**当论文 SR。  
3. **池以失败轨迹为主。** 对 NGC Step-1（难状态上的恢复/对比）通常是期望形态。  
4. **Combination 仍未启用。** 上游无 camera+robot 联合类目；需独立配对协议 + fork 验收后再开 20% 配额。

---

## 5. 后续待办

1. （可选）对 pilot 中 1 个失败 + 1 个成功 snapshot 做 ForkableEnv 往返（同动作序列，像素一致 / pose 误差 `<1e-9`）。  
2. 按需扩采 camera/robot（更大 `episodes` / 平衡 suite 与 L1–L5）；**暂不**开 combination。  
3. 正式塌缩曲线（每格多 ep 或 `--profile full`）可与扩采并行，不挡采集。  
4. **暂不**启动全量 ~20k NGC / **不**微调 SmolVLA。

---

## 6. 一句话结论

**2026-07-17：NGC 真采集 preflight（2 ep）与 camera/robot pilot（20 ep，5% SR，367 states）工程验收通过；下一步优先可选 fork 往返，再决定扩采规模。**
