# W1 空跑验收：状态池 dry-run + LIBERO-Plus 塌缩 dry-run

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-17 11:46:19 CST**（UTC 2026-07-17T03:46:19Z） |
| 状态 | **已完成**（两项 dry-run 均通过；未执行真 policy） |
| 目的 | 在 ForkableEnv 硬门槛之后，验收 NGC 状态池写入/续采与塌缩评测任务展开 |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |
| Git SHA（collapse manifest 记录） | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |

前置记录：

- [SmolVLA clean baseline 70.0%](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md)
- [ForkableEnv W1 gate 4/4](2026-07-17_forkable_env_w1_gate.md)

---

## 1. NGC 状态池 dry-run

| 项 | 内容 |
|---|---|
| 命令时间 | 2026-07-17 约 11:36–11:40 CST（含一次路径冲突修复后的重跑） |
| 命令 | `python scripts/collect_state_pool.py --config configs/collect_smoke.yaml --dry-run` |
| Adapter | `dry-run`（合成数据；不加载 SmolVLA / MuJoCo） |
| 配置 | `configs/collect_smoke.yaml`：10 episodes，chunk=6，cadence=2，成功保留 20% |

### 结果

| 指标 | 值 |
|---|---|
| episodes | 10（success 2 / failure 8；dry-run 规则，非真实成功率） |
| snapshots_seen | 30 |
| snapshots_retained | 26 |
| 磁盘 states（manifest） | **26** |
| 二次运行 | `states_created=0`，`states_idempotently_skipped=26` |

### 配额（10 ep 整数近似）

| 维度 | 数 | Suite | 数 |
|---|---|---|---|
| camera | 3 | Long | 4 |
| robot | 3 | Goal | 3 |
| combination | 2 | Spatial | 2 |
| layout | 1 | Object | 1 |
| other | 1 | | |

### 产物

- 目录：`runs/collect_smoke/state_pool/`
- Manifest：`runs/collect_smoke/state_pool/manifest.json`
- 单状态文件：`sim_state.npz`、`obs_*.png`、`proprio.npy`、`meta.json`、`checksums.json`

### 过程问题与修复

首次在路径布局从 `states/sp1_*` 迁到 `task/episode/step` 后出现 `conflicting manifest entry`。  
已修 `rase/collect/state_pool.py`：同 key 且内容一致时按 manifest 既有路径幂等跳过。修复后两次 dry-run 均跳过 26 条。

### 结论

**状态池 schema / 原子写 / 校验和 / 断点续采空跑通过。** 未证明真实扰动采集。

---

## 2. LIBERO-Plus 塌缩 dry-run

| 项 | 内容 |
|---|---|
| 命令时间 | **2026-07-17 11:36:16 CST**（manifest `recorded_at`: `2026-07-17T03:36:16+00:00`） |
| 命令 | 见下 |
| Profile | `smoke` |
| Policy | **未加载**（`--dry-run`） |

```bash
export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
export RASE_COLLAPSE_OUTPUT=/data/data2/yuxuan/RASE/runs/collapse_dry_run
export RASE_ENV_LOCK=/data/data2/yuxuan/RASE/env.lock.md

python scripts/eval_libero_plus_collapse.py \
  --config configs/collapse_camera_robot.yaml \
  --profile smoke \
  --dry-run
```

### 终端摘要

```json
{
  "dry_run": true,
  "manifest": "/data/data2/yuxuan/RASE/runs/collapse_dry_run/manifest.json",
  "pending_tasks": 40,
  "profile": "smoke",
  "selected_tasks": 40
}
```

### Manifest 核对（正确）

| 检查 | 结果 |
|---|---|
| 任务数 | 40 = 4 suite × 2 维（camera/robot）× 5 level，每格 1 task |
| camera / robot | 20 / 20 |
| L1–L5 | 各 8 |
| 四 suite | 各 10 |
| status | 全部 `pending` |
| 评测配置 | `num_steps=10`，`n_action_steps=10`，`seed=0`，`episodes_per_task=1`，`batch_size=2` |
| provenance | 含 `git_sha`、`env_lock_sha256`、`resolved_config`、`recorded_at` |

产物：`runs/collapse_dry_run/manifest.json`

### 结论

**塌缩任务展开与可续跑 manifest 空跑通过。** 尚未得到任何 Plus 成功率/塌缩数字。

---

## 3. 当日 W1 进度总览

| 里程碑 | 状态 | 时间 |
|---|---|---|
| ForkableEnv 4/4 | 完成 | 见 `2026-07-17_forkable_env_w1_gate.md`（11:36 CST） |
| 状态池 dry-run + 幂等 | 完成 | ~11:36–11:40 CST |
| 塌缩 dry-run（40 pending） | 完成 | manifest `recorded_at` 11:36:16 CST |
| 塌缩 smoke 真跑 | **已完成** | 见 [collapse smoke nas10](2026-07-17_smolvla_libero_plus_collapse_smoke_nas10.md)（40/40，均值 2.5%） |
| NGC 真采集 | **已完成** | 见 [preflight+pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md)（2+20 ep，pilot 5% SR，367 states） |

---

## 4. 后续待办

1. ~~Collapse smoke 真评测 + progress 记录~~ → 已完成（2026-07-17）  
2. ~~实现状态池真采集 adapter + preflight/pilot~~ → 已完成（见 [preflight+pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md)）  
3. （可选）pilot snapshot fork 往返；按需扩采 camera/robot  
4. 正式塌缩曲线（每格多 ep 或 `--profile full`）后补，不挡采集  
5. 暂不启动全量 20k NGC / 不微调 SmolVLA；combination 待配对协议

---

## 5. 一句话结论

**2026-07-17 11:46 CST：状态池与塌缩两条 dry-run 管道均验收通过。同日后续 collapse smoke（2.5%）与 NGC 真采集 pilot（5% SR / 367 states）均已完成；下一步见 [preflight+pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md) §5。**
