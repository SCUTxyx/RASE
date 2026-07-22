# SmolVLA LIBERO-Plus collapse smoke（seed 0, n_action_steps=10）

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-17 12:23 CST**（UTC 跑完约 `2026-07-17T04:21:58Z`） |
| 状态 | **已完成**（smoke 40/40 `completed`） |
| 目的 | 冻结 SmolVLA 在 LIBERO-Plus camera/robot × L1–L5 上的**定性塌缩摸底**（工程验收 + 粗曲线） |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |
| Git SHA（manifest） | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |
| LIBERO-Plus | commit `4976dc3` |
| LeRobot | 0.5.1 |
| Backend | 默认 `lerobot` → `rase.backends.lerobot_libero_plus:evaluate` |

前置记录：

- [Clean LIBERO baseline 70.0%](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md)（**勿与本表混报**）
- [W1 dry-run gates](2026-07-17_w1_dry_run_gates.md)

---

## 1. 评测设定

| 项 | 值 |
|---|---|
| Profile | `smoke`（每格 1 任务） |
| 维度 | camera, robot |
| 难度 | L1–L5 |
| Suites | 四 suite 全开 |
| 任务数 | **40**（4×2×5） |
| Episodes / task | **1** |
| Seed | 0 |
| `num_steps` / `n_action_steps` | **10 / 10** |
| Policy | 冻结 `ckpts/smolvla_libero` |
| Device | cuda（本机 `CUDA_VISIBLE_DEVICES=1`） |

**重要：** 每任务仅 1 episode，成功率只能是 0% 或 100%。本表只作工程验收与定性摸底，**不是**论文级塌缩曲线。

---

## 2. 核心结果

| 汇总 | 值 |
|---|---|
| 总体均值 | **2.5%**（1/40 任务成功） |
| camera | **5.0%**（1/20） |
| robot | **0.0%**（0/20） |
| 评测合计耗时 | ~386 s（~9.7 s/ep） |

### 维度 × 难度（每格 4 任务均值）

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| camera | 25%（1/4） | 0% | 0% | 0% | 0% |
| robot | 0% | 0% | 0% | 0% | 0% |

### 按 suite

| Suite | 均值 | 成功任务 |
|---|---|---|
| libero_spatial | 10% | 1/10 |
| libero_object | 0% | 0/10 |
| libero_goal | 0% | 0/10 |
| libero_10 | 0% | 0/10 |

### 唯一成功

- `libero_spatial:609:camera:L1`
- 任务名：`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_2_352_initstate_0`

### 对照（勿混报）

| 设定 | 结果 |
|---|---|
| Clean LIBERO，seed 0，nas10，50 ep/task | **70.0%** 四 suite 均值 |
| 本 smoke（Plus 扰动，1 ep/task） | **2.5%** |

定性结论：相对 clean，camera/robot 扰动下出现**强塌缩**信号；robot 侧在本次抽样中更硬。数字方差极大，不可直接当 before 终值。

---

## 3. 窄冒烟（链路验收）

| 项 | 内容 |
|---|---|
| 输出 | `runs/collapse_smoke_camera_l3_spatial/` |
| 任务 | `libero_spatial:617:camera:L3`（1 ep） |
| 结果 | `completed`，`pc_success=0.0%` |
| 作用 | 确认 Plus 路径、init-state patch、SmolVLA 推理链路可跑通 |

首次失败原因：`tokenizer_processor` 访问 HuggingFace 超时；本地镜像/缓存就绪后 resume 成功。另修过 resume 时 `levels` list/tuple 导致的 provenance 误拒。

---

## 4. 命令（可复现）

```bash
conda activate smolvla
cd /data/data2/yuxuan/RASE

export LIBERO_PLUS_ROOT=/data/data2/yuxuan/LIBERO-plus
export RASE_POLICY_PATH=/data/data2/yuxuan/RASE/ckpts/smolvla_libero
export RASE_ENV_LOCK=/data/data2/yuxuan/RASE/env.lock.md
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

export RASE_COLLAPSE_OUTPUT=/data/data2/yuxuan/RASE/runs/collapse_smoke_nas10
python scripts/eval_libero_plus_collapse.py \
  --config configs/collapse_camera_robot.yaml \
  --profile smoke
```

---

## 5. 产物

- Manifest：`runs/collapse_smoke_nas10/manifest.json`
- 每任务：`runs/collapse_smoke_nas10/tasks/*/metrics.json`
- 窄冒烟：`runs/collapse_smoke_camera_l3_spatial/`

---

## 6. 验收清单

- [x] 40 任务全部 `completed`
- [x] nas10（未混入 nas1）
- [x] 与 clean 70.0% 分表记录
- [ ] 每格 ≥20 ep 的正式塌缩曲线（**未做**；full 仍是 1 ep/task）
- [x] `--profile full` → 见 [collapse full nas10](2026-07-18_smolvla_libero_plus_collapse_full_nas10.md)（0.38%）

---

## 7. 后续

1. ~~Collapse full~~ → 已完成（2026-07-18，0.38%）  
2. ~~状态池真采集 pilot~~ → 已完成  
3. 扩采状态池 + NGC/恢复；暂不 20k / 不微调  

一句话：**2026-07-17：collapse smoke 真评测完成（40 任务，均值 2.5%）。同线 full 见 2026-07-18 记录（0.38%）。**
