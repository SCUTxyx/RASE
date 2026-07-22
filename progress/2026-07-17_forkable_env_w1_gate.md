# ForkableEnv W1 硬门槛验收

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-17 11:36:22 CST**（UTC 2026-07-17T03:36:22Z） |
| 状态 | **已完成**（4/4 集成测试通过） |
| 目的 | W1 硬门槛：证明 `ForkableEnv` 快照/恢复可信，方可进入 NGC 采集 |
| 机器 | SCUT-407-03（本机 `/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |

说明：本记录**不是** VLA 成功率评测；验收对象是环境分叉（sim + controller + RNG）的确定性。  
关联 baseline：[2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md)（frozen nas10 均值 70.0%）。

---

## 1. 验收结论

| 测试 | 结果 | 含义 |
|---|---|---|
| `test_same_snapshot_replays_identically` | PASS | 同一 snapshot 恢复两次 × 50 步，观测一致、状态差 `<1e-9` |
| `test_restore_rewinds_libero_process_global_rng` | PASS | restore 后 NumPy 全局 RNG 被拨回 |
| `test_noisy_observation_replays_identically` | PASS | sensor-noise 任务短轨迹可确定性重放 |
| `test_snapshot_rejects_a_different_task_before_mutation` | PASS | 跨任务 restore 被拒绝且目标状态不变 |

**汇总：4 passed，0 failed，0 skipped（集成测试全部执行）。**

另：`scripts/smoke_test.py --steps 5` 与 `--steps 50` 均已通过。

---

## 2. 配置与路径

| 参数 | 值 |
|---|---|
| `RASE_TEST_BDDL` | `/data/data2/yuxuan/LIBERO-plus/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl` |
| `RASE_TEST_OTHER_BDDL` | `/data/data2/yuxuan/LIBERO-plus/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl` |
| GPU / EGL | `CUDA_VISIBLE_DEVICES=1`，`MUJOCO_EGL_DEVICE_ID=1`，`MUJOCO_GL=egl`，`PYOPENGL_PLATFORM=egl` |
| `RASE_TEST_GPU_ID` | 1 |
| LIBERO-Plus | `/data/data2/yuxuan/LIBERO-plus` @ `4976dc3`（含本地 NumPy2 补丁，见下） |
| 核心代码 | `rase/envs/forkable_env.py`，`rase/envs/snapshot.py` |
| 契约文档 | `docs/forkable_env_contract.md` |

### 启动命令（复现）

```bash
conda activate smolvla
cd /data/data2/yuxuan/RASE

export BDDL_ROOT=/data/data2/yuxuan/LIBERO-plus/libero/libero/bddl_files/libero_spatial
export RASE_TEST_BDDL="${BDDL_ROOT}/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
export RASE_TEST_OTHER_BDDL="${BDDL_ROOT}/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export RASE_TEST_GPU_ID=1

pytest -q tests/test_fork_roundtrip.py \
  tests/test_fork_noise_rng.py \
  tests/test_wrong_task_restore.py
```

---

## 3. 过程中修复（时间线）

| 时间（约） | 事件 |
|---|---|
| 2026-07-17 上午 | 仓库 W1–W3 代码/文档骨架落地；轻量单测 `41 passed, 4 skipped` |
| 同日 | 缺 `Wand` / ImageMagick / `scikit-image` / `gym` → 补齐后 `ControlEnv` 可 import |
| 同日 | `language_instruction` + `hasattr` 触发 `KeyError` → 改为安全 `getattr` |
| 同日 | 5 步 smoke：`PASS: two 5-step forks were deterministic` |
| 同日 | noise 测试因 NumPy 2 移除 `np.float_` 失败 → 补丁 `dtype=np.float64` |
| **11:36 CST** | 四项集成测试全绿，记入本文件 |

本地补丁说明：[`third_party/patches/libero_plus_numpy2_float64.md`](../third_party/patches/libero_plus_numpy2_float64.md)。

---

## 4. 环境补充（相对 env.lock 增量）

为跑通 LIBERO-Plus `env_wrapper`，`smolvla` 环境额外安装：

- conda-forge：`imagemagick`
- pip：`Wand==0.7.2`，`scikit-image`，`gym==0.26.2`

详见 [`env.lock.md`](../env.lock.md)、[`envs/smolvla.yaml`](../envs/smolvla.yaml)。

---

## 5. 后续待办

1. LIBERO-Plus collapse **dry-run**（不加载 policy）→ 再 smoke 真跑 camera/robot  
2. NGC 状态池 **dry-run**（`collect_smoke.yaml`）验证幂等写入  
3. 配 SmolVLA adapter 后做小规模 Step 1 真采集（Fork 已绿，允许进入）  
4. **不要**在未写 collapse/state-pool smoke 前启动全量 20k 采集

---

## 6. 一句话结论

**2026-07-17 11:36 CST：ForkableEnv W1 硬门槛在本机通过（4/4），可进入 LIBERO-Plus 塌缩与 NGC 状态池小规模试验；frozen SmolVLA nas10 clean baseline 仍为 70.0%。**
