# SmolVLA Clean LIBERO Baseline（seed 0, n_action_steps=10）

| 项 | 内容 |
|---|---|
| 日期 | 2026-07-16 |
| 状态 | 已完成（四 suite × seed 0） |
| 目的 | RASE W1 frozen SmolVLA clean baseline（n_action_steps=10 配置线） |
| 总成功率（四 suite 均值） | **70.0%** |

说明：本记录对应项目统一的吞吐配置 `n_action_steps=10`。  
另有 `n_action_steps=1` 的高精度对照（Spatial 已测 82.0%，见 `runs/smolvla_best_libero_spatial_seed0`），不要与本表混报。

---

## 1. 任务概述

| 项目 | 内容 |
|---|---|
| 任务类型 | Clean LIBERO 仿真评测（无扰动） |
| Benchmark | LIBERO 四 suite，各 10 个 task |
| 策略 | 冻结官方 SmolVLA checkpoint，不做微调 |
| 指标 | 每 suite `pc_success`；四 suite 算术平均 |
| Seed | 0（正式 baseline 后续还需补 seed 1、2） |

### Suite 定义

| Suite | 含义 | Task 数 | Episodes |
|---|---|---|---|
| libero_spatial | 空间关系 | 10 | 500（50×10） |
| libero_object | 物体操作 | 10 | 500 |
| libero_goal | 目标导向 | 10 | 500 |
| libero_10 | 长 horizon（Long） | 10 | 500 |
| 合计 | — | 40 | 2000 |

---

## 2. 核心结果

| Suite | Success Rate | Episodes | 评测耗时 | 秒/episode |
|---|---|---|---|---|
| libero_spatial | 67.2% | 500 | 1.68 h | 12.1 s |
| libero_object | 90.2% | 500 | 1.29 h | 9.3 s |
| libero_goal | 78.6% | 500 | 1.41 h | 10.2 s |
| libero_10 | 44.0% | 500 | 3.02 h | 21.7 s |
| **均值** | **70.0%** | **2000** | **7.40 h** | — |

### 与参考数字对比

| Suite | 论文（原模型） | HF Hub 社区复现（约） | 本次（nas10） |
|---|---|---|---|
| Spatial | ~90% | 63–73% | 67.2% |
| Object | ~96% | ~93% | 90.2% |
| Goal | ~92% | ~81% | 78.6% |
| Long (libero_10) | ~71% | 43–56% | 44.0% |
| 均值 | 87.3% | ~73% | 70.0% |

解读：整体落在 Hub checkpoint 社区区间附近；低于论文 87.3% 主要因公开权重 ≠ 论文自训模型（架构/训练差异），不是环境配错。  
`n_action_steps=10` 相对 `=1` 会再掉几个点（尤其 Long），符合预期。

---

## 3. 评测参数（锁死配置）

| 参数 | 值 |
|---|---|
| Checkpoint | `/root/autodl-tmp/RASE-project/ckpts/smolvla_libero`（HuggingFaceVLA/smolvla_libero） |
| policy.num_steps | 10（flow-matching 推理步数） |
| policy.n_action_steps | 10 |
| env.type | libero |
| eval.n_episodes | 50 / task |
| eval.batch_size | 2（防 OOM；不影响 50 ep 下成功率） |
| policy.device | cuda |
| seed | 0 |
| 动作模式 | relative（LeRobot LIBERO 默认） |
| 初始状态 | init_states=True（官方 init states） |

### 运行环境

| 项目 | 值 |
|---|---|
| Conda env | `/root/autodl-tmp/envs/smolvla` |
| Python | 3.12 |
| LeRobot | 0.5.1 |
| GPU | NVIDIA GeForce RTX 5090 |
| 渲染 | `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` |
| HF 镜像 | `HF_ENDPOINT=https://hf-mirror.com` |

更完整的环境锁定见 [../env.lock.md](../env.lock.md)。

### 启动命令（按 suite 串行，防 OOM）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/smolvla
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/RASE

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "=== Running ${SUITE} ==="
  lerobot-eval \
    --policy.path=/root/autodl-tmp/RASE-project/ckpts/smolvla_libero \
    --policy.num_steps=10 \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=$SUITE \
    --eval.n_episodes=50 \
    --eval.batch_size=2 \
    --policy.device=cuda \
    --seed=0 \
    --output_dir=/root/autodl-tmp/RASE/runs/smolvla_clean_seed0_nas10_${SUITE} \
    2>&1 | tee /root/autodl-tmp/RASE/runs/smolvla_clean_seed0_nas10_${SUITE}.log
done
```

---

## 4. 时间线

| Suite | 开始 | 结束 | 墙钟耗时 |
|---|---|---|---|
| libero_spatial | 2026-07-16 15:45:14 | 17:27:12 | ~1.7 h |
| libero_object | 17:27:20 | 18:45:50 | ~1.3 h |
| libero_goal | 18:45:58 | 20:11:38 | ~1.4 h |
| libero_10 | 20:11:45 | 23:14:17 | ~3.0 h |
| 全程 | 15:45 | 23:14 | ~7.5 h |

---

## 5. 逐 task 成功率

### libero_spatial — 67.2%（336/500）

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| SR % | 70 | 74 | 78 | 56 | 68 | 74 | 60 | 56 | 74 | 62 |

### libero_object — 90.2%（451/500）

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| SR % | 90 | 98 | 96 | 86 | 94 | 68 | 96 | 94 | 94 | 86 |

### libero_goal — 78.6%（393/500）

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| SR % | 76 | 90 | 78 | 68 | 94 | 88 | 66 | 100 | 74 | 52 |

### libero_10 — 44.0%（220/500）

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| SR % | 32 | 52 | 58 | 72 | 10 | 66 | 38 | 34 | 28 | 50 |

难点：`libero_10` task 4 仅 10%；task 8 28%。Object 整体最稳，仅 task 5（68%）明显偏低。

---

## 6. 产物路径

| 类型 | 路径 |
|---|---|
| Spatial 结果 | `runs/smolvla_clean_seed0_nas10_libero_spatial/eval_info.json` |
| Object 结果 | `runs/smolvla_clean_seed0_nas10_libero_object/eval_info.json` |
| Goal 结果 | `runs/smolvla_clean_seed0_nas10_libero_goal/eval_info.json` |
| Long 结果 | `runs/smolvla_clean_seed0_nas10_libero_10/eval_info.json` |
| 各 suite 日志 | `runs/smolvla_clean_seed0_nas10_libero_{spatial,object,goal,10}.log` |
| 视频 | 各 run 目录下 `videos/`（每 task 保存前 10 条 episode） |

读取汇总示例：

```python
import json
from pathlib import Path

runs = Path("/root/autodl-tmp/RASE/runs")
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
vals = []
for s in suites:
    d = json.load(open(runs / f"smolvla_clean_seed0_nas10_{s}" / "eval_info.json"))
    pc = d["overall"]["pc_success"]
    vals.append(pc)
    print(f"{s}: {pc:.1f}%")
print(f"mean: {sum(vals)/len(vals):.1f}%")
```

---

## 7. 相关对照（勿混入本表）

| 实验 | 配置差异 | Spatial | 备注 |
|---|---|---|---|
| 预检 | nas10, 2 ep/task | 70% | 方差大，非正式 |
| 本实验 | nas10, 50 ep, 四 suite | 67.2%（四 suite 均值 70.0%） | W1 吞吐配置线 |
| 高精度对照 | nas1, 50 ep, Spatial only | 82.0% | `smolvla_best_libero_spatial_seed0` |

---

## 8. 后续待办

- 补 seed 1、2（同配置），报均值 ± 标准差
- 若需要「最强 frozen」数字：补齐 `n_action_steps=1` 的 object / goal / libero_10
- 不要微调 SmolVLA；下一步主线是 LIBERO-Plus 塌缩曲线 + NGC 采集 + RASE selector
- 加 RASE 时：VLA 配置必须与本 baseline 完全一致（同 checkpoint、同 `n_action_steps`）

---

## 9. 一句话结论

SmolVLA 官方 checkpoint 在 clean LIBERO 上（`n_action_steps=10`, 50 ep/task, seed 0）四 suite 成功率分别为 **67.2 / 90.2 / 78.6 / 44.0%**，均值 **70.0%**，与 Hub 社区复现量级一致，可作为 RASE `n_action_steps=10` 配置下的 frozen before baseline。
