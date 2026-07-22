# SmolVLA LIBERO-Plus collapse full（seed 0, n_action_steps=10）

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-18 12:58 CST**（manifest `updated_at` ≈ `2026-07-18T04:50:13Z`） |
| 状态 | **已完成**（3142 `completed` + 7 `skipped` = 3149；无 pending/running） |
| 目的 | 冻结 SmolVLA 在 Plus **全部** camera/robot 分类任务上的塌缩地图 |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |
| Git SHA | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |
| LIBERO-Plus | `/data/data2/yuxuan/LIBERO-plus` @ `4976dc3` |
| Policy | 冻结 `ckpts/smolvla_libero`（`num_steps=10`，`n_action_steps=10`） |
| Profile | `full`，`episodes_per_task=1`，`seed=0` |
| 输出 | `runs/collapse_full_nas10/` |

前置：

- [Clean LIBERO 70.0%](2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md)
- [Collapse smoke 2.5%](2026-07-17_smolvla_libero_plus_collapse_smoke_nas10.md)
- [NGC collect pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md)

---

## 1. 核心结果（仅 `completed`；skipped 不进均值）

| 汇总 | 值 |
|---|---|
| 总体均值 | **0.382%**（12/3142 任务成功） |
| camera | **0.000%**（0/1597） |
| robot | **0.777%**（12/1545） |
| skipped | **7**（原生 SIGFPE/SIGSEGV/SIGILL，`--max-attempts 2` 或手标） |

### 维度 × 难度（成功数 / 完成数）

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| camera | 0/198 | 0/360 | 0/325 | 0/228 | 0/486 |
| robot | 3/248 | 2/289 | 3/374 | 2/285 | 2/349 |

### 按 suite

| Suite | 均值 | 成功 |
|---|---|---|
| libero_goal | 1.35% | 11/816 |
| libero_spatial | 0.14% | 1/721 |
| libero_object | 0.00% | 0/794 |
| libero_10 | 0.00% | 0/811 |

### 对照（分表，勿混报）

| 设定 | 结果 |
|---|---|
| Clean LIBERO，nas10，50 ep/task | **70.0%** |
| Collapse smoke，40×1 ep | **2.5%**（camera 5% / robot 0%） |
| **本 full**，3142×1 ep | **0.38%**（camera 0% / robot 0.78%） |

说明：smoke 抽到少数易格（含 camera L1），full 对全部分类任务平均后更低；**camera 全灭、robot 仅 Goal 上偶发成功**。每任务 1 ep，单任务只能是 0/100%。

### 12 个成功任务（全部 robot；11 个 Goal bowl→plate）

多为 `put_the_bowl_on_the_plate ... initstate_*`；另 1 个 spatial L3。详见 manifest。

---

## 2. 工程备注

- 运行中多次原生崩溃（SIGFPE / segfault / SIGILL）；`--max-attempts 2` + 外层 while 续跑。
- 7 skipped 不计入成功率；覆盖率 3142/3149 ≈ 99.8%。
- 墙钟约 2026-07-17 13:37 → 2026-07-18 12:50 CST（含中断）。

---

## 3. 后续

1. 写汇总脚本 / 热力图（可选）  
2. 扩采状态池：倾斜 **camera 全难度** + **Long/Object**（本表上最塌）  
3. Pilot 池 fork 往返（可选）  
4. NGC / 恢复实验；暂不 20k / 不微调 / combination 待协议  

---

## 4. 一句话

**Full collapse：3142 任务均值 0.38%（camera 0% / robot 0.78%），相对 clean 70% 断崖确认；下一步按塌缩图扩采状态池并进入恢复方法。**
