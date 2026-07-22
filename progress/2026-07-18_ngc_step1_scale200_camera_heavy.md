# NGC Step-1 scale200（camera-heavy）状态池采集

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-18 15:23 CST** |
| 状态 | **已完成**（watchdog exit 0；`COLLECT_EPISODE_DONE index=199`） |
| 目的 | 在 Plus 难扰动上扩采 camera/robot 可恢复快照，供后续 NGC / fork 验收 |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |
| Git SHA | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |
| LIBERO-Plus | `/data/data2/yuxuan/LIBERO-plus` @ `4976dc3` |
| Policy | 冻结 `ckpts/smolvla_libero`（`num_steps=10`，`n_action_steps=10`） |
| 配置 | `configs/collect_scale_camera_heavy.json` |
| 产物 | `pool/ngc_step1_scale200/`（`manifest.json` + `summary.json`） |

前置：

- [Collect preflight + pilot](2026-07-17_ngc_step1_real_collect_preflight_pilot.md)
- [Collapse full 0.38%](2026-07-18_smolvla_libero_plus_collapse_full_nas10.md)

---

## 1. 终态汇总

| 指标 | 值 |
|---|---|
| 目标 episodes | 200 |
| 池内 episodes | **199**（缺 index **51**，SIGFPE） |
| 本进程新跑 | 148 failure（另 skip_list=1，already_in_pool=51） |
| states（manifest） | **3666** |
| 本进程 `states_created` | 2705（其余来自更早的 0–50 段） |
| outcomes（池内） | success **0** / failure **199** |
| 磁盘 | ~**4.0 GB** |
| 每 ep 快照 | mean **18.4**（min 14 / max 26） |
| 唯一 task_id | 185 |
| 校验抽查 | 20/20 `verify_state` 通过 |

### 配额达成（按 episode；缺 51）

| 维度 | 计划 | 实得 | Suite | 计划 | 实得 |
|---|---|---|---|---|---|
| camera | 140 | **139** | Long | 70 | 70 |
| robot | 60 | **60** | Goal | 40 | 40 |
| combination | 0 | 0 | Spatial | 30 | **29** |
| layout / other | 0 | 0 | Object | 60 | 60 |

子类：`camera/viewpoint`×139，`robot/initial_state`×60。  
难度（episode）：L3=70，L4=68，L5=61（无 L1–L2，与 camera-heavy 难档一致）。

### 跳过的 episode 51

| 项 | 值 |
|---|---|
| 原因 | 原生 **SIGFPE**（watchdog 之前手标；重启后保留） |
| 计划请求 | Spatial / camera / viewpoint / **L3** |
| 影响 | Spatial 与 camera 各少 1；不阻塞后续 fork / NGC 小试点 |

---

## 2. 工程修复（本轮踩坑）

1. **Resume 非比特可复现**：重跑已有 episode 会同 `state_key`、不同 payload → `FileExistsError`。  
   修复：`pipeline` 对 manifest 已有 `episode_id` 直接 `already_in_pool` 跳过。  
2. **Watchdog 误 skip**：曾把 Python `exit 1` 当崩溃并 skip index。  
   修复：仅信号死亡（SIGFPE/SIGSEGV/…）才写 `.skip_episodes.json`。  
3. **入口**：watchdog 改为调用 `scripts/collect_state_pool.py`（无 `rase.collect.__main__`）。

---

## 3. 解读

1. **扩采工程门：通过。** 真实 SmolVLA + Plus + 配额 + 原子池 + resume/watchdog 已跑完 200 目标（199 有效）。  
2. **0% 成功率是预期量级，不是采集失败。** Collapse full 均值 0.38%、camera 格 0%；本配置偏 L3–L5 camera-heavy，池以失败轨迹为主，适合 NGC「难状态」用途。  
3. **不要把本池 SR 当论文数字。** 与 clean LIBERO 70% 不可比。  
4. **Combination 仍未开。** 需配对协议 + fork 验收后再谈 20% 配额。  
5. **相对 pilot（367 states / 20 ep）**：规模约 **10× episodes、~10× states**，已够做 fork 往返与小规模 NGC recovery 试点。

---

## 4. 后续待办（建议顺序）

1. **Pool → ForkableEnv 往返**：**已通过**（`POOL_FORK_GATE_DONE passed=2`，steps=50）。
2. （可选）补采 index 51，或接受缺口。
3. **W2 小试点**：**已完成** → [W2 candidates pilot](2026-07-18_ngc_w2_candidates_pilot.md)
   （2 states，K=8，mean endpoint L2≈1.69；未测续完成功率）。
4. **下一步 W3**：candidate → fork 续完 + Wilson triage（需补端到端脚本）。
5. **暂不**启动全量 ~20k 池、**不**微调 SmolVLA、**不开** combination。

---

## 5. 一句话结论

Camera-heavy scale200 采集完成：**199 ep / 3666 states / ~4 GB**；fork 往返已验收；W2 candidates pilot 已通过，下一步做 **fork 续完 + Wilson triage**。
