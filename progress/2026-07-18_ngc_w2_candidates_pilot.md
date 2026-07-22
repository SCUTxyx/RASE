# NGC W2 K=8 candidates 小试点

| 项 | 内容 |
|---|---|
| 记录时间 | **2026-07-18 21:29 CST** |
| 状态 | **已完成**（`CANDIDATES_DONE n=2 written=1`；多样性门禁通过） |
| 目的 | 从 scale200 池 restore 状态，用冻结 SmolVLA 生成 K=8 候选，验收形状/provenance/多样性 |
| 机器 | SCUT-407-03（`/data/data2/yuxuan/RASE`） |
| Conda | `smolvla`（Python 3.12） |
| Git SHA | `ddb2dc7cb0ce596f3d4adf36c3d2fb9d06c8f714` |
| `env.lock.md` SHA-256 | `0609adae34282dfba0408745070c8d718385124f1751c6d74d2b0af14a71b0f2` |
| LIBERO-Plus | `/data/data2/yuxuan/LIBERO-plus` @ `4976dc3` |
| Policy | 冻结 `ckpts/smolvla_libero`（`num_steps=10`，`n_action_steps=10`） |
| 配置 | `configs/candidates_w2_pilot.json` |
| 产物 | `runs/ngc_w2_candidates_pilot/`（`summary.json` + `candidates/*.npz`） |

前置：

- [scale200 camera-heavy pool](2026-07-18_ngc_step1_scale200_camera_heavy.md)（199 ep / 3666 states；pool fork gate 已过）

---

## 1. 终态汇总

| 指标 | 值 |
|---|---|
| 采样 states | **2**（camera + robot 偏好抽样） |
| 磁盘 artifacts | **2**（`written=1`，`skipped=1` resume） |
| shape | **`[8, 10, 7]`**（两份均合规） |
| temperature | 0.7 |
| `policy_hash` | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| mean endpoint L2 | **1.692** |
| min endpoint L2 | **0.376**（≫ `min_mean_endpoint_l2=1e-4`） |
| max endpoint L2 | **5.974** |
| mean chunk L2 | **0.840** |

### per-state

| state_key | mean endpoint L2 | min endpoint L2 | mean chunk L2 |
|---|---|---|---|
| `sp1_69771078bea26d7efe824db6d389b57c` | 2.107 | 0.376 | 1.034 |
| `sp1_f478ef09f38e3c249c53ded5b1820595` | 1.276 | 0.405 | 0.646 |

---

## 2. 过程备注

1. **首次跑挂在 seed 溢出**：`candidate_base_seed` 混入 `state_key[:8]` 后超过 `2**32-1`，`np.random.seed` 抛 `ValueError`。已对 `2**32 - K` 取模修复；首个 state 的 `.npz` 在崩溃前已写出。
2. **重跑为 resume**：第二次 `CANDIDATES_DONE n=2 written=1`——已有相同 `policy_hash`+temperature 的 artifact 被 skip，只补写第二个 state。
3. **本结果不回答「换候选是否成功」**：未做 fork 续完、无成功率、无 Set A/B/C。仅证明候选可生成且未 mode-collapse。

---

## 3. 解读

1. **W2 多样性 sanity check：通过。** 候选末端位移未坍缩，协议形状与 provenance 齐全。
2. **工程链路通：** pool → ForkableEnv restore → SmolVLA K=8 sample → 原子写 artifact → summary。
3. **下一步必须进 W3：** 对 `(state, candidate)` 做 fork 续跑 + Wilson triage，才能谈 NGC / 可恢复性。

---

## 4. 后续待办（建议顺序）

1. 合同测试：`test_wilson_triage` / `test_scheduler_resume` / `test_oracle_protocol`。
2. **补脚本**（当前缺端到端入口）：pool state + candidate chunk → fork 执行 → success，挂 `DiskRolloutScheduler` + `adaptive_sample`。
3. 小试点：2 states × 8 candidates，先固定少量续完（如每候选 3→最多 10），输出 triage 计数。
4. 验收门槛（runbook）：约 20 次 forked rollouts 记墙钟；fork 不确定 / 形状缺失则停。
5. OFT oracle RPC 交叉验证可后置；**暂不**全量池、微调、combination。

---

## 5. 一句话结论

W2 candidates pilot **通过**：**2 states / K=8 / shape `[8,10,7]` / mean endpoint L2≈1.69**；下一步做 **fork 续完 + Wilson triage**，才能判定候选替换是否成功。
