# RASE Goldilocks Regime Screening Plan（2026-08-20）

> 上游结论：96-state Goal/Long Eligibility FAIL（`H_within=0`，oracle gain 0pp，
> fallback 弱支配，0 continue-only）。三个独立协议（legacy proxy / 修正 recovery
> pilot / 96-state 真实终局）均得 `H_within=0` → 绑定约束是 domain/policy-pair
> 结构，不是模型容量/特征/标签。下一步：域与策略对筛查，不调模型。

---

## 0. 现有证据速查表（全部来自已存在 artifact，零新增 rollout）

| (source, fallback) | suite | source | fallback | 初步判定 |
|---|---|---|---|---|
| SmolVLA, OFT-spatial | spatial | 67.2% | 92% | fallback 强，仍可测 continue-only？ |
| SmolVLA, π0-fast | spatial | 67.2% | 85% | **候选**（同 env，跨架构） |
| SmolVLA, π0-fast | long(libero_10) | 44.0% | **?（缺失）** | **首选**：若 π0-fast ∈ 40–70% 即为 Goldilocks |
| SmolVLA, OFT-goal/10 | goal/long | 79%/44% | 64.6%/72.9%（96-state） | 已测 FAIL（支配），跳过 |
| π0-fast, OFT-spatial | spatial | 85% | 92% | 备选（source 偏高） |
| π0-fast, OFT-object | object | 89% | 100% | 备选（fallback 几乎处处最优） |
| SmolVLA, OFT-object | object | 90.2% | 100% | source 近饱和，跳过 |
| 任何对 × robot/camera 扰动 | — | 0% | 53–72% | E1 FAIL（96-state 实测），跳过 |

数据来源：SmolVLA clean baseline（`progress/2026-07-16_..._nas10.md`：67.2/90.2/78.6/44.0），
dp_collect_*（OFT/π0-fast 决策点成功率），96-state Goal/Long 分层表。

## 1. 筛选管线（先零 GPU，再最便宜 GPU）

### S0 — 桌面筛查（已在本表完成，零 GPU）
- 冻结上表；预注册 gate；确认缺失数字 = **π0-fast 在 clean libero_10 的成功率**。

### S1 — 廉价探针（GPU ~1–1.5h，全部可并行或顺序）
1. **S1a E1**：π0-fast 在 clean libero_10：10 tasks × 8–10 eps（~30–40 min，
   复用 `rase/collect/forked_rollout.py::load_lerobot_policy_bundle` +
   LIBERO-plus 评估路径）。若 π0-fast long ∈ [30%, 70%] → 首选 pair 成立。
2. **S1b E0**：双生成多样性探针：同一状态（5–10 个）对 SmolVLA 与 π0-fast 各
   生成 2 次 chunk，测 action L2 / gripper disagreement。注意：跨模型仲裁的
   candidate 多样性来自"两个不同 VLA"，确定性 policy 只杀死同模型 resample 臂，
   不是阻塞项——该探针仅用于记录 capability mask。
3. **S1c E2 mini same-root pilot**：首选 pair，4–6 tasks × 4 个同任务 state
   （steps 0,2,4,6）× 2 candidates；修正协议（single native chunk；真实终局
   success；不做 disagreement/outcome 筛选）。复用
   `export_decision_context_keys.py` + `analyze_continue_fallback_opportunity.py`
   的 state 冻结与 task-cluster bootstrap 逻辑。
   - 产出：continue-only / fallback-only / tie 计数、`H_within`、oracle gain +
   CI、支配审计。
   - Gate：`H_within ≥ 5%` 且 oracle gain `≥ 5pp`（sample 允许时看 CI 下界 > 0）
   且 continue-only ≥ 1。

### S2 — 正式 Eligibility（若 S1 通过；GPU ~2–4h）
- 24 tasks × 4–8 个冻结同任务 state，均匀采样；真实终局标签；task-cluster
  bootstrap 10k；分层 + 支配审计。Gate 同 S1。

## 2. 工程约束（修正协议，不可回退）

- 候选执行只用 **single native chunk**（禁止重复冻结 chunk 8 次）。
- recovery/evaluator 从 **branch-end 快照** 起（禁止从 s_t 起）。
- 标签只用真实终局 success；oracle 不得用代理位移（防 Gate C 循环回归）。
- 成本/latency 只进部署效用，不污染训练标签。
- cohort 冻结、均匀，禁按 disagreement/outcome 筛选；robot/camera 扰动子集
  显式分层（E1 窗口测量只用 clean）。
- 优先 **同 env pair**（SmolVLA+π0-fast 都在 smolvla env）；OFT×LeRobot
  跨 env two-pass 仅在同 env 全失败后考虑。

## 3. 决策出口

| 结果 | 动作 |
|---|---|
| S1/S2 PASS | 修正 P3 采集（RGB + branch-end recovery 标签 + π0 候选）→ P1 CRR 用 q_recovery、task-held-out 为主 → P4 视觉 → P5 跨架构 → P6 离线模拟 → P7 smoke → P8 闭环 |
| 首选 pair FAIL | 依序测 spatial 候选对；再测 π0-fast×OFT 备选对 |
| 全部 LIBERO checkpoint 组合 FAIL | (a) 微调 SmolVLA/π0-fast 到 libero_90 子集 30–70% 窗口（guide2 管线）；(b) 任务级粒度 + bandit 退路；(c) 换接触丰富域；WM 仅在修正 B1 低且 oracle headroom >10pp 后重开 |
| 论文定位 | 当前负证据链（三协议 × 多域，H_within=0）已可支撑"eligibility screen 作为方法贡献 + 域边界"叙事；Goldilocks 找到则转正 |

## 4. 立即执行项

1. S1a：π0-fast clean long 成功率（缺失数字，~40 min GPU）。
2. S1b：双生成多样性记录（5–10 states，~10 min）。
3. S1c：首选 pair 的 4–6 task mini same-root pilot（~40 min GPU）。
4. 全程产出 `runs/eligibility_screen_*/` + `progress/2026-08-20_*` 报告；
   git 提交（沿用 2d73b38 的审计纪律）。
