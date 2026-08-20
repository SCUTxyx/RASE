# RASE v3 顶会路线 — 服务器执行指令(无卡模式期间准备)

> 服务器恢复后按本文执行。所有脚本已就绪:
> 本地 `/Users/xueyuxuan/RASE_analysis/`(上传到服务器 `RASE/scripts/`)。
> 规划依据:`RASE_ROADMAP_2026_08_19.md`(v3)。
> 纪律:每个 Phase 先跑 Gate,FAIL 即止损,不调旧结果。

---

## 0. 准备(一次性)

```bash
# 服务器上:
cd /root/autodl-tmp/RASE
# 上传本地 RASE_analysis/ 下新增脚本到 scripts/(rase_common.py 也要)
# 确认两个 env:
source /root/miniconda3/etc/profile.d/conda.sh
conda activate oft      # OpenVLA/OFT 栈 + numpy/torch
conda activate smolvla  # lerobot(π0-fast/smolvla 推理)

# 确认冻结资产存在(禁止重训):
ls runs/oft_opportunity/oft_risk_model_v3.npz   # 冻结风险模型
ls runs/oft_opportunity/oft_risk_vocab.json
ls runs/oft_opportunity/dp_collect_spatial.jsonl dp_collect_object.jsonl
```

---

## P0 — Phase 1:Current OPD Zero-Shot Falsification(Gate A)

### 0a. 采集未见 VLA 决策点数据

```bash
# oft 域:oft_goal + oft_10(同架构 checkpoint shift,Level 1)
conda activate oft
export LIBERO_ROOT=/root/autodl-tmp/src/LIBERO-plus LIBERO_BENCHMARK_ROOT=/root/autodl-tmp/src/LIBERO-plus
export HF_HOME=/root/autodl-tmp/hf_cache TOKENIZERS_PARALLELISM=false
python scripts/collect_zs_vla.py --vla-type openvla --model oft_goal \
  --matrix runs/oft_opportunity/oft_matrix_analysis.json \
  --output runs/oft_opportunity/dp_collect_goal.jsonl --episodes 3
python scripts/collect_zs_vla.py --vla-type openvla --model oft_10 \
  --matrix runs/oft_opportunity/oft_matrix_analysis.json \
  --output runs/oft_opportunity/dp_collect_10.jsonl --episodes 3

# π0-fast(architecture shift,Level 2)→ smolvla env
conda activate smolvla
python scripts/collect_zs_vla.py --vla-type lerobot \
  --policy-path /root/autodl-tmp/RASE/ckpts/pi0fast_libero \
  --model pi0fast \
  --matrix runs/oft_opportunity/oft_matrix_analysis.json \
  --output runs/oft_opportunity/dp_collect_pi0fast.jsonl --episodes 3
# 注意:collect_zs_vla.py 中 LeRobot proprio 提取有 TODO(服务器验证),
# 若失败会写 0 占位——验证通过后再用于打分。
```

### 0b. Zero-shot 打分分析(Gate A)

```bash
conda activate oft
python scripts/analyze_zs_vla.py \
  --risk-model runs/oft_opportunity/oft_risk_model_v3.npz \
  --vocab runs/oft_opportunity/oft_risk_vocab.json \
  --train-a runs/oft_opportunity/dp_collect_spatial.jsonl \
  --train-b runs/oft_opportunity/dp_collect_object.jsonl \
  --test-vlas goal=runs/oft_opportunity/dp_collect_goal.jsonl \
                 oft10=runs/oft_opportunity/dp_collect_10.jsonl \
                 pi0fast=runs/oft_opportunity/dp_collect_pi0fast.jsonl \
  --output runs/oft_opportunity/zero_shot_vla_analysis.json
```

**Gate A 判定**(脚本输出 verdict):
- PASS → 继续 P1(同时保留 direct OPD 作 baseline)
- FAIL(AUROC≈0.5 + VLA-ID probe 高)→ **论文 motivation 成立**:
  "direct risk exploits policy-specific shortcuts and fails under policy shift";
  直接进 P1,OPD-v3 不再做任何部署调优。
- 产出 `ZERO_SHOT_VLA_ANALYSIS.md`。

---

## P1 — Phase 2 + 2.5:Same-root 反事实采集 + 分叉分析(Gate B)

```bash
conda activate oft
# 小规模先行:8 任务 × 2 ep × 4 决策点 × 3 候选(A/B/goal)
python scripts/collect_same_root.py \
  --models oft_spatial,oft_object,oft_goal \
  --matrix runs/oft_opportunity/oft_matrix_analysis.json \
  --output runs/oft_opportunity/same_root_v1.jsonl \
  --episodes 2 --horizon 8 --label-mode progress \
  --tasks 8 --decisions-per-episode 4

python scripts/analyze_future_divergence.py \
  --data runs/oft_opportunity/same_root_v1.jsonl \
  --output runs/oft_opportunity/future_divergence_report.json
```

**Gate B**:`verdict=PASS` 要求 candidate diversity 非退化 + future divergence 分叉
(≥30% roots)+ within-state advantage(≥10%)。
- PASS → 扩大采集(20 任务 × 3 ep × 全决策点)后进 P2
- FAIL → 换 candidate provider / decision boundary,不训练 WM。

---

## P2 — Phase 3:Oracle Future Risk(Gate C)

```bash
python scripts/analyze_oracle_future.py \
  --data runs/oft_opportunity/same_root_v1.jsonl \
  --output runs/oft_opportunity/oracle_future_report.json
```

**Gate C**:`verdict=PASS` 要求 pairwise consistency ≥0.6 且 discrimination
AUROC >0.6。
- PASS → P3(WM MVP)
- FAIL → 重定义 risk target / candidate set,**不训练大 WM**。

---

## P3 — Phase 4:World Model MVP(Gate D,核心 milestone)

```bash
python scripts/train_wm_mvp.py \
  --data runs/oft_opportunity/same_root_v1.jsonl \
  --output runs/oft_opportunity/wm_mvp_report.json \
  --lovo-test oft_goal
# 轮换 LOVO:--lovo-test oft_object / oft_spatial(数据量够时)
```

输出 B0/B1/B2 的:iid AUROC、LOVO AUROC、TransferRetention、identity probe。

**Gate D / 核心 Milestone**:
> B2(future-bottleneck)必须在 **held-out VLA** 上 B2 > B1(direct),且
> transfer retention 更高;仅 IID 更好不算。

- PASS → P4/P5(顶会主线成立)
- FAIL → Kill 4:不把 WM 作为主论文;改 direct-risk + 在线适配定位。

---

## P4 — Phase 5/6(可选,需 P3 PASS)

- latent visual WM(需重采集 RGB 或 visual embedding;冻结 encoder)
- micro-deviation 实验:动作加 ε 扰动 → future divergence 随 horizon 放大 →
  lead time 分析(脚本待写,框架在 roadmap §13)。

---

## P5 — Phase 8:Zero-shot closed-loop selector

```bash
# 用 P3 胜出的 B2 模型(npz,带 feature_version)跑闭环
python scripts/rase_selector_loop.py \
  --selector runs/oft_opportunity/wm_mvp_B2.npz \
  --vocab runs/oft_opportunity/oft_risk_vocab.json \
  --matrix runs/oft_opportunity/oft_matrix_analysis.json \
  --output runs/oft_opportunity/selector_loop.json \
  --num-trials-per-task 4 --beta 1.0 --delta 0.05 --dwell 2
```

Baselines 对比(分析脚本 `analyze_closed_loop.py` 待写或手工汇总):
`each-VLA-only / best-fixed / random router / task-router / Direct OPD /
Predictive Risk Selector / oracle`。

**Core Gate 3**:held-out VLA/task 上 Success(RASE) > Success(best-fixed),
paired bootstrap 95% CI 下界 > 0,绝对增益 ≥ 3pp。
**Core Gate 4**:RASE > task-router,或证明收益来自 within-task changes。

---

## P6 — 跨平台 + OPD 蒸馏 + bandit(核心 PASS 后)

- 第二平台:ManiSkill3 兼容性 probe(obs preprocessing/action mapping/
  state restore/≥2 candidate)——Kill 7 保护:工程量过高则先投 cross-VLA paper
- OPD 蒸馏(§23):Teacher = Risk(WM(s,a)) → student ridge;adaptive imagination
- bandit(§24):`bandit_adapter.py` 已实现(OPD prior 初始化 + Thompson residual),
  报告 episode 0 zero-shot 与 adaptation 分开

---

## 输出文档清单

| Phase | 产出 |
|---|---|
| P0 | `ZERO_SHOT_VLA_ANALYSIS.md` |
| P1 | `COUNTERFACTUAL_DATASET_V2_ANALYSIS.md`(含 divergence report) |
| P2 | `ORACLE_FUTURE_RISK_ANALYSIS.md` |
| P3 | `WM_MVP_TRANSFER_ANALYSIS.md` |
| P5 | `RASE_ZERO_SHOT_SELECTOR_RESULTS.md` |
| P6 | `RASE_CROSS_PLATFORM_RESULTS.md` / `RASE_ONLINE_ADAPTATION_RESULTS.md` |

每个 Phase 结束:更新 `RASE_EXPERIMENT_PROGRESS.md` + `RASE_PROBLEMS.md`,
并做 identity probe / task-router 对照 / paired bootstrap 三项诚实性检查。

---

## 本地已就绪脚本(全部通过语法检查,纯逻辑部分已本地单测)

```
rase_common.py              # canonical 特征/metrics/probe/ridge(测试全过)
analyze_zs_vla.py           # Phase 1 zero-shot 分析
collect_zs_vla.py           # Phase 1 采集(openvla+lerobot wrapper)
collect_same_root.py        # Phase 2 same-root 采集(future 轨迹保存)
analyze_future_divergence.py# Phase 2.5 Gate B(测试全过)
analyze_oracle_future.py    # Phase 3 Gate C(测试全过)
train_wm_mvp.py             # Phase 4 B0/B1/B2 + LOVO(测试全过)
selector_risk.py            # Phase 8 LCB/UCB/hysteresis(测试全过)
rase_selector_loop.py       # Phase 8 闭环(服务器集成,语法 OK)
bandit_adapter.py           # Phase 6 OPD+bandit(测试全过)
train_opd_v2.py             # 备用:OPD-v2 ablation(合成测试全过)
```
