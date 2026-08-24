# RASE E3-B 执行报告（2026-08-21）

## 结论

E3-B 已按 exact-root、task-disjoint 协议完整执行，但 qualification Gate **FAIL**。残差模型在训练/开发数据上学到了 teacher recovery 动作，在线 B2 独立状态上却没有产生新的成功集合：one-shot 与 source 完全同成败，oracle gain 为 0。

这不是协议或数据泄漏问题，而是当前 residual candidate 尚未打破 SmolVLA 的能力嵌套；因此本轮不进入 verifier、conformal abstention 或闭环 selector 训练。

## 已实现的代码

- 冻结 B0/B1/B2 的 root/task 分区，并做 root、episode、logical-task 不相交审计。
- 新增 OFT teacher calibration；同一状态两次复跑二值结果零漂移。
- 新增 exact-root collector：`source_h8`、`one_shot_h8`、`persistent_h8` 三臂，native H=8，记录 snapshot/teacher chunk/checksum/activation diagnostics。
- 新增 task-conditioned H=8 residual ensemble：3 seeds、root-balanced Huber、identity/correction 分层、teacher-fail capability-gap 排除。
- 新增 B2 qualification summarizer：H_within、oracle gain、candidate-only task 覆盖、task-cluster bootstrap、DAgger 状态泄漏审计。
- 远程回归测试：`19 passed`；远程 reproducibility commit：`df54eed`。

代码副本：[collector](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/work/e3b_remote/scripts/collect_e3b_b0_onpolicy.py)、[residual model](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/work/e3b_remote/rase/recovery/e3b_chunk_residual.py)、[trainer](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/work/e3b_remote/scripts/train_e3b_chunk_residual.py)、[qualification summarizer](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/work/e3b_remote/scripts/summarize_e3b_qualification.py)。

## 实验链

### Teacher calibration

- B0：12 roots，重复两次，10/12 成功（83.3%），Gate PASS。
- Long/Object/Goal 的 teacher-fail roots 被标记为 capability gap，不进入 correction supervision。

### B0 smoke

- 12 roots、36 arm rollouts、3,496 correction steps、845 个 `(8,7)` teacher chunks。
- one-shot 对 source：3 both、1 candidate-only、8 neither；这只作为机制 smoke，不作为资格结论。
- persistent 对 source：3 both、9 neither。

### DAgger B1

- round-0、round-1：各 36 roots × 3 arms，均完整结束。
- 最终训练集：2,439 samples，33/36 roots；3 个 teacher capability-gap roots 排除。
- 开发集 delta-MSE：0.1604 → 0.1154，改善 28.1%；gate TPR 93.7%、FPR 8.7%。
- 这些是离线/训练诊断，不代表 held-out 在线增益。

### B2 qualification（最终裁决）

- 24 独立 roots、8 logical tasks、72 arm rollouts；B0/B1/B2 root 完全不相交，B2 状态未出现在 DAgger 采集目录中。

| candidate arm | both | source-only | candidate-only | neither | H_within | oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| one-shot | 8 | 0 | 0 | 16 | 0.0% | 0.0pp |
| persistent | 6 | 2 | 0 | 16 | 8.3% | 0.0pp |

one-shot 是主资格臂：没有 candidate-only，也没有跨 task 的 rescue。persistent 只引入 source-only 损失，符合 compounding-error 风险。

完整机器可读产物：

- [B2 Gate JSON](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/outputs/RASE_E3B_B2_GATE_2026-08-21.json)
- [final residual training JSON](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/outputs/RASE_E3B_FINAL_RESIDUAL_TRAINING_2026-08-21.json)
- [teacher Gate JSON](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/outputs/RASE_E3B_TEACHER_GATE_2026-08-21.json)
- [B0 Gate JSON](/Users/xueyuxuan/Documents/Codex/2026-08-20/lian/outputs/RASE_E3B_B0_GATE_2026-08-21.json)

## 解释与下一步

当前证据支持：DAgger 修复了 off-policy compounding error 的训练分布问题，但没有制造 residual 与 source 的可分失败域；neither 状态仍占 16/24。按既定 Gate 纪律，下一步应保留 RASE 的 residual idea，但把监督升级到能超越 OFT teacher 的来源（privileged progress / RL 或针对 failure domain 的更强 recovery 数据），并在新的 task-disjoint cohort 复测同一 B2 Gate。只有出现 source-only 与 residual-only 的真实交叉、H_within ≥5%、oracle gain ≥5pp 且覆盖 ≥2 tasks 后，才恢复 verifier/仲裁闭环。
