# RASE 后续研究规划（2026-08-21：R2 正面结果路线）

> 上游：九环排除链确认 clean 确定性域无仲裁空间（见 PROGRESS §3）；BCP
> （arXiv:2608.03483）证明"学习自适应 horizon"在扰动域有增益（LIBERO-PRO
> +6.8%）。**本规划把 BCP 吸收为基线组件，主攻其未覆盖的差异化层：
> 跨策略反事实仲裁（switch）、same-root 校准、统计保证、zero-shot 跨 VLA。**

---

## 0. 战略定位（一句话）

> 在**扰动域**（LIBERO-PRO / RoboTwin Randomized / 真机）上实现
> **Adaptive Cross-Policy Counterfactual Arbitration**：以 BCP 式自适应
> continue/replan 为基线（一维特例），加入跨策略 switch 臂 + same-root
> 反事实校准 + 保形统计保证，相对 BCP 与固定策略取得可证明增益。

## 1. 优先级与 Gate 体系

### R2a：LIBERO-PRO 实现 + π0.5 基线（1-2 天，GPU ~2h）

- **实现**：LIBERO Object suite 物体位置扰动（position perturbations，
  多幅度，AAC 评估协议）——LIBERO clean + init-state 物体位姿偏移脚本
  （参照 BCP 论文 §LIBERO-PRO 描述）；
- **基线**：π0.5（`ckpts/pi05_libero`）固定 horizon（10 步）在扰动集上
  测量 → 目标复现 ~30.9% 量级（BCP 报告值）；
- 同时测量 SmolVLA（44% long 窗口）作为备选 source；
- **Gate R2a**：扰动集成功率 ∈ 10%–90%（E1）且候选结局分叉
  （E0'：oracle@K > best-of-1，K=8 采样，≥2 rescue 状态）。

### R2b：自适应仲裁头（BCP 复现 + 监督版先行）（3-5 天，GPU ~1 天）

- **头架构**：冻结 π0.5 + 轻量头（输入：VLA hidden/动作特征 + proprio；
  输出：伯努利 continue-or-replan 序列，前缀共享归纳偏置——BCP 式）；
- **训练信号两版**：
  - 监督版（我们的先发优势）：same-root 反事实标签（同一状态 K 候选 →
    真实结局）构造"该 chunk 该继续/该重规划"的密集标签；
  - RL 版（BCP 式）：轨迹级结局 + Replanning-Efficiency Reward（防塌缩
    到过短 horizon）；
- **对照**：固定 k=10 / 固定 k=4 / 监督版 / RL 版；
- **Gate R2b**：扰动域上自适应 ≥ 固定 k=10 且 ≥3pp（复现 BCP 量级）；
  clean 域上自适应 ≈ 固定（无增益是预期的，作为对照）。

### R2c：跨策略 switch 臂 + same-root 校准 + 保形保证（3-5 天，GPU ~1 天）

- **switch 臂**：仲裁头决策空间扩展为 {continue, replan, switch}——
  switch = 切到纠正候选（π0-fast / OFT / SmolVLA 取决于 source）；
  switch 的触发条件用 same-root 反事实标签校准（"该状态 switch 相对
  continue/replan 的 ΔV"）；
- **保形校准**：false-replan / false-switch 的保形上界（决策级，
  [Conformal Decision Theory](https://arxiv.org/abs/2310.05921) 思路）；
- **对照**：BCP 复现（无 switch）/ +switch 无校准 / 完整版；
- **Gate R2c**：完整版相对 BCP 复现 ≥3pp，且保形上界成立（覆盖率达标）。

### R2d：zero-shot 跨 VLA（2-3 天）

- 训练仲裁头时不用 policy identity（canonical 特征）→ 对未见 VLA
  （如训练用 π0.5，测试用 SmolVLA/π0-fast 候选）直接打分；
- **Gate R2d**：未见 VLA 上仲裁增益保留 ≥50%（retention 度量，对齐 LOVO 惯例）。

### R3：扩展验证（后续）

- RoboTwin 2.0 Randomized 设置（需获取平台，工程 ~1 周）；
- 真机（AGIBOT 类，成本高，后置）；
- 论文整理可与 R2b 并行启动。

---

## 2. 时间线与预算

| 周 | 任务 | GPU | Gate |
|---|---|---|---|
| 第 1 周 | R2a（LIBERO-PRO + 基线）+ R2b 头实现 | ~3h/天 | R2a、R2b |
| 第 2 周 | R2b 训练 + 对照 | ~3h/天 | R2b ≥3pp |
| 第 3 周 | R2c（switch + 校准 + 保形） | ~3h/天 | R2c ≥3pp |
| 第 4 周 | R2d（zero-shot）+ 论文骨架 | 低 | R2d |
| 并行 | 论文轨（九环证据链 + 方法论） | 零 | — |

总 GPU 预算：~2-3 周 × 3h/天 ≈ 40-60 GPU 小时（RTX 5090 单卡）。

## 3. 停止纪律（写死）

1. R2a 失败（扰动集无 E1/E0' 窗口）→ 不进入 R2b；换扰动幅度/换 source；
2. R2b 失败（监督版与 RL 版均无法复现 BCP 量级）→ 论文轨独立成稿
   （九环证据链 + BCP 对比的负面贡献）；
3. R2c 失败（switch 臂无增益）→ 收缩 claim 为"自适应 horizon + 校准"
   （BCP+ 定位），不硬推跨策略；
4. 不在 oracle gain=0 的域训练仲裁头（E0' 前置）；
5. 所有 run 落盘 + git 快照 + task-disjoint 评估（沿用既有纪律）。

## 4. 关键风险

| 风险 | 缓解 |
|---|---|
| LIBERO-PRO 实现与 BCP 协议不完全一致 | 按 AAC 协议描述实现 + 多幅度扫描；论文中如实说明 |
| π0.5 推理速度（7.7GB 模型 ~2-3s/次） | 控制评估规模（扰动集 10 tasks × 8 eps × 3 幅度） |
| 监督版 same-root 标签在扰动域的信息量不足 | 先做 E0' 诊断（候选结局分叉）再决定监督 vs RL |
| BCP 代码未开源 | 自主实现（方法细节已从论文提取） |
| 跨策略 switch 在扰动域仍嵌套 | R2c gate 前先做同状态 switch vs continue 的 oracle 检查（E0' 扩展） |

## 5. 立即执行清单（R2a）

```text
1. 写 LIBERO-PRO 扰动实现（物体位姿偏移，幅度 {0.02, 0.05, 0.1} 量级扫描）
2. π0.5 固定 horizon 基线评估（10 tasks × 8 eps × 3 幅度）
3. E0' 候选池诊断（K=8 采样，oracle@K vs best-of-1，≥2 rescue 状态）
4. 产出 progress/2026-08-2x_r2a_lib_pro_report.md + git 提交
```

---

## 6. 论文主线（与 R2 并行推进骨架）

> **Adaptive Cross-Policy Counterfactual Arbitration**
> 贡献 1：same-root 反事实数据协议（文献空白）；
> 贡献 2：统一仲裁 head（自适应 horizon + 跨策略臂，BCP 严格超集）；
> 贡献 3：反事实校准 + 保形统计保证；
> 贡献 4：zero-shot 跨 VLA；
> 贡献 5（负向）："何时仲裁不可能"的系统刻画（九环证据链 + 三定理锚点）。
