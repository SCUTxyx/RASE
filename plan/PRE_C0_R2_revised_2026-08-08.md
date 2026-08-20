# PRE-C0-R2（修订版）：Task-Disjoint Gate Scaling + Paired Envelope Evaluation

日期：2026-08-08  
执行环境：`/root/autodl-tmp/RASE`，RTX 5090，`tmux 0`  
主问题：学习式 activation gate 是否比“始终允许纠正、仅依赖 bounded safety envelope”带来额外净收益？

## 1. 服务器当前进展

### Route C / PRE-C0 已完成证据

- spatial B0：27/40 = 67.5%；
- F0 always-on：25/40 = 62.5%；
- legacy bounded F0：32/40 = 80.0%；
- F2 bounded：20/40 = 50.0%；
- cross-suite：spatial F0 方向与 object 方向 cosine=-0.19，解释了 object 上的结构性 harm；
- no-takeover action parity 已修复并通过；
- R1 activation labels：24 snapshots / 12 source episodes / 2 spatial dev tasks；
- label composition：3 rescue、3 harm、17 neutral、1 both-fail；
- snapshot oracle headroom `H_activation=12.5pp`。

`H_activation=12.5pp` 只说明“若知道反事实结果，选择是否激活存在机会”，并不证明当前特征能够预测这个机会。

### 当前仓库状态

- 服务器工作树最新代码基于 commit `8d82126`，存在大量未提交 Route C/PRE-C0 文件；
- GPU 在 R2 启动前为空闲；
- 服务器原先没有 tmux，会话 `0` 已新建；
- GitHub 版本明显落后于服务器，实验结论必须记录 code/artifact identity。

## 2. 原计划不能直接执行的原因

### 2.1 数据泄漏

原 collector 固定读取：

```python
protocol["splits"][suite]["dev"]
```

而 40-episode final eval 也使用相同两个 dev tasks。R1 collector 与部分 eval 还共享 seed 生成公式。因此原计划可能让 gate 在训练时看到最终选择集中的 task/init/seed。

修订：R2 labels 只能来自 spatial `train` tasks；两个 spatial dev tasks 只用于最终 gate/envelope 决策；test tasks 保持锁定。

### 2.2 `val_acc>70%` 是无效 gate

R1 中正样本仅 3/24=12.5%。永远预测“不激活”即可达到 87.5% accuracy。因此不再使用 raw accuracy 作为训练 gate。

主训练指标改为：

- task/episode-grouped out-of-fold average precision；
- balanced accuracy；
- rescue recall；
- harm activation rate；
- utility = activated rescue − 2×activated harm − 0.05×unnecessary activation。

### 2.3 snapshot 随机划分会泄漏

同一 source episode 通常产生 first-deviation 与 recovery 两个 snapshot。随机 snapshot split 会把高度相关状态放入 train/val 两侧。

修订：有 ≥3 tasks 时采用 leave-one-task-out；否则以 `(task, init_state, seed)` 为 group 做 episode folds。

### 2.4 train/eval lean feature 不一致

原 collector 把 `stagnation_len` 设为 `len(history)<=8`，eval 则用持续增长到 16 的 selector history；训练计算 progress delta，eval 固定写 0。

修订：该字段明确改为 normalized context length，train/eval 均截断为 8；eval 使用当前 progress 与上一 history progress 的差。

### 2.5 plugin delta 不一致

collector 的 gate feature 使用固定 `f0_constant_vector_c`，eval 的 gate feature 却来自随 history 变化的 F0 plugin 输出。R0 已证明 F0 网络输出并不恒定。

修订：R2 learned gate 明确控制同一个 constant F0 vector。eval runner 支持“constant delta + learned gate”，避免输入分布改变。

### 2.6 “Safety Envelope”不是独立定义的 arm

原计划同时写了 Gate Bounded 与 Safety Envelope，但代码没有定义两者唯一差异。

修订后新跑两个严格对照：

- `Envelope-only`：使用同一 gate runner 与 safety envelope，但 threshold=0，始终允许；
- `Learned gate + envelope`：唯一变化是使用 checkpoint 的 OOF-selected threshold。

两者的 paired difference 才能归因于 learned activation gate。

### 2.7 40 episodes 无法分辨 2–3pp

40 episodes 的最小成功率步长是 2.5pp。`Gate ≥ bounded +3pp` 实际不可能精确实现：

- +1 episode = +2.5pp；
- +2 episodes = +5.0pp。

因此 dev PASS 定义为 learned gate 相对 envelope-only 至少净增 2 episodes（5pp），且相对 B0 的 harm≤2/40。±1 episode 记为 TIE，而不是统计等价。

## 3. 修订后的数据采集

```bash
python scripts/collect_activation_labels.py \
  --protocol runs/route_c_final/protocol_frozen.json \
  --f0-vector runs/pre_c0_r0/f0_constant_vector.json \
  --output-dir runs/pre_c0_r2 \
  --suite libero_spatial \
  --split train \
  --task-limit 4 \
  --n-episodes-per-task 8 \
  --snapshot-limit 60 \
  --max-snapshots-per-episode 3 \
  --seed 20260808 \
  --dev-high 0.15 --dev-low 0.05 --dev-recover 0.10
```

snapshot 类型除了 privileged OFT deviation/recovery 外，增加 uniform-early/uniform-mid，减少训练只看到 oracle-selected states、部署却查询所有帧的 covariate gap。

Data gate：

- labels≥50；
- rescue≥6；
- harm≥3；
- source tasks≥3；
- source episode groups≥20；
- `source_split=train` 100%；
- feature schema=`rase-activation-gate-lean/v2` 100%。

如果 positive rate 低于 10%，不通过 oversampling 伪造独立样本；应增加独立 episodes/tasks。

## 4. 修订后的 gate 训练

```bash
python scripts/train_activation_gate.py \
  --labels-path runs/pre_c0_r2/activation_labels.jsonl \
  --output-dir runs/pre_c0_r2 \
  --device cuda \
  --epochs 200 \
  --lr 1e-3 \
  --hidden-dim 16 \
  --seed 20260808 \
  --patience 35
```

相对原 hidden=48，hidden=16 更符合 50–60 snapshot 的数据量。初始化 gain 从 0.1 改为 1.0；harm negatives 使用额外权重；早停 epoch 由 grouped OOF folds 得到，最终模型只在 train labels 上重训。

Training gate：

- OOF AP > positive prevalence；
- balanced accuracy≥0.55；
- rescue recall≥0.50；
- harm activation rate≤0.25；
- OOF utility>0；
- 数据数量/覆盖 gate 同时通过。

阈值从 OOF predictions 选择并写入 checkpoint，eval 不再硬编码 0.5。

如果训练 gate 失败，管线停止，不运行昂贵 dev eval。

## 5. 修订后的 spatial dev 实验矩阵

所有新 arm 使用同一个 40-key manifest：

```text
runs/route_c_final/s2_manifest_b3.json
```

已逐键确认该 manifest 与已有 B0、legacy bounded F0 的 `(task, init_state, seed)` 集合完全一致。

| Arm | 是否新跑 | 作用 |
|---|---:|---|
| B0 | 否 | 27/40 baseline |
| F0 always-on | 否 | 25/40；证明无选择持续纠正会 harm |
| Legacy bounded F0 | 否 | 32/40；旧 runner 的工程基线 |
| Envelope-only | 是 | 新 runner，threshold=0；严格安全包络 control |
| Learned gate + envelope | 是 | 主方法；唯一变化为 learned threshold |

主因果比较是 `Learned gate + envelope` vs `Envelope-only`，不是只与 B0 比较。

每个 episode 输出：

- success / steps；
- takeover action steps；
- takeover entries；
- gate queries；
- positive gate decisions；
- checkpoint threshold；
- task/init/seed。

## 6. 决策规则

### PASS

- training gate通过；
- Gate vs Envelope-only 配对净增≥2/40（≥5pp）；
- Gate vs B0 harm≤2/40；
- Gate 的不可逆/异常终止不增加；
- 收益覆盖不止一个 init cluster。

### TIE

- Gate vs Envelope-only 的净差在 ±1 episode；
- harm 不超过 5%。

选择 deterministic envelope，因为结构更简单、数据依赖更低。TIE 不能写成 learned gate 有效。

### FAIL

- grouped-CV training gate失败；或
- Gate 比 Envelope-only 少≥2 successes；或
- Gate vs B0 harm>5%；或
- gate 几乎 always-on/always-off；或
- 只在训练 task/episode 有效。

回归 deterministic bounded/envelope，并停止扩大 MLP gate；下一步应改 failure progress signal 或 recovery operator。

## 7. 统计报告

必须使用 key-based join，禁止按 JSONL 行号直接配对。输出：

- 每 arm 成功率；
- rescue/harm/net；
-完整 `n00/n01/n10/n11`；
- exact McNemar p；
- gate activation/takeover burden；
- task/init cluster coverage。

40 episodes 是 dev decision，不足以证明 2pp clean non-inferiority。若 R2 PASS，再设计 locked task evaluation 和多 training seeds。

## 8. 已实现文件

```text
scripts/collect_activation_labels.py
scripts/train_activation_gate.py
scripts/eval_route_c_paired.py
scripts/summarize_pre_c0_r2.py
scripts/run_pre_c0_r2.sh
```

原三个脚本在服务器保留可恢复备份：

```text
scripts/*.20260808_pre_c0_r2.bak
```

## 9. 正在执行

```bash
tmux attach -t 0
```

tmux 中运行：

```bash
bash scripts/run_pre_c0_r2.sh
```

主输出：

```text
runs/pre_c0_r2/pipeline.log
runs/pre_c0_r2/activation_labels.jsonl
runs/pre_c0_r2/gate_training_report.json
runs/pre_c0_r2/gate_checkpoint.pt
runs/pre_c0_r2/paired_results_gate.jsonl
runs/pre_c0_r2/paired_results_envelope_only.jsonl
runs/pre_c0_r2/summary.json
runs/pre_c0_r2/summary.md
```

## 10. 研究定位

即使 R2 PASS，learned binary gate 本身仍不足以形成顶会方法贡献。可以支持的下一层问题是：

> 在同一 failure state 上存在同时可能 rescue 和 harm 的低成本纠正算子时，是否可以用 deployment-observable state history 预测 intervention utility，并通过安全包络获得正净收益？

若 learned gate 只与 deterministic envelope 打平，应诚实把结果写成“简单时序安全规则优于小数据 learned gate”，这同样是有价值的负结果；不要继续通过扩大 hidden dimension 或查看 dev 后调 threshold 强行制造正结果。
