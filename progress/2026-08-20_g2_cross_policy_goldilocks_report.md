# RASE G2 Cross-Policy Goldilocks 执行报告

日期：2026-08-20

## 正式结论

G2 已完整执行，Long 与 Spatial 两个分支都没有打开 relative-risk verifier 的训练 Gate。

- G2a：π0-fast 在官方 clean LIBERO-10 上为 **69/80 = 86.25%**，高于预注册的 70% 上界，裁决为 `PI0FAST_DOMINATES_TRY_SPATIAL_PAIR`。
- G2b Spatial16：SmolVLA strict continue 为 **5/16 = 31.25%**，π0-fast direct fallback 为 **11/16 = 68.75%**；fallback-only 6、both-success 5、neither 5、continue-only 0。
- Spatial 的 state oracle 与 task-best-fixed 都是 68.75%，oracle gain 0pp；`H_within=0%`，task bootstrap 95% CI `[0%, 0%]`。
- 因此 π0-fast 在该 same-root cohort 上逐状态弱支配 SmolVLA，G2b 正式 FAIL。

这不改变 RASE idea。它裁决的是候选资格：当前冻结候选集合缺少双向反事实优势，风险模型没有可利用的选择空间。按预注册纪律，不训练 verifier、不调 margin、不跑闭环。

## 对原规划的修订

原 G2a 的方向正确，但做了四项必要修订：

1. 将模糊的“10 tasks × 8–10 eps”冻结为官方 clean LIBERO-10 的 10 tasks × init-state 0–7，共 80 条。
2. 在结果出现前冻结 environment/policy seeds、checkpoint identity、协议哈希和 30%–70% 点估计 Gate。
3. 除总体成功率外，报告 Wilson 95% CI、task-cluster bootstrap、每任务成功率、真实 env steps、model forwards 与推理时间。
4. G2b 不把两个 arm 宣称为 compute-matched；primary label 仅为同 snapshot 的真实终局 success，成本单独报告。

G2a 的历史 R7 数据只覆盖 4 个 clean Long tasks × 4 init states，且该子集偏强，不能替代完整十任务测量。本轮补齐全部官方任务后，结论仍是 π0-fast 明显过强。

## G2a 协议与结果

- Protocol schema：`rase-g2a-pi0fast-clean-direct-protocol/v1`
- Protocol SHA256：`663d8847c86291a0c7eefcb8a0a4e848cb5256cf49533e622358c9b597a90a43`
- Cohort：10 official clean Long tasks × 8 frozen init states = 80 episodes
- Policy：`ckpts/pi0fast_libero`，10-step native action chunks
- 结果：69/80 = 86.25%
- Wilson 95% CI：77.03%–92.15%
- Task-cluster bootstrap 95% CI：76.25%–95.00%
- Policy errors：0
- 总 env steps：24,019
- Model forwards：2,432

每任务成功数：

| Task | Success |
|---|---:|
| 1 | 8/8 |
| 2 | 5/8 |
| 3 | 7/8 |
| 4 | 7/8 |
| 5 | 7/8 |
| 6 | 7/8 |
| 7 | 8/8 |
| 8 | 8/8 |
| 9 | 4/8 |
| 10 | 8/8 |

任务 9 只有 50%，说明不是每个任务绝对支配；但总体率与区间都明确排除 Goldilocks Long pair。

## G2b Spatial16 协议与结果

- Cohort：phase0g 已冻结池中的 4 clean Spatial tasks × steps 0/2/4/6 = 16 states。
- 选择只使用 task/step/clean metadata，0 rejected，不读取 outcomes。
- Continue：保存的 SmolVLA active suffix，然后冻结 SmolVLA 继续到真实终局。
- Fallback：恢复同一 snapshot，空 prefix，π0-fast direct takeover 到真实终局。
- Gate：`H_within ≥ 5%` 且 oracle gain ≥ 5pp；还必须存在双向独占成功。

| Metric | Result |
|---|---:|
| Smol strict continue | 5/16 = 31.25% |
| π0-fast direct | 11/16 = 68.75% |
| Continue-only | 0 |
| Fallback-only | 6 |
| Both-success | 5 |
| Neither | 5 |
| State oracle | 68.75% |
| Task-best-fixed | 68.75% |
| Oracle gain | 0pp |
| H_within | 0% |

四任务中，tasks 3/6/7 均为 Smol 1/4、π0-fast 3/4，且各有两个 fallback-only；task 8 为两者都 2/4、四状态全 tie。没有任何 continue-only 状态。

## 代码完善

- `scripts/freeze_g2a_protocol.py`：冻结完整 80-episode outcome-blind 协议与哈希。
- `scripts/eval_g2a_pi0fast_clean.py`：resume-safe direct eval、逐 episode 落盘、Wilson/task bootstrap、成本和 hash 审计。
- `scripts/run_g2a_pi0fast_clean_long_parallel.sh`：两路错峰加载，避免多进程同时读取 7.7GB checkpoint 造成 I/O 拥塞。
- `scripts/rollout_lerobot_direct_from_pool.py`：从任意冻结 pool state 执行 generic LeRobot direct takeover。
- `scripts/analyze_continue_fallback_opportunity.py`：支持 OFT 与 generic LeRobot fallback，label provenance 改为通用冻结 fallback policy。
- `scripts/run_g2b_spatial16.sh`：keys freeze → strict continue → π0-fast direct → task-bootstrap Gate 的可恢复管线。
- 新增 G2a/G2b 单元测试；相关回归 **19 passed**。

Git snapshots：

- `c7d3964`：结果前冻结 G2a 协议与 evaluator。
- `a913820`：resume-safe sharding。
- `d020837`：结果前冻结 cross-policy Spatial eligibility 代码。

## 下一阶段：保持 RASE idea，改变候选构造

自然冻结 pair 的筛查已经给出一致结论：π0-fast 是更强通用执行器，而不是具有互补失败集合的 fallback。下一阶段不应继续换 selector，而应先构造专门化候选：

1. **Failure-specialized residual candidate**：冻结基础 VLA，只训练轻量 residual/proposal head；训练数据仅来自 source-failure same-root states，目标是产生与 source 不同的纠正动作，不替换基础 policy。
2. **严格任务切分**：训练、development eligibility、validation/test 按 task 隔离；当前 G0/G1/G2 states 只能用于 development/negative controls。
3. **新的 E3 Eligibility**：至少 24 tasks × 4–8 states；必须同时有 continue-only 与 residual-only，`H_within ≥5%`、oracle gain ≥5pp，且增益跨至少 2 个 tasks。
4. **保留 RASE 主线**：只有 E3 PASS 后，才训练 same-root relative verifier，做 action-shuffled/label-permuted/state-only 对照，再进入 conformal abstention 与 risk-guided regeneration 闭环。
5. **任务级 routing 仅作基线**：可以报告 task-level provider selection，但不能替代 RASE 的 state-level 仲裁主张。

如果 E3 仍失败，应把 G0/G1/G2 组织成候选资格与策略支配的系统性负面结果，而不是继续扩大模型或调阈值。

## Artifact 哈希

```text
e128ab7fcea0b4938d1c990337a1ff9b1378c763e5d6d3b8605559ebaf1c3bab  configs/g2a_pi0fast_clean_long_v1.json
0e1cbf14504af18b778922e2dcc442e4fae62e4465233b5087c20bb56b876ba9  runs/oft_opportunity/g2a_pi0fast_clean_long_v1/summary.json
30db86c0beb815a7bfdf5e6ff3f9357c17eff63aaa27f710271db40ab56c6169  runs/g2b_spatial16_keys_v1.json
2531f2236c59dde21d6daf989b48d79639949d54591d224d05216de0f1ec2e9c  runs/g2b_spatial16_smol_continue_v1/summary.json
94a6b8f8d350a0d833c16ce51d23cd45073467edd809a4a473a396ad6c8f817f  runs/g2b_spatial16_pi0fast_direct_v1/summary.json
b8f8c4564e7b637116ea08963b6a527a6769976c250246a306680b530f79e66c  runs/g2b_spatial16_opportunity_v1.json
```

最终服务器状态：GPU 0 MiB / 0%，无实验进程；原有未跟踪调研文档未修改。
