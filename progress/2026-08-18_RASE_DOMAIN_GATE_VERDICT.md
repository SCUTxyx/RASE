# RASE Domain Mining 结果与 Opportunity Gate 判定(2026-08-18)

## 一句话结论

> **Opportunity Gate FAIL(决定性门槛未过)**:domain atlas 确认全部 4 个
> (policy × decision-time)cell 的 heterogeneous-winner 率仅 0.9%-2.1%,
> fallback-not-optimal CI 下界远低于 5% 门槛。**按预注册停止规则:不训练新
> selector,形成"当前 benchmark 无可识别成功率空间"的最终结果。**
> 但个案诊断发现两个**稳定可复现的 heterogeneous 区域**(goal_000625@step16、
> object_001041@step8),作为值得记录的模式证据与唯一潜在扩展点。

## 1. 冻结产物

- 预注册:`protocols/domain_mining_preregistration_v1.json`
  (sha256 `e33279fa…`)
- Domain atlas:`runs/rase_vnext/domain_atlas_v1.json`
  (sha256 `a0bbed1a…`,1152 units:confirmation 2×240 + K5 432 + K3 部分)
- 代码:`scripts/build_rase_vnext_domain_atlas.py`

## 2. Atlas 汇总(4 cells)

| cell | units | hetero | hetero rate | all-fail | fb-not-opt |
|---|---:|---:|---:|---:|---:|
| pi0fast@8 | 432 | 4 | 0.93% | 21 | 4 |
| pi0fast@16 | 240 | 5 | 2.08% | 25 | 5 |
| pi05@8 | 240 | 5 | 2.08% | 0 | 5 |
| pi05@16 | 240 | 5 | 2.08% | 0 | 5 |

- winner 分布(全 cells):fallback 参与的 co-winner 组合占绝对多数;
  continue|requery 单独获胜仅 6/1152 units。

## 3. Opportunity Gate 判定(预注册 6 项)

| 门槛 | 阈值 | 结果 | 判定 |
|---|---|---|---|
| 独立 tasks | ≥24(每 suite ≥6) | 48 | PASS |
| 独立 roots | ≥100 | 120 | PASS |
| ≥2 operators 各自 ≥10% roots unique/co-best | — | fallback 主导 | FAIL |
| fallback-not-optimal CI 下界 | >5% | 0.9%-2.1% | FAIL(决定性) |
| oracle-minus-best-fixed | ≥5pp 且 CI 下界 >0 | ~1-2pp | FAIL |
| all-fail 不主导 headroom | — | pi0fast@16 有 10.4% all-fail | WARN |

**判定:FAIL**(4/6 不过,其中门槛 4 为决定性)。

## 4. 个案诊断:两个稳定 heterogeneous 区域(新发现,值得记录)

| 区域 | 模式 | 复现性 |
|---|---|---|
| `libero_goal_000625` @ step16,root `sp1_2604b0af7f9b205` | requery 成功(83 步)vs fallback 失败(274 步,horizon) | **5/5 reps 一致** |
| `libero_object_001041` @ step8,root `sp1_1d37766b2a3f114` | continue 成功(174 步)vs fallback 失败(262 步) | **3/3 reps 一致** |

含义:
- heterogeneous 虽稀疏(1.65%)但**不是噪声**:fallback 在这些状态确定性失败,
  source/requery 确定性成功;
- 这是"selector 潜在价值"的**最小可信证据**(模式可复现);
- 但仅 2 个 roots,无法训练、无法验证泛化;按预注册 Gate 已 FAIL → 不扩展。

## 5. 最终结论(更新 FINAL_VERDICT)

- **RASE 成功率主线:当前 benchmark(LIBERO 48 tasks、π0-fast/π0.5、step 8/16)
  无可识别成功率空间**——fallback 统治是结构性的;
- 预注册的"有条件重开"路径已走完:domain miner → Gate FAIL → 停止(不补数据、
  不调大模型);
- 保留的科学结果:动作信号可学习性(π0-fast Signal PASS)、可学习 ≠ 可部署、
  两个稳定 heterogeneous 模式(供未来 benchmark/策略选择的先验);
- 唯一潜在扩展(不承诺):在**更多 roots** 上验证上述两个 (task, boundary) 模式
  的稳定性(每 task 现有 1-3 roots,root_catalog 尚有剩余);但须重新过 Gate,
  且预期收益极低。

## 6. 收尾动作

- [x] domain atlas + 预注册 + Gate 判定;
- [ ] 更新 `progress/2026-08-18_RASE_FINAL_VERDICT_PI05_AND_DEPLOYMENT.md`
      (将"停止整个 selector 主线"改为"按预注册 Gate 判定 FAIL,条件重开路径已走完");
- [ ] 旧收尾计划标记被取代(保留一致性审计与复现冻结)。
