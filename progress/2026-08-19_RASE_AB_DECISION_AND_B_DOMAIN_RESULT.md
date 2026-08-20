# RASE A/B 决策与 B 域实验结果(2026-08-19)

## 一句话结论

> **A(disagreement)与 B(libero_90)均不可行,原因互补**:A 因 π0-fast 确定性推理
> (resample 32/32 bitwise 相同)而无 disagreement 信号;B 因 π0-fast 对 libero_90
> 零样本完全失败(source 成功率 0/7,continue 与 requery 全败)而无 outcome 差异。
> **RASE selector 的适用域边界因此被完整刻画**:需要"source 部分失败且部分成功、
> 候选有可观测差异"的域——LIBERO(fallback 统治)与 libero_90(source 全败)都
> 不满足,当前 policy+benchmark 组合下无可成立的选择器域。

## 1. A 决策:disagreement detector 对 π0-fast 不可行(证据链)

| 证据 | 值 |
|---|---|
| π0-fast config temperature | **0.0(确定性)** |
| resample 候选首动作一致性(K5) | **32/32 bitwise 相同** |
| M=4 proposals 推论 | 同状态同 seed → 全部相同 → disagreement 恒 0 |
| Phase 1 正控 smoke | detector 的 phase/stagnation 在三次阈值校准后仍无稳定工作点 |

**结论**:A 需要一个 temperature>0 的 source policy(改变 source 语义,违反
不变方法定义)或新 policy——当前不可行。

## 2. B 执行:libero_90(contact-rich harder 域)

### 2.1 环境集成(已修复)
- LIBERO-plus 的 `get_task_init_states` 缺少无标记(pruned_init)文件分支 →
  UnboundLocalError;已打补丁(guard 分支),`py_compile` 通过;
- smoke 脚本的 `SyncVectorEnv` 工厂化修复。

### 2.2 Source-only smoke(4 tasks,128 步)
- **π0-fast 在 libero_90 全部失败(3/3,actions_valid=True,stop=max_steps)**;
- 确认域 harder(LIBERO 上 π0-fast continue 45.8%)。

### 2.3 Opportunity smoke(7 tasks × continue/requery × K3)
| verdict | 数量 |
|---|---|
| **all_fail** | **7/7** |
| heterogeneous | 0 |
| both_succeed | 0 |

- continue 与 requery **全部 0/3** 失败——source 完全失败,无 outcome 差异,
  动作信号不可观测;
- 即使 OFT fallback 可用,域也退化为"fallback vs 全败 source"= always-fallback
  最优(S0 结构),selector 无价值;若 OFT 也全败 = 纯 all-fail。

**结论**:B(libero_90)对 π0-fast 过难,opportunity 不成立;需要更强 source
policy(不在手头)或中等难度域。

## 3. RASE 适用域边界的完整刻画(最终)

```text
域类型            LIBERO(step8/16)   libero_90         RASE selector 需要
source 成功率     46%-67%            0%                部分失败(20%-80%)
fallback 成功率   83%-100%(统治)     未知(未适配)       非统治(<90%)
heterogeneous 率  0.9%-2.1%          0%                ≥5%(Gate)
候选可观测差异     fallback 分布      无(source 全败)    存在
```

- **LIBERO**:fallback 统治 → selector 无 headroom(Opportunity Gate FAIL);
- **libero_90**:source 全败 → 无信号(B 域 FAIL);
- **需要的域**:如 π0-fast 微调后的 libero_90、中等难度的 contact-rich 任务、
  或真实机器人(长视界、失败模式多样)。

## 4. 按计划的处理与收尾

- ✅ A 关闭(证据完整,不额外实验);
- ✅ B 域 opportunity FAIL(停止扩大);
- RASE 实验链完整闭合:方法、信号、边界均已量化;
- 若未来获得更强/不同 source policy 或中等难度 benchmark,可复用全部
  基础设施(捕获协议、selector 模块、cost 账本、domain miner、dynamic
  detector、libero_90 补丁)重新过 Opportunity Gate。

## 5. 最终科学资产(全部冻结)

1. 动作反事实信号可学习性:π0-fast 0.871(K5)/0.778(K3 确认),policy-dependent;
2. 可学习 ≠ 可部署:免费 OFT 域 headroom 0.7%(全部场景 FAIL);
3. 事件触发边界:opportunity 存在(正控可复现)但 detector 信号不可用;
4. **适用域边界**:fallback 统治域与 source 全败域均无 selector 空间——
   RASE 需要"中等难度"域,这是方法适用性的精确条件;
5. 工程资产:Phase 0-1 全部代码(动态边界协议、libero_90 集成)可复用。
