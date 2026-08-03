# RASE 本机完全复刻指南（源机：AutoDL `/root/autodl-tmp`，2026-08-03）

目标：在新服务器上恢复与本机一致的 **RASE 科研工作台**——代码、配置、progress、环境、checkpoint、pool/runs 产物，以及 LIBERO-plus / OpenVLA-OFT 依赖栈。

配套文件：

- `env.lock.md` — 环境锁定（含本快照实测修正）
- `envs/smolvla.yaml` / `envs/oft.yaml` — conda 规格
- `envs/smolvla.pip.freeze.txt` / `envs/oft.pip.freeze.txt` — 完整 freeze
- `third_party/PINS.md` — 上游 pin
- `reports/run_summaries/` — 从本地 `runs/**/summary.json` 抽出的可进 Git 摘要（~2MB）
- `reports/local_artifact_manifest.csv` — `ckpts/` / `pool/` 体积清单
- `REPRODUCIBILITY.md` — 精简版；**以本文件为迁机主文档**

---

## 0. 源机盘点

### 0.1 仓库体量（源机）

| 路径 | ~体积 | Git？ |
|---|---:|---|
| 代码+配置+progress+tests | <20M | **是** |
| `reports/run_summaries/` | ~2–5M | **是**（摘要镜像） |
| `ckpts/` | **68G** | 否 |
| `pool/` | **12G** | 否 |
| `runs/` | **3.0G** | 否（摘要进 reports） |
| 旁路：`../envs/smolvla` `../envs/oft` | 11G×2 | 否 |
| 旁路：`../src/LIBERO-plus` | ~9.7G | 否（无 `.git`） |
| 旁路：`../src/openvla-oft` | 2.7M | 可 git |
| 旁路：`../src/transformers-openvla-oft` | 88M | 否（无 `.git`） |

### 0.2 Checkpoints（本地约定目录）

| 本地目录 | ~体积 | 来源 |
|---|---:|---|
| `ckpts/smolvla_libero` | 1.2G | `HuggingFaceVLA/smolvla_libero` |
| `ckpts/SmolVLM2-500M-Instruct` | 1.9G | `HuggingFaceTB/SmolVLM2-500M-Instruct` |
| `ckpts/oft_spatial` | 15G | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| `ckpts/oft_object` | 20G | `moojink/openvla-7b-oft-finetuned-libero-object` |
| `ckpts/oft_goal` | 15G | `moojink/openvla-7b-oft-finetuned-libero-goal` |
| `ckpts/oft_10` | 15G | `moojink/openvla-7b-oft-finetuned-libero-10` |

### 0.3 Pool（继续实验通常需要）

| 目录 | ~体积 |
|---|---:|
| `pool/ngc_step1_scale200` | 3.8G |
| `pool/ngc_w9b_clean_controls` | 2.2G |
| `pool/ngc_w9_clean_controls` | 2.1G |
| `pool/ngc_w9c_clean_controls` | 1.6G |
| `pool/ngc_w10_object_spatial_failures` | 1.2G |
| `pool/ngc_w5_l1_l2_camera_robot` | 773M |
| 其余 pilot/preflight | <0.5G |

### 0.4 当前科研结论落点（已在 `progress/`）

- W9C：`kill_method_branch`（ridge 未过门）— `progress/2026-07-31_w9c_selector_gate_result.md`
- W10：Object/Spatial failure 覆盖；held-out `NOT_READY` —
  `progress/2026-07-31_w10_object_spatial_benchmark.md`
- RASE-UI phase0→phase1a 系列：`progress/2026-08-01_*` / `2026-08-02_*`
- 总叙事：`progress/2026-07-31_rase_full_project_narrative.md`

数值细节以 `progress/*.md` + `reports/run_summaries/**/summary.json` 为准；
完整 pool/rollout 二进制在 `pool/`、`runs/`。

---

## 1. 推荐目标布局

```text
$STACK=/data/rase-stack
$STACK/
  RASE/                      # git clone
  CareVLA/                   # 若并行迁 CareVLA
  src/
    LIBERO-plus/             # editable libero（smolvla+oft 共用）
    LIBERO/                  # 可选；源机与 plus assets 近似重复
    openvla-oft/             # @ e4287e9
    transformers-openvla-oft/
  envs/
    smolvla/
    oft/
  hf_cache/
```

源机历史绝对路径曾出现 `/data/data2/yuxuan/...`；AutoDL 实际为
`/root/autodl-tmp/...`。新机统一用 `$STACK`，并改所有 config / `~/.libero*`。

---

## 2. Git：拉代码

```bash
export STACK=/data/rase-stack
mkdir -p "$STACK"
cd "$STACK"
git clone https://github.com/SCUTxyx/RASE.git
cd RASE
git log -1 --oneline
```

若远程尚未包含本机最新提交，用已打好的 bundle：

```bash
# 源机已生成：/root/autodl-tmp/RASE.bundle
# 新机
git clone RASE.bundle RASE
cd RASE
git remote set-url origin https://github.com/SCUTxyx/RASE.git
```

仓库：`https://github.com/SCUTxyx/RASE.git`

---

## 3. 大文件 rsync（核心）

把 `SRC` 换成源机。建议在 tmux 里跑。

### 3.1 权重（最大，优先）

```bash
mkdir -p "$STACK/RASE/ckpts"
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/RASE/ckpts/ \
  "$STACK/RASE/ckpts/"
```

只跑 Smol 线时可先拉：

```bash
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/RASE/ckpts/smolvla_libero \
  SRC:/root/autodl-tmp/RASE/ckpts/SmolVLM2-500M-Instruct \
  "$STACK/RASE/ckpts/"
```

HF 重下备选：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$STACK/hf_cache"
huggingface-cli download HuggingFaceVLA/smolvla_libero --local-dir ckpts/smolvla_libero
huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Instruct --local-dir ckpts/SmolVLM2-500M-Instruct
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial --local-dir ckpts/oft_spatial
# object / goal / 10 同理
```

### 3.2 Pool + Runs

```bash
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/RASE/pool/ "$STACK/RASE/pool/"
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/RASE/runs/ "$STACK/RASE/runs/"
```

若磁盘紧：可先不拷完整 `runs/*_pool`（数百 MB–1GB 级），保留
`reports/run_summaries` + `progress/` 做结论复盘；但**继续训练/续跑必须有对应 pool**。

### 3.3 第三方源码

```bash
mkdir -p "$STACK/src"
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/src/LIBERO-plus \
  SRC:/root/autodl-tmp/src/openvla-oft \
  SRC:/root/autodl-tmp/src/transformers-openvla-oft \
  "$STACK/src/"
# 可选
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/src/LIBERO "$STACK/src/"
```

注意：源机 `LIBERO-plus` / `LIBERO` / `transformers-openvla-oft` **不是 git checkout**。
文档里的 `4976dc3` 是设计 pin；字节级复现请整树拷贝。

`openvla-oft` 可改为：

```bash
git clone https://github.com/moojink/openvla-oft "$STACK/src/openvla-oft"
git -C "$STACK/src/openvla-oft" checkout e4287e94541f459edc4feabc4e181f537cd569a8
```

### 3.4 Conda 环境（可选整拷）

```bash
mkdir -p "$STACK/envs"
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/envs/smolvla "$STACK/envs/"
rsync -aH --info=progress2 \
  SRC:/root/autodl-tmp/envs/oft "$STACK/envs/"
```

整拷后必须重装 editable（§4.3）。

---

## 4. 环境配置

RASE 使用**两套互斥重环境**：

| Env | Python | 用途 |
|---|---|---|
| `smolvla` / `rase-smolvla` | 3.12.13 | SmolVLA、LIBERO-plus collect、多数 NGC/W* |
| `oft` / `rase-oft` | 3.10.20 | OpenVLA-OFT 验证与交叉策略 |

### 4.1 源机实测版本（2026-08-03）

**smolvla**

| 包 | 版本 |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0+cu128 |
| lerobot | 0.5.1 |
| numpy | 2.2.6 |
| mujoco | 3.3.2 |
| robosuite | 1.4.0 |

（旧 `env.lock.md` 曾写 mujoco 3.8.1；以本机 `pip` 实测与
`envs/smolvla.pip.freeze.txt` 为准。）

**oft**

| 包 | 版本 |
|---|---|
| Python | 3.10.20 |
| torch | 2.10.0+cu128 |
| transformers | 4.40.1 editable |
| numpy | 1.26.4 |
| mujoco | 2.3.7 |
| robosuite | 1.4.1 |

### 4.2 从 yaml + freeze 重建（推荐）

```bash
cd "$STACK/RASE"

# SmolVLA
conda env create -f envs/smolvla.yaml -p "$STACK/envs/smolvla"
conda activate "$STACK/envs/smolvla"
pip install -e "$STACK/src/LIBERO-plus"
pip install -e .
# 用 freeze 对齐剩余包
grep -vE '^-e |@ file://' envs/smolvla.pip.freeze.txt | pip install -r /dev/stdin

# OFT
conda env create -f envs/oft.yaml -p "$STACK/envs/oft"
conda activate "$STACK/envs/oft"
pip install -e "$STACK/src/transformers-openvla-oft"
pip install -e "$STACK/src/openvla-oft"
pip install -e "$STACK/src/LIBERO-plus"
pip install -e .
grep -vE '^-e |@ file://' envs/oft.pip.freeze.txt | pip install -r /dev/stdin
```

### 4.3 整拷 env 后必做

```bash
conda activate "$STACK/envs/smolvla"
pip install -e "$STACK/src/LIBERO-plus"
pip install -e "$STACK/RASE"
python -c "import libero; print(libero.__file__)"
# 应指向 $STACK/src/LIBERO-plus/...

conda activate "$STACK/envs/oft"
pip install -e "$STACK/src/transformers-openvla-oft"
pip install -e "$STACK/src/openvla-oft"
pip install -e "$STACK/src/LIBERO-plus"
pip install -e "$STACK/RASE"
```

### 4.4 运行时环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$STACK/hf_cache"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export RASE_POLICY_PATH="$STACK/RASE/ckpts/smolvla_libero"
export RASE_OFT_CHECKPOINT="$STACK/RASE/ckpts/oft_spatial"
```

源机 `.bashrc` 已设 `HF_ENDPOINT` / `HF_HOME`。

### 4.5 LIBERO 配置文件（三份）

源机：

| 文件 | 用途 |
|---|---|
| `~/.libero/config.yaml` | 默认可指向 LIBERO-plus |
| `~/.libero_plus_rase/config.yaml` | Plus 扰动线 |
| `~/.libero_clean_rase/config.yaml` | W9C clean 身份线 |

模板（把 `ROOT` 换成 `$STACK/src/LIBERO-plus`）：

```yaml
assets: ROOT/libero/libero/assets
bddl_files: ROOT/libero/libero/bddl_files
benchmark_root: ROOT/libero/libero
datasets: ROOT/libero/datasets
init_states: ROOT/libero/libero/init_files
```

```bash
mkdir -p ~/.libero ~/.libero_plus_rase ~/.libero_clean_rase
# 三份内容在源机几乎相同路径；W9C clean 线依赖 vanilla BDDL/init 身份逻辑（代码内），
# 但磁盘 config 仍指向 LIBERO-plus 树。复制后改绝对路径即可。
for d in .libero .libero_plus_rase .libero_clean_rase; do
  cp machine/libero.config.yaml.template "$HOME/$d/config.yaml"
done
# 然后 sed 替换 ROOT
```

仓库内模板：`machine/libero.config.yaml.template`。

---

## 5. 迁机验收

### 5.1 轻量（无需 GPU 权重）

```bash
cd "$STACK/RASE"
conda activate "$STACK/envs/smolvla"
pip install -e '.[dev]'
pytest -q tests/test_task_fingerprint_stability.py tests/test_w9c_schedule.py
python -m rase.envs.task_catalog --check || true
```

### 5.2 SmolVLA 权重加载

```bash
export RASE_POLICY_PATH="$STACK/RASE/ckpts/smolvla_libero"
python - <<'PY'
from pathlib import Path
p=Path("ckpts/smolvla_libero/config.json")
assert p.is_file(), p
print("smolvla ok", p)
PY
```

### 5.3 OFT smoke

```bash
conda activate "$STACK/envs/oft"
export RASE_OFT_CHECKPOINT="$STACK/RASE/ckpts/oft_spatial"
pytest -q tests/test_oft_adapter.py || true   # 若测试名有演进，改跑现有 oft 相关测试
python -c "from pathlib import Path; assert Path('ckpts/oft_spatial/config.json').is_file()"
```

### 5.4 确认摘要与进度

```bash
test -f progress/2026-07-31_w9c_selector_gate_result.md && echo OK_progress
test -d reports/run_summaries && echo OK_summaries
find reports/run_summaries -name summary.json | wc -l   # 源机约 3000+
```

---

## 6. Git 提交策略（本机）

**提交：**

- `rase/` `scripts/` `configs/` `tests/` `protocol/` `docs/` `plan/`
- 全部 `progress/*.md`（忽略 `._*` AppleDouble）
- `reports/run_summaries/**`、artifact manifest
- `MACHINE_REPLICATION.md`、更新后的 `env.lock.md`、pip freeze

**不提交：**

- `ckpts/` `pool/` `runs/`（体积禁止）
- `*.log`、conda env、`._*` 垃圾文件
- 源机旁路目录 `RASE_pre_w*` / `*.tar.gz` patch（历史补丁；主仓已合并则不必迁）

推送：

```bash
cd /root/autodl-tmp/RASE
git push -u origin main
```

---

## 7. 历史旁路目录（源机 `/root/autodl-tmp`）

| 路径 | 说明 | 要不要拷 |
|---|---|---|
| `RASE_pre_w8_*` `RASE_pre_w9_*` | 打补丁前快照 | 一般不需要 |
| `RASE_w8_*.tar.gz` `RASE_w9_*.tar.gz` | 补丁包 | 主仓已含代码则否 |
| `RASE-project/` | 几乎空壳 | 否 |
| `runs/`（autodl-tmp 根下） | 早期全局 runs | 按需 |
| `CareVLA/` | 并行项目 | 见 CareVLA 文档 |

---

## 8. 常见坑

1. **LIBERO-plus 只 clone 没拷 assets（9.5G）** → 仿真缺模型文件。
2. **`~/.libero*` 绝对路径未改** → 静默读到空/错树。
3. **W9B/W9C 任务身份**：Plus index 0–9 ≠ clean-10；必须用 clean 身份加载逻辑（见 progress W9C）。
4. **Smol / OFT 环境混用** → mujoco/robosuite 主版本不同，必炸。
5. **只迁代码不迁 `pool/`** → 无法续跑 selector / held-out。
6. **把 `ckpts/` git add** → push 不可能成功。
7. **GPU ordinal**：`CUDA_VISIBLE_DEVICES` 裁剪后进程内永远是 `cuda:0`。

---

## 9. 与 CareVLA 联合迁移

两者共享：

- `envs/oft`
- `src/openvla-oft` + `transformers-openvla-oft` + `LIBERO-plus`
- `RASE/ckpts/oft_*`

建议**一次 rsync 共享层**，再分别 `git clone` 两个仓库。CareVLA 文档：
`../CareVLA/MACHINE_REPLICATION.md`。

### 建议迁机顺序

1. 建 `$STACK` 目录树  
2. clone RASE + CareVLA  
3. rsync `src/` + `envs/` + `RASE/ckpts` + `RASE/pool` + `RASE/runs`  
4. rsync `CareVLA/data`  
5. 写环境变量与 `~/.libero*`  
6. 重装 editable + `pip install -e .`  
7. 跑两边最小验收  
8. 再打开科研主线（CareVLA：natural `t_sym`；RASE：按最新 progress）

---

## 10. 磁盘预算（新机）

| 组合 | 大约 |
|---|---:|
| 仅代码 + summaries | <100M |
| + Smol ckpts | ~3G |
| + 全部 OFT ckpts | ~68G |
| + pool + runs | +15G |
| + LIBERO-plus assets | +10G |
| + 两个 conda env | +22G |
| **全量舒适复刻** | **≈110–130G 可用空间** |

源机 `autodl-tmp` 总量 200G，已用约 132G。
