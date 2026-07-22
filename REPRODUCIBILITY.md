# RASE 跨服务器完全复现说明

这份文档的目标是：在另一台服务器上，尽量把当前 RASE 的代码、环境、权重、结果和运行方式恢复到和本机一致。

RASE 的复现分成五层：

1. Git 代码版本一致；
2. Python 包和运行时一致；
3. 本地 checkpoint / tokenizer 一致；
4. 运行配置和输出目录结构一致；
5. `progress/` 里的实验结论能被继续追加，而不是被覆盖。

## 1. 仓库里要提交什么

建议提交：

- `rase/`、`configs/`、`scripts/`、`docs/`、`tests/`、`third_party/`、`plan/`；
- `progress/` 里压缩后的实验记录；
- `README.md`、`pyproject.toml`、`env.lock.md`、`LICENSE` 以及必要的说明文档；
- 小体积、可读的结果摘要，例如 markdown 表格、结论、runbook 和手工复盘。

不要提交：

- `ckpts/` 下的模型权重本体；
- `runs/`、`pool/`、`results/` 这类大体积运行产物；
- 虚拟环境目录、缓存目录、临时日志。

当前仓库的 `.gitignore` 已经把 `ckpts/`、`runs/`、`pool/`、`results/` 视作本地产物，这是符合目标的。

## 2. 当前有哪些权重

这个仓库里已经出现并使用过的本地权重主要有四类。

### 2.1 SmolVLA 主 policy

- Hugging Face 仓库 ID：`HuggingFaceVLA/smolvla_libero`
- 本地约定路径：`ckpts/smolvla_libero`

这是 RASE 的冻结 baseline 权重，默认用于 LIBERO clean / collapse / collect 相关流程。

### 2.2 Tokenizer / VLM 权重

- 本地约定路径：`ckpts/SmolVLM2-500M-Instruct`

这个权重在 NGC / W3 相关流程里和 `ckpts/smolvla_libero` 配套使用。

### 2.3 OpenVLA-OFT 系列权重

仓库中已经出现并使用过的本地目录包括：

- `ckpts/oft_spatial`
- `ckpts/oft_object`
- `ckpts/oft_goal`
- `ckpts/oft_10`

这几个目录用于不同 LIBERO suite 的 OFT 验证。它们是本地 checkpoint 目录，不要提交到 Git。

### 2.4 其他本地候选权重

如果后面你又加了新的实验权重，也应该继续遵守同一规则：

- 本地目录可以存在；
- Git 只保存 SHA、来源、用途和版本锁定信息；
- 真正的大权重文件留在磁盘或对象存储里。

## 3. 新服务器上怎么拉权重

### 3.1 SmolVLA 主 checkpoint

```bash
mkdir -p ckpts
huggingface-cli download HuggingFaceVLA/smolvla_libero --local-dir ckpts/smolvla_libero
```

如果你的网络要走镜像，先设置 Hugging Face 镜像端点或代理，再下载。

### 3.2 Tokenizer / VLM 权重

如果你还需要 `SmolVLM2-500M-Instruct`，同样放到本地约定目录：

```bash
huggingface-cli download <对应来源仓库或镜像源> --local-dir ckpts/SmolVLM2-500M-Instruct
```

如果你不确定来源，优先查看对应 runbook、`configs/*.yaml` 和 `progress/*.md` 里记录的来源说明，再补下载命令。仓库里当前约定的关键是目录名必须对上，不要把 tokenizer 放错位置。

### 3.3 OFT 系列 checkpoint

本仓库当前使用的 OFT 路线已经把不同 suite 的权重放到了本地 `ckpts/oft_*` 目录中。迁移到新服务器时有两种办法：

1. 直接拷贝整个 `ckpts/oft_*` 目录；
2. 按对应 suite 重新从 Hugging Face 或上游仓库下载，再放回相同目录名。

如果你只是想复现当前结果，最稳妥的是直接拷贝目录。因为某些 OFT 目录里不仅有权重，还有模型实现文件和 README，目录结构本身就是协议的一部分。

## 4. 环境怎么配

RASE 已经把环境定义拆成了两层：

- `pyproject.toml` 负责 Python 包依赖；
- `env.lock.md` 记录已经验证过的机器级版本和 provenance。

当前 `pyproject.toml` 的核心要求是：

- Python `>=3.10,<3.13`
- `numpy>=1.26,<3`
- `PyYAML>=6,<7`

开发环境还会用到：

- `pytest`
- `ruff`

推荐的创建方式：

```bash
conda env create -f envs/smolvla.yaml
conda activate rase-smolvla
pip install -e /path/to/LIBERO-plus
pip install -e .
```

如果你只做开发检查，也可以安装 dev extras：

```bash
pip install -e '.[dev]'
```

为了跨服务器保持一致，建议额外保存这几项：

- Python 版本；
- conda 环境名；
- `pip freeze` 输出；
- `env.lock.md` 的 SHA-256；
- LIBERO-plus 的 commit；
- 运行时用到的 `CUDA_VISIBLE_DEVICES`、`MUJOCO_EGL_DEVICE_ID`、`MUJOCO_GL`、`PYOPENGL_PLATFORM`。

## 5. 运行时必须的环境变量

对 MuJoCo / EGL 相关流程，常见设置是：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

做 forkable env / restore / smoke 相关测试时，还应该保证：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

如果只暴露一张卡，进程内 ordinal 通常是 `0`，不要把物理 GPU 编号和进程编号混淆。

## 6. 怎么跑一遍最小验证

### 6.1 代码和配置检查

```bash
cd /data/data2/yuxuan/RASE
python -m rase.envs.task_catalog --check
pytest -q
```

### 6.2 冻结 baseline 验证

```bash
export RASE_POLICY_PATH=/data/data2/yuxuan/RASE/ckpts/smolvla_libero
python scripts/eval_libero_plus_collapse.py \
  --config configs/collapse_camera_robot.yaml --dry-run
```

### 6.3 ForkableEnv gate

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RASE_TEST_BDDL=/absolute/path/to/task_a.bddl
export RASE_TEST_OTHER_BDDL=/absolute/path/to/task_b.bddl
export RASE_TEST_GPU_ID=0

pytest -q tests/test_fork_roundtrip.py \
  tests/test_fork_noise_rng.py tests/test_wrong_task_restore.py
```

### 6.4 OFT 交叉验证

如果你已经把 `ckpts/oft_*` 拷到新服务器，可以再跑：

```bash
export RASE_OFT_CHECKPOINT=/data/data2/yuxuan/RASE/ckpts/oft_spatial
pytest -q tests/test_oft_adapter.py
```

如果实际脚本名称以后变了，就以 `scripts/` 和 `tests/` 里现有文件为准；原则是不变的：先验证最小闭环，再跑完整 suite。

## 7. Git 怎么做

建议的 Git 工作流是：

1. 只把代码、配置、文档和压缩后的实验结论放进 Git；
2. 不把 `ckpts/`、`runs/`、`pool/`、`results/` 提交进去；
3. 提交前检查 `git status --short`；
4. 如果某个权重目录以前已经被误提交过，先 `git rm --cached`，再补 `.gitignore`；
5. `progress/` 适合存实验记录和阶段结论。

一个比较安全的提交流程：

```bash
git status --short
git add README.md pyproject.toml env.lock.md configs docs plan progress rase scripts tests third_party
git commit -m "Add reproducibility docs and experiment records"
git push
```

如果你要把某个实验结果保留下来，优先把它写成 markdown 报告放到 `progress/`，而不是把整个 `runs/` 或 `ckpts/` 目录塞进仓库。

## 8. 新服务器迁移顺序

推荐按下面顺序恢复：

1. `git clone` RASE；
2. 复制或重新创建 conda 环境；
3. 安装 `pyproject.toml` 里的依赖；
4. 安装 LIBERO-plus 以及对应的上游项目；
5. 拉取 `ckpts/smolvla_libero`、`ckpts/SmolVLM2-500M-Instruct` 和需要的 `ckpts/oft_*`；
6. 设置 `MUJOCO_GL`、`PYOPENGL_PLATFORM`、`CUDA_VISIBLE_DEVICES`、`MUJOCO_EGL_DEVICE_ID`；
7. 跑 `pytest -q` 和最小 dry-run；
8. 再跑正式实验。

## 9. 最容易出错的地方

- 只 clone 了代码，没有拉 `ckpts/`；
- `ckpts/` 拉了，但目录名不对；
- 新服务器的 Python 版本超出了 `>=3.10,<3.13`；
- `MUJOCO_GL` 没设成 `egl`，导致 headless 失败；
- 把 `results/` 或 `runs/` 当成代码一起提交，仓库很快膨胀；
- 实验结果写了，但没有在 `progress/` 里记录 checkpoint、seed 和配置。

如果你以后换了 checkpoint，先更新这里的权重清单和 `progress/`，再继续跑实验。这样新服务器才能真正对齐，不会只是“代码能跑”但“数值不一样”。