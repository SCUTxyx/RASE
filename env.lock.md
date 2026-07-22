# RASE environment lock

This file is the human-readable source of truth for reproduced environments.
Conda specifications live in `envs/`. Record exact resolved package versions
and upstream revisions here before treating a new environment as reproducible.

## Verified W1 SmolVLA environment

| Component | Locked value | Provenance |
|---|---|---|
| Python | 3.12.13 | local `smolvla` environment; AutoDL baseline used Python 3.12 |
| LeRobot | 0.5.1 | installed package |
| PyTorch | 2.10.0+cu128 | local `smolvla` environment |
| MuJoCo | 3.8.1 | local `smolvla` environment |
| robosuite | 1.4.0 | local `smolvla` environment |
| num2words | 0.5.14 | installed package |
| SmolVLA checkpoint | `HuggingFaceVLA/smolvla_libero` | local path `ckpts/smolvla_libero` |
| VLM weights | `SmolVLM2-500M-Instruct` | local path `ckpts/SmolVLM2-500M-Instruct` |
| LIBERO-plus | commit `4976dc3` | editable checkout at `/data/data2/yuxuan/LIBERO-plus` |
| LIBERO assets | local symlink | `/data/data2/yuxuan/LIBERO/libero/libero/assets` |
| ImageMagick | conda-forge `imagemagick` | required by Wand / LIBERO-plus motion blur |
| Wand | 0.7.2 | `pip install Wand` |
| scikit-image | 0.26.0 | LIBERO-plus `env_wrapper` noise path |
| gym | 0.26.2 | LIBERO-plus `venv` import (not Gymnasium) |

The local `~/.libero/config.yaml` points to the SmolVLA environment's LIBERO
paths. This is machine state, not a portable repository setting.

### Locked baseline behavior

| Field | Value |
|---|---|
| Policy mode | frozen |
| `policy.num_steps` | 10 |
| `policy.n_action_steps` | 10 |
| Seed | 0 |
| Episodes | 50 per task; 2,000 total |
| Clean suite mean success | **70.0%** |

Suite results are 67.2% Spatial, 90.2% Object, 78.6% Goal, and 44.0%
LIBERO-10. The complete record is
`progress/2026-07-16_smolvla_clean_libero_baseline_seed0_nas10.md`.

### Runtime constraints

- Use EGL for headless MuJoCo: `MUJOCO_GL=egl` and
  `PYOPENGL_PLATFORM=egl`.
- `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` must select the same
  physical GPU.
- Do not infer reproducibility from this summary alone. Archive `python -V`,
  `pip freeze`, the resolved evaluation config, Git SHA, and this file's
  SHA-256 with every formal run.

## OFT environment

Status: **load smoke verified locally; source archives are not Git checkouts**.

| Component | Locked value | Provenance |
|---|---|---|
| Python | 3.10.20 | local `oft` environment |
| PyTorch | 2.2.0+cu121 | local `oft` environment |
| transformers | 4.40.1 | editable OpenVLA-OFT fork at `/data/data2/yuxuan/transformers-openvla-oft` |
| protobuf | 4.25.9 | compatibility pin |
| tensorflow-metadata | 1.14.0 | compatibility pin |
| OpenVLA-OFT source | local archive checkout | `/data/data2/yuxuan/openvla-oft`; no Git SHA available |
| dlimp | local archive checkout | `/data/data2/yuxuan/dlimp_openvla`; no Git SHA available |
| checkpoint | OFT LIBERO Spatial | `ckpts/oft_spatial` |

The checkpoint load smoke succeeded with all GPUs visible and
`.to("cuda:0")`. If only one GPU is exposed through `CUDA_VISIBLE_DEVICES`,
the process-local CUDA ordinal remains `0`, regardless of the physical GPU ID.
Archive `pip freeze` before a formal OFT baseline because the source archives
do not provide commit identities.

## RL environment

Status: **not created**. `envs/rl.yaml` is a future bootstrap only; do not
install or cite it before W10.

## Change log

- 2026-07-17: normalized W1 provenance; locked Python 3.12, LeRobot 0.5.1,
  LIBERO-plus `4976dc3`, and the frozen seed-0 `nas10` baseline.
- 2026-07-17: recorded local SmolVLA simulator versions and the OFT load-smoke
  environment; marked archive-based OFT dependencies as lacking Git SHAs.
