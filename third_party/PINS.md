# Third-party pins

Third-party source is consumed as an installed dependency; it is not vendored
or modified in this repository. Any external patch must live under
`third_party/patches/`, state its upstream revision, and have a focused test.

## Active pins

### LeRobot

- Package: `lerobot`
- Version: `0.5.1`
- Role: frozen SmolVLA loading and LIBERO evaluation
- Install: `pip install "lerobot[libero]==0.5.1"`

### LIBERO-plus

- Upstream: <https://github.com/sylvestf/LIBERO-plus>
- Commit: `4976dc3`
- Role: perturbed LIBERO environments
- Install from a detached, verified checkout:

```bash
git -C /path/to/LIBERO-plus checkout 4976dc3
git -C /path/to/LIBERO-plus rev-parse --short HEAD
pip install -e /path/to/LIBERO-plus
python -c "import libero; print(libero.__file__)"
```

The final command must resolve to the intended LIBERO-plus checkout. Do not
patch the checkout in place; use repository-side wrappers.

### SmolVLA checkpoint

- Registry ID: `HuggingFaceVLA/smolvla_libero`
- Local convention: `ckpts/smolvla_libero`
- Mode: frozen inference

The model snapshot hash has not yet been recorded. Until it is, the registry ID
plus local files are insufficient for byte-for-byte checkpoint provenance.

### OpenVLA-OFT stack

- OpenVLA-OFT source: `/data/data2/yuxuan/openvla-oft`
- transformers fork: `/data/data2/yuxuan/transformers-openvla-oft`
- dlimp fork: `/data/data2/yuxuan/dlimp_openvla`
- OFT checkpoint: `ckpts/oft_spatial`
- Runtime: Python 3.10.20, PyTorch 2.2.0+cu121, transformers 4.40.1
- Compatibility pins: protobuf 4.25.9, tensorflow-metadata 1.14.0

These three sources were unpacked from archives and are not Git worktrees, so
they cannot supply a trustworthy commit SHA. The load smoke is verified, but a
formal OFT baseline must additionally archive source checksums and `pip freeze`.

## Pending pins

- Byte-level hashes for the SmolVLA and OFT checkpoints.
- Archive SHA-256 values for OpenVLA-OFT, its transformers fork, and dlimp.
- The W10 RL environment and feature-model checkpoints.
