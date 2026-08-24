#!/usr/bin/env bash
# RASE 迁移验证脚本 —— 在新服务器上执行，逐项确认迁移成功
# 用法: bash verify_migration.sh [RASE_DIR]
# 源服务器参考值(2026-08-24):
#   git HEAD      001b23a (GitHub, 24 commits)
#   smolvla cfg   e346a63094732ea28e05b08302a218306cf6541960b663a3cb4b6948e4efe6e0
#   pi0fast cfg   742bd4fed667274e2a497b8d5fff34b968e1349f1d43c7dca22ffc2df41d9944
#   pi05 cfg      2f6d4b96b032593e1de65b391ee7252a70227bff398ef1b94658fea17818142f
#   core tar      a0985835aa9784ac0c495f230a5cd04baaac283cbdb425854c8619638e2f4dd1
set -uo pipefail
RASE="${1:-/root/autodl-tmp/RASE}"
PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi }

echo "========== RASE 迁移验证 =========="
echo "RASE_DIR: $RASE"
cd "$RASE" || { echo "RASE_DIR 不存在: $RASE"; exit 1; }

echo; echo "--- 1. 代码(Git) ---"
check "git repo 存在"       "test -d .git"
check "提交历史 >=20"       "test \"\$(git rev-list --count HEAD 2>/dev/null)\" -ge 20"
check "远程指向 SCUTxyx/RASE" "git remote -v 2>/dev/null | grep -q 'github.com/SCUTxyx/RASE'"

echo; echo "--- 2. 关键小文件 hash(应等于源服务器) ---"
check "smolvla config.json" "test \"\$(sha256sum ckpts/smolvla_libero/config.json 2>/dev/null | cut -d' ' -f1)\" = e346a63094732ea28e05b08302a218306cf6541960b663a3cb4b6948e4efe6e0"
check "pi0fast config.json" "test \"\$(sha256sum ckpts/pi0fast_libero/config.json 2>/dev/null | cut -d' ' -f1)\" = 742bd4fed667274e2a497b8d5fff34b968e1349f1d43c7dca22ffc2df41d9944"
check "pi05 config.json"    "test \"\$(sha256sum ckpts/pi05_libero/config.json 2>/dev/null | cut -d' ' -f1)\" = 2f6d4b96b032593e1de65b391ee7252a70227bff398ef1b94658fea17818142f"

echo; echo "--- 3. 模型文件大小(应等于源服务器) ---"
check "smolvla model 1.2GB" "test \"\$(stat -c%s ckpts/smolvla_libero/model.safetensors 2>/dev/null)\" = 1218047032"
check "pi0fast model 7.7GB" "test \"\$(stat -c%s ckpts/pi0fast_libero/model.safetensors 2>/dev/null)\" = 7729830688"
check "pi05 model 7.5GB"    "test \"\$(stat -c%s ckpts/pi05_libero/model.safetensors 2>/dev/null)\" = 7473096344"

echo; echo "--- 4. 核心数据 runs/ ---"
check "e4_candidate_pool_audit_v1" "test -f runs/e4_candidate_pool_audit_v1/summary.json"
check "e5_ev_probe_v1"             "test -f runs/e5_ev_probe_v1/summary.json"
check "e6_replan_freq_v2"          "test -f runs/e6_replan_freq_v2/summary.json"
check "g0_pos_long_smolvla_l02"    "test -f runs/g0_pos_long_smolvla_l02_v1/summary.json"
check "g0_pro_object_v1"           "test -f runs/g0_pro_object_v1/summary.json"

echo; echo "--- 5. LIBERO-PRO 扰动数据(autodl-tmp 下) ---"
for d in libero_pro_data libero_pro_root_object libero_pro_root_lan libero_pro_root_swap libero_pro_root_task; do
  check "$d" "test -d /root/autodl-tmp/$d"
done
check "libero_pro_data 有 bddl" "test \$(find /root/autodl-tmp/libero_pro_data -name '*.bddl' 2>/dev/null | wc -l) -ge 100"

echo; echo "--- 6. 源码依赖 ---"
check "src/LIBERO-plus 存在" "test -d /root/autodl-tmp/src/LIBERO-plus"
check "src/LIBERO 存在"      "test -d /root/autodl-tmp/src/LIBERO"

echo; echo "--- 7. Python 环境(激活 smolvla 后) ---"
PY=/root/autodl-tmp/envs/smolvla/bin/python
if [ -x "$PY" ]; then
  check "lerobot 可导入"   "$PY -c 'import lerobot' 2>/dev/null"
  check "transformers 5.x" "$PY -c 'import transformers; assert transformers.__version__.startswith(\"5\")' 2>/dev/null"
  check "numpy/torch 可导入" "$PY -c 'import numpy, torch' 2>/dev/null"
  check "rase 可导入"       "$PY -c 'import sys; sys.path.insert(0,\"scripts\"); import rase_common' 2>/dev/null"
else
  bad "smolvla python 不存在 ($PY) —— 环境未迁移/重建"
fi

echo; echo "--- 8. 模型加载冒烟(需要 GPU, 约2分钟) ---"
if [ -x "$PY" ]; then
  SMOKE=$($PY -c "
import sys; sys.path.insert(0, 'scripts')
import os
os.environ.setdefault('LIBERO_CLEAN_ROOT','/root/autodl-tmp/src/LIBERO')
from rase.collect.forked_rollout import load_lerobot_policy_bundle
b = load_lerobot_policy_bundle('ckpts/smolvla_libero', device='cuda',
    num_steps=10, n_action_steps=10,
    tokenizer_path='ckpts/SmolVLM2-500M-Instruct',
    observation_height=360, observation_width=360)
print('SMOLVLA_LOAD_OK')
" 2>/dev/null | tail -1)
  check "SmolVLA 加载" "test \"$SMOKE\" = SMOLVLA_LOAD_OK"
else
  bad "跳过模型加载(无 python)"
fi

echo; echo "==================================="
echo "结果: PASS=$PASS FAIL=$FAIL"
if [ "$FAIL" -eq 0 ]; then echo "✅ 迁移验证全部通过"; else echo "❌ 有 $FAIL 项失败，见上方 [FAIL] 行"; fi
