#!/usr/bin/env bash
# Run ON THE NEW MACHINE. Example:
#   SRC=user@old-host STACK=/data/rase-stack bash machine/rsync_from_autodl.sh
set -euo pipefail
SRC=${SRC:?set SRC=user@host}
STACK=${STACK:?set STACK=/path/to/stack}
RSYNC=(rsync -aH --info=progress2)
FULL=${FULL:-1}   # set FULL=0 to skip pool/runs

mkdir -p "$STACK/RASE" "$STACK/src" "$STACK/envs" "$STACK/hf_cache"

"${RSYNC[@]}" "$SRC:/root/autodl-tmp/RASE/ckpts/" "$STACK/RASE/ckpts/"
if [[ "$FULL" == "1" ]]; then
  "${RSYNC[@]}" "$SRC:/root/autodl-tmp/RASE/pool/" "$STACK/RASE/pool/"
  "${RSYNC[@]}" "$SRC:/root/autodl-tmp/RASE/runs/" "$STACK/RASE/runs/"
fi
"${RSYNC[@]}" \
  "$SRC:/root/autodl-tmp/src/LIBERO-plus" \
  "$SRC:/root/autodl-tmp/src/openvla-oft" \
  "$SRC:/root/autodl-tmp/src/transformers-openvla-oft" \
  "$STACK/src/"
"${RSYNC[@]}" "$SRC:/root/autodl-tmp/envs/smolvla/" "$STACK/envs/smolvla/" || true
"${RSYNC[@]}" "$SRC:/root/autodl-tmp/envs/oft/" "$STACK/envs/oft/" || true
echo "Done. Next: clone/pull RASE git, fix ~/.libero*, reinstall editables. See MACHINE_REPLICATION.md"
