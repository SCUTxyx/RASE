import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("LIBERO_PLUS_ROOT", "/root/autodl-tmp/src/LIBERO-plus")
sys.path.insert(0, "/root/autodl-tmp/RASE")

from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
from rase.backends.libero_plus_paths import ensure_libero_plus_paths
from rase.collect.forked_rollout import (
    InProcessLeRobotContinuation,
    load_lerobot_policy_bundle,
    restore_pool_state,
)
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.state_pool import StatePool

ensure_libero_plus_paths(os.environ.get("LIBERO_PLUS_ROOT"))
_patch_lerobot_init_states()

ROOT = Path("/root/autodl-tmp/RASE")
pool = StatePool(ROOT / "runs/pre_c0_r7/r7a_pi0fast_reset_pool_v1")
bundle = load_lerobot_policy_bundle(
    ROOT / "ckpts/pi0fast_libero", device="cuda",
    num_steps=10, n_action_steps=10,
    tokenizer_path=ROOT / "ckpts/paligemma_tokenizer_35e4f46",
    action_tokenizer_path=ROOT / "ckpts/pi0fast_action_tokenizer_79ae83e",
    observation_height=360, observation_width=360,
)
policy = bundle["policy"]
print("policy type:", type(policy).__name__)
print("has _queues:", hasattr(policy, "_queues"), "| has _action_queue:", hasattr(policy, "_action_queue"))
print("after load _action_queue len:", len(getattr(policy, "_action_queue", [])))
policy.reset()
print("after reset _action_queue len:", len(getattr(policy, "_action_queue", [])))
print("after reset _queues:", {k: len(v) for k, v in getattr(policy, "_queues", {}).items()})

manifest = json.loads((ROOT / "runs/rase_vnext/frozen/k3_e0_native_capture_smoke_manifest_v1.json").read_text())
job = manifest["jobs"][0]
restored = restore_pool_state(pool, job["state_key"], libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT"))
single = restored.handle.vector_env.envs[0]
obs = observation_from_libero_env(single)
instruction = str(getattr(single, "task_description", "") or restored.loaded.metadata.instruction)
print("instruction:", instruction[:60])
print("obs keys:", list(obs.keys()))

cont = InProcessLeRobotContinuation(bundle, seed=2118738816, capture=True, capture_horizon=10)
cont.note_boundary_step(8)
try:
    action = cont.act(obs, task=instruction)
    print("act #1 OK, action:", action.shape)
    event = cont.current_inference_event()
    print("event:", None if event is None else (event.inference_event_id, event.env_chunk.shape, event.chunk_size))
    print("consumed:", cont.consumed_in_current_event())
    # second act (should consume queue, not forward)
    action2 = cont.act(obs, task=instruction)
    print("act #2 OK, consumed:", cont.consumed_in_current_event())
except Exception:
    traceback.print_exc()
    print("queues at failure:", {
        "action_queue": len(getattr(policy, "_action_queue", [])),
        "queues": {k: len(v) for k, v in getattr(policy, "_queues", {}).items()},
    })
restored.close()
print("DIAG_DONE")
