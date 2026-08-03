from __future__ import annotations

from types import SimpleNamespace

import pytest

from rase.envs.forkable_env import ForkableEnv, UnsupportedEnvironmentError


class _RuntimeMutableModel:
    nq = 12
    nv = 11
    nu = 7
    nbody = 9
    njnt = 8
    ngeom = 10
    nsite = 4
    ncam = 2
    names = b"world\x00robot\x00agentview\x00"

    def __init__(self) -> None:
        self.runtime_value = 0

    def get_xml(self) -> str:
        return f'<mujoco runtime_value="{self.runtime_value}"/>'


def _fingerprinter(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("(define (problem clean-control))", encoding="utf-8")
    task = SimpleNamespace(
        problem_name="clean-control",
        domain_name="libero",
        language_instruction="move the object",
        task_id=1,
        bddl_file_name=str(bddl),
    )
    wrapper = SimpleNamespace(env=task, sim=SimpleNamespace(model=_RuntimeMutableModel()))
    forkable = object.__new__(ForkableEnv)
    forkable.env = wrapper
    return forkable, task, wrapper.sim.model


def test_task_fingerprint_ignores_runtime_xml_values(tmp_path):
    forkable, _, model = _fingerprinter(tmp_path)
    before = forkable._compute_task_fingerprint()
    model.runtime_value = 123
    assert forkable._compute_task_fingerprint() == before


def test_task_fingerprint_still_changes_across_tasks(tmp_path):
    forkable, task, _ = _fingerprinter(tmp_path)
    before = forkable._compute_task_fingerprint()
    task.task_id = 2
    assert forkable._compute_task_fingerprint() != before


def test_model_without_stable_topology_requires_explicit_identity(tmp_path):
    forkable, _, _ = _fingerprinter(tmp_path)
    forkable.env.sim.model = SimpleNamespace(get_xml=lambda: "<mujoco/>")
    with pytest.raises(UnsupportedEnvironmentError, match="model_fingerprint"):
        forkable._compute_task_fingerprint()
