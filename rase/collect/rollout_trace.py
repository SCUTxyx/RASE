"""Optional compressed rollout traces for QC and video export.

The recorder is dependency-light at import time. Pillow/imageio are imported
only when tracing or MP4 export is explicitly requested.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraceFrame:
    index: int
    timestep: int
    phase: str
    jpeg: bytes


def _image_leaf(observation: Mapping[str, Any], name: str) -> np.ndarray | None:
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping) or name not in pixels:
        return None
    array = np.asarray(pixels[name])
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"pixels.{name} must be HWC RGB (optionally batched), got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            maximum = float(np.nanmax(array)) if array.size else 0.0
            if maximum <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def observation_montage(observation: Mapping[str, Any]) -> np.ndarray:
    """Return agentview/wrist side-by-side as uint8 RGB."""
    agent = _image_leaf(observation, "image")
    wrist = _image_leaf(observation, "image2")
    if agent is None and wrist is None:
        raise ValueError("observation has neither pixels.image nor pixels.image2")
    if agent is None:
        return wrist  # type: ignore[return-value]
    if wrist is None:
        return agent
    if agent.shape[0] != wrist.shape[0]:
        from PIL import Image

        width = max(1, round(wrist.shape[1] * agent.shape[0] / wrist.shape[0]))
        wrist = np.asarray(
            Image.fromarray(wrist, mode="RGB").resize((width, agent.shape[0]))
        )
    return np.concatenate([agent, wrist], axis=1)


class RolloutTraceRecorder:
    """Collect JPEG-compressed frames through ``evaluate_candidate`` callback."""

    def __init__(
        self,
        *,
        stride: int = 5,
        max_frames: int = 256,
        jpeg_quality: int = 85,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.stride = int(stride)
        self.max_frames = int(max_frames)
        self.jpeg_quality = int(jpeg_quality)
        self.frames: list[TraceFrame] = []
        self._seen = 0

    def __call__(
        self,
        observation: Mapping[str, Any],
        *,
        phase: str,
        timestep: int,
    ) -> None:
        seen = self._seen
        self._seen += 1
        if seen % self.stride or len(self.frames) >= self.max_frames:
            return
        from PIL import Image

        montage = observation_montage(observation)
        buffer = io.BytesIO()
        Image.fromarray(montage, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        self.frames.append(
            TraceFrame(
                index=len(self.frames),
                timestep=int(timestep),
                phase=str(phase),
                jpeg=buffer.getvalue(),
            )
        )

    def write_frame_archive(
        self,
        directory: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write portable JPEG frames plus a JSON manifest."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        records = []
        for frame in self.frames:
            name = f"{frame.index:05d}.jpg"
            (target / name).write_bytes(frame.jpeg)
            row = asdict(frame)
            row.pop("jpeg")
            row["file"] = name
            records.append(row)
        manifest = {
            "version": "rase-rollout-trace/v1",
            "stride": self.stride,
            "max_frames": self.max_frames,
            "frames_seen": self._seen,
            "frames_saved": len(self.frames),
            "frames": records,
            "metadata": dict(metadata or {}),
        }
        path = target / "trace.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_mp4(self, path: str | Path, *, fps: float = 10.0) -> Path:
        """Encode captured frames to MP4 using optional imageio dependencies."""
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not self.frames:
            raise ValueError("cannot write an empty rollout trace")
        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "MP4 export requires `pip install -e '.[video]'`"
            ) from exc

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(target, fps=float(fps), codec="libx264") as writer:
            for frame in self.frames:
                writer.append_data(np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB")))
        return target
