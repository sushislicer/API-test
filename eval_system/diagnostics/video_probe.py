from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common.serialization import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rollout video motion without simulator/model dependencies")
    parser.add_argument("video", help="Video path")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride for motion sampling")
    parser.add_argument("--low-motion-threshold", type=float, default=1.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = probe_video(Path(args.video), stride=max(1, args.stride), low_motion_threshold=args.low_motion_threshold)
    print(json.dumps(report, indent=2))
    if args.output:
        output_path = Path(args.output)
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def probe_video(path: Path, *, stride: int = 10, low_motion_threshold: float = 1.0) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video_probe requires opencv-python in the current environment") from exc

    if not path.exists():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    sampled = 0
    diffs: list[float] = []
    previous_gray = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                diffs.append(float(diff.mean()))
            previous_gray = gray
            sampled += 1
        frame_index += 1
    capture.release()

    avg_motion = sum(diffs) / len(diffs) if diffs else None
    max_motion = max(diffs) if diffs else None
    low_motion = avg_motion is not None and avg_motion < low_motion_threshold
    return {
        "video": str(path),
        "opened": True,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sample_stride": stride,
        "sampled_frames": sampled,
        "avg_frame_diff": round(avg_motion, 6) if avg_motion is not None else None,
        "max_frame_diff": round(max_motion, 6) if max_motion is not None else None,
        "low_motion_suspected": low_motion,
    }


if __name__ == "__main__":
    main()
