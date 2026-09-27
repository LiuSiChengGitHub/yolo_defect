"""Encode actual Qt captures as reusable README and offline demo assets.

Run `qt.cmd media` to capture and encode, or supply an existing --frames folder.
Requires Pillow. GIF frame durations are editorial; this is not a benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "docs/assets/qt")
    args = parser.parse_args()
    manifest = json.loads((args.frames / "frames.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    frames, durations = [], []
    for item in manifest["frames"]:
        with Image.open(args.frames / item["file"]) as source:
            frame = source.convert("RGB")
        frame.thumbnail((1440, 960), Image.Resampling.LANCZOS)
        frame.save(args.output / (Path(item["file"]).stem + ".webp"), quality=88, method=6)
        if item["file"] == "03-results.png":
            frame.save(args.output / "workbench.png", optimize=True)
        frame.thumbnail((1080, 720), Image.Resampling.LANCZOS)
        frames.append(frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
        durations.append(item["duration_ms"])
    if not frames or not (args.output / "workbench.png").is_file():
        raise ValueError("Capture is missing the expected real results frame")
    frames[0].save(args.output / "walkthrough.gif", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)
    (args.output / "capture.json").write_text(json.dumps({
        "source": "cpp_infer/tools/qt_demo_capture.cpp",
        "regenerate": "cpp_infer\\tools\\qt.cmd media",
        "kind": "Actual Qt widget captures with edited timing; not realtime video or performance evidence.",
        "frames": manifest["frames"],
        "device_pixel_ratio": manifest["device_pixel_ratio"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(frames)} real UI frames, workbench.png and walkthrough.gif in {args.output}")


if __name__ == "__main__":
    main()
