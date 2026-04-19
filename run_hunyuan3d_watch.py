#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.mesh_pipeline import PipelineSettings, run_pipeline


class ConsoleSink:
    def stage(self, stage_key: str, *, status: str, progress: float, message: str | None = None) -> None:
        percent = f"{progress * 100:5.1f}%"
        print(f"[{stage_key:>13}] {status:<9} {percent} {message or ''}".rstrip())

    def asset(
        self,
        stage_key: str,
        *,
        kind: str,
        label: str,
        path: Path,
        mime_type: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        print(f"[{stage_key:>13}] asset     {kind:<5} {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hunyuan3D watch pipeline from the command line.")
    parser.add_argument("--image", default="watch.png", help="Path to the input image.")
    parser.add_argument("--output", default="output/watch_hunyuan3d21_full.glb", help="Output GLB path.")
    parser.add_argument("--keep-background", action="store_true", help="Skip background removal.")
    args = parser.parse_args()

    settings = PipelineSettings(
        image_path=Path(args.image).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        remove_background=not args.keep_background,
    )
    run_pipeline(settings, ConsoleSink())


if __name__ == "__main__":
    main()
