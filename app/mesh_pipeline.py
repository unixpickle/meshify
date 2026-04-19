from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch
from PIL import Image

from . import config


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", str(config.HF_HOME))
os.environ.setdefault("HY3DGEN_MODELS", str(config.HY3DGEN_MODELS))


class StageSink(Protocol):
    def stage(
        self,
        stage_key: str,
        *,
        status: str,
        progress: float,
        message: str | None = None,
    ) -> None: ...

    def asset(
        self,
        stage_key: str,
        *,
        kind: str,
        label: str,
        path: Path,
        mime_type: str,
        metadata: dict[str, object] | None = None,
    ) -> None: ...


@dataclass(slots=True)
class PipelineSettings:
    image_path: Path
    output_path: Path
    remove_background: bool = True
    model_path: str = config.MODEL_PATH
    subfolder: str = config.MODEL_SUBFOLDER
    variant: str = config.MODEL_VARIANT
    steps: int = config.DEFAULT_STEPS
    octree_resolution: int = config.DEFAULT_OCTREE_RESOLUTION
    num_chunks: int = config.DEFAULT_NUM_CHUNKS
    seed: int = config.DEFAULT_SEED
    device: str | None = config.DEFAULT_DEVICE


_PIPELINE_CACHE: dict[tuple[str, str, str, str, str | None], object] = {}
_PIPELINE_LOCK = threading.Lock()


def choose_device() -> str:
    if config.DEFAULT_DEVICE:
        return config.DEFAULT_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def choose_variant(requested_variant: str, dtype: torch.dtype) -> str | None:
    if requested_variant == "none":
        return None
    if requested_variant == "fp16":
        return "fp16"
    return "fp16" if dtype == torch.float16 else None


def create_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path)


def save_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def preprocess_image(image_path: Path, remove_background: bool) -> Image.Image:
    image = _load_image(image_path)
    if remove_background:
        from hy3dgen.rembg import BackgroundRemover

        image = BackgroundRemover()(image)
    return image.convert("RGBA")


def _load_shape_pipeline(
    model_path: str,
    subfolder: str,
    device: str,
    dtype: torch.dtype,
    variant: str | None,
):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.utils import smart_load_model

    cache_key = (model_path, subfolder, device, str(dtype), variant)
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[cache_key]

    candidate_variants: list[str | None] = [variant]
    if variant is not None:
        candidate_variants.append(None)

    last_error: Exception | None = None
    for candidate_variant in candidate_variants:
        for use_safetensors in (True, False):
            config_path, ckpt_path = smart_load_model(
                model_path,
                subfolder=subfolder,
                use_safetensors=use_safetensors,
                variant=candidate_variant,
            )
            if not os.path.exists(ckpt_path):
                last_error = FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                continue

            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                ckpt_path,
                config_path,
                device=device,
                dtype=dtype,
                use_safetensors=use_safetensors,
            )
            with _PIPELINE_LOCK:
                _PIPELINE_CACHE[cache_key] = pipeline
            return pipeline

    raise RuntimeError(
        "Unable to locate a compatible checkpoint for "
        f"{model_path}/{subfolder}. Last error: {last_error}"
    )


class ProgressReporter:
    def __init__(self, sink: StageSink):
        self.sink = sink
        self.active_stage: str | None = None
        self.last_bucket: dict[str, int] = {}

    def map_desc(self, desc: str) -> str | None:
        if "Diffusion Sampling" in desc:
            return "diffusion"
        if "Volume Decoding" in desc:
            return "volume_decode"
        return None

    def update(self, desc: str, index: int, total: int | None) -> None:
        stage_key = self.map_desc(desc)
        if stage_key is None:
            return
        if self.active_stage != stage_key:
            if self.active_stage is not None:
                self.sink.stage(self.active_stage, status="completed", progress=1.0, message="Done")
            self.active_stage = stage_key
            self.sink.stage(stage_key, status="running", progress=0.0, message=desc)

        if not total:
            return
        progress = min(index / total, 1.0)
        bucket = int(progress * 100)
        if self.last_bucket.get(stage_key) == bucket and bucket not in {0, 100}:
            return
        self.last_bucket[stage_key] = bucket
        self.sink.stage(stage_key, status="running", progress=progress, message=f"{desc} {bucket}%")

    def finish(self) -> None:
        if self.active_stage is not None:
            self.sink.stage(self.active_stage, status="completed", progress=1.0, message="Done")
            self.active_stage = None


@contextlib.contextmanager
def instrument_progress(reporter: ProgressReporter):
    import hy3dgen.shapegen.models.autoencoders.volume_decoders as volume_decoders
    import hy3dgen.shapegen.pipelines as pipelines
    from tqdm import tqdm as base_tqdm

    original_pipeline_tqdm = pipelines.tqdm
    original_volume_tqdm = volume_decoders.tqdm

    def wrapped_tqdm(iterable=None, *args, **kwargs):
        desc = kwargs.get("desc", "") or ""
        total = kwargs.get("total")
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        inner = base_tqdm(iterable, *args, **kwargs)

        class IteratorProxy:
            def __iter__(self):
                for index, item in enumerate(inner, start=1):
                    reporter.update(desc, index, total)
                    yield item

            def __getattr__(self, name):
                return getattr(inner, name)

        return IteratorProxy()

    pipelines.tqdm = wrapped_tqdm
    volume_decoders.tqdm = wrapped_tqdm
    try:
        yield
    finally:
        pipelines.tqdm = original_pipeline_tqdm
        volume_decoders.tqdm = original_volume_tqdm
        reporter.finish()


def run_pipeline(settings: PipelineSettings, sink: StageSink) -> Path:
    device = settings.device or choose_device()
    dtype = choose_dtype(device)
    variant = choose_variant(settings.variant, dtype)

    sink.stage("preprocess", status="running", progress=0.0, message="Preparing image")
    prepared_image = preprocess_image(settings.image_path, settings.remove_background)
    processed_path = settings.output_path.parent / "preprocessed.png"
    save_image(prepared_image, processed_path)
    sink.asset(
        "preprocess",
        kind="image",
        label="Background Removed",
        path=processed_path,
        mime_type="image/png",
    )
    sink.stage("preprocess", status="completed", progress=1.0, message="Background removed")

    sink.stage("model_load", status="running", progress=0.0, message="Loading model weights")
    pipeline = _load_shape_pipeline(
        model_path=settings.model_path,
        subfolder=settings.subfolder,
        device=device,
        dtype=dtype,
        variant=variant,
    )
    sink.stage("model_load", status="completed", progress=1.0, message=f"Ready on {device}")

    sink.stage("diffusion", status="running", progress=0.0, message="Sampling shape")
    reporter = ProgressReporter(sink)
    generator = create_generator(settings.seed)
    with instrument_progress(reporter):
        mesh = pipeline(
            image=prepared_image,
            num_inference_steps=settings.steps,
            octree_resolution=settings.octree_resolution,
            num_chunks=settings.num_chunks,
            generator=generator,
            output_type="trimesh",
            enable_pbar=True,
        )[0]

    sink.stage("export", status="running", progress=0.0, message="Writing GLB")
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(settings.output_path)
    sink.asset(
        "export",
        kind="model",
        label="Generated Mesh",
        path=settings.output_path,
        mime_type="model/gltf-binary",
    )
    sink.stage("export", status="completed", progress=1.0, message="GLB saved")
    return settings.output_path
