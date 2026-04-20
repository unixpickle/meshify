from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import trimesh
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
    texgen_model_path: str = config.TEXGEN_MODEL_PATH
    texgen_subfolder: str = config.TEXGEN_SUBFOLDER
    variant: str = config.MODEL_VARIANT
    steps: int = config.DEFAULT_STEPS
    texgen_delight_steps: int = config.TEXGEN_DELIGHT_STEPS
    texgen_multiview_steps: int = config.TEXGEN_MULTIVIEW_STEPS
    octree_resolution: int = config.DEFAULT_OCTREE_RESOLUTION
    num_chunks: int = config.DEFAULT_NUM_CHUNKS
    seed: int = config.DEFAULT_SEED
    device: str | None = config.DEFAULT_DEVICE


_SHAPE_PIPELINE_CACHE: dict[tuple[str, str, str, str, str | None], object] = {}
_TEXGEN_PIPELINE_CACHE: dict[tuple[str, str, str], object] = {}
_PIPELINE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


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


def save_tensor_image(image: torch.Tensor, destination: Path) -> None:
    array = image.detach().float().clamp(0, 1).cpu().numpy()
    save_image(Image.fromarray((array * 255).astype(np.uint8)), destination)


def preprocess_image(image_path: Path, remove_background: bool) -> Image.Image:
    image = _load_image(image_path)
    if remove_background:
        from hy3dgen.rembg import BackgroundRemover

        image = BackgroundRemover()(image)
    return image.convert("RGBA")


def build_contact_sheet(images: list[Image.Image], *, columns: int = 3, padding: int = 12) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required to build a contact sheet")

    prepared = [image.convert("RGB") for image in images]
    cell_width = max(image.width for image in prepared)
    cell_height = max(image.height for image in prepared)
    rows = (len(prepared) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + padding * (columns + 1),
            rows * cell_height + padding * (rows + 1),
        ),
        (245, 243, 238),
    )

    for index, image in enumerate(prepared):
        row = index // columns
        column = index % columns
        x = padding + column * (cell_width + padding) + (cell_width - image.width) // 2
        y = padding + row * (cell_height + padding) + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))

    return sheet


def save_contact_sheet(images: list[Image.Image], destination: Path, *, columns: int = 3) -> None:
    save_image(build_contact_sheet(images, columns=columns), destination)


def load_image_file(image_path: Path, mode: str | None = None) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert(mode) if mode else image.copy()


def save_image_sequence(images: list[Image.Image], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stale_path in directory.glob("*.png"):
        stale_path.unlink()
    for index, image in enumerate(images):
        save_image(image, directory / f"{index:02d}.png")


def load_image_sequence(directory: Path, *, mode: str = "RGB") -> list[Image.Image]:
    return [load_image_file(path, mode) for path in sorted(directory.glob("*.png"))]


def load_mesh_file(mesh_path: Path):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_texture_tensor(texture_path: Path, device: str) -> torch.Tensor:
    texture_array = np.asarray(load_image_file(texture_path, "RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(texture_array).to(device)


def ensure_mesh_has_uvs(mesh, *, context: str) -> None:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        raise RuntimeError(f"{context} is missing UV coordinates")


def prepare_uv_mesh_for_export(mesh):
    ensure_mesh_has_uvs(mesh, context="UV mesh before export")
    placeholder_texture = Image.new("RGB", (2, 2), (255, 255, 255))
    material = trimesh.visual.texture.SimpleMaterial(
        image=placeholder_texture,
        diffuse=(255, 255, 255),
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        image=placeholder_texture,
        material=material,
    )
    return mesh


def _uv_wrap_worker(input_mesh_path: str, output_mesh_path: str, result_conn) -> None:
    try:
        from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

        mesh = load_mesh_file(Path(input_mesh_path))
        wrapped_mesh = mesh_uv_wrap(mesh)
        wrapped_mesh = prepare_uv_mesh_for_export(wrapped_mesh)
        wrapped_mesh.export(output_mesh_path)
        result_conn.send({"status": "ok"})
    except Exception as exc:  # pragma: no cover - defensive subprocess handling
        result_conn.send(
            {
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        result_conn.close()


def run_uv_wrap_in_subprocess(input_mesh_path: Path, output_mesh_path: Path, sink: StageSink) -> Path:
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_uv_wrap_worker,
        args=(str(input_mesh_path), str(output_mesh_path), child_conn),
    )

    logger.info("Starting UV unwrap subprocess for %s", input_mesh_path)
    started_at = time.monotonic()
    process.start()
    child_conn.close()

    try:
        sink.stage("uv_unwrap", status="running", progress=0.1, message=f"UV unwrap running in subprocess (pid {process.pid})")
        while process.is_alive():
            process.join(timeout=2.0)
            if process.is_alive():
                elapsed = int(time.monotonic() - started_at)
                sink.stage(
                    "uv_unwrap",
                    status="running",
                    progress=0.5,
                    message=f"UV unwrap running in subprocess for {elapsed}s",
                )

        process.join()
        result = parent_conn.recv() if parent_conn.poll() else None
        if process.exitcode != 0:
            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(
                    f"UV unwrap subprocess failed: {result['message']}\n{result['traceback']}"
                )
            raise RuntimeError(f"UV unwrap subprocess exited with code {process.exitcode}")
        if not output_mesh_path.exists():
            raise RuntimeError("UV unwrap subprocess finished without writing the UV mesh")
        logger.info("Finished UV unwrap in %.1fs", time.monotonic() - started_at)
        return output_mesh_path
    finally:
        parent_conn.close()


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
        if cache_key in _SHAPE_PIPELINE_CACHE:
            return _SHAPE_PIPELINE_CACHE[cache_key]

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
                _SHAPE_PIPELINE_CACHE[cache_key] = pipeline
            return pipeline

    raise RuntimeError(
        "Unable to locate a compatible checkpoint for "
        f"{model_path}/{subfolder}. Last error: {last_error}"
    )


def _load_texture_pipeline(model_path: str, subfolder: str, device: str):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    cache_key = (model_path, subfolder, device)
    with _PIPELINE_LOCK:
        if cache_key in _TEXGEN_PIPELINE_CACHE:
            return _TEXGEN_PIPELINE_CACHE[cache_key]

    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        device=device,
    )
    with _PIPELINE_LOCK:
        _TEXGEN_PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


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


def run_pipeline(settings: PipelineSettings, sink: StageSink, previous_run: dict[str, object] | None = None) -> Path:
    device = settings.device or choose_device()
    dtype = choose_dtype(device)
    variant = choose_variant(settings.variant, dtype)
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_stages = {
        stage["stage_key"]: stage["status"]
        for stage in (previous_run or {}).get("stages", [])
        if isinstance(stage, dict)
    }

    def stage_can_resume(stage_key: str, *paths: Path) -> bool:
        return previous_stages.get(stage_key) == "completed" and all(path.exists() for path in paths)

    processed_path = settings.output_path.parent / "preprocessed.png"
    white_mesh_path = settings.output_path.parent / "white_mesh.glb"
    delighted_path = settings.output_path.parent / "delighted.png"
    uv_mesh_path = settings.output_path.parent / "uv_mesh.glb"
    normal_sheet_path = settings.output_path.parent / "normal_maps.png"
    position_sheet_path = settings.output_path.parent / "position_maps.png"
    multiview_sheet_path = settings.output_path.parent / "painted_views.png"
    multiview_dir = settings.output_path.parent / "painted_views"
    mask_path = settings.output_path.parent / "texture_mask.png"
    texture_path = settings.output_path.parent / "texture_map.png"

    if stage_can_resume("preprocess", processed_path):
        prepared_image = load_image_file(processed_path, "RGBA")
        sink.asset(
            "preprocess",
            kind="image",
            label="Background Removed",
            path=processed_path,
            mime_type="image/png",
        )
        sink.stage("preprocess", status="completed", progress=1.0, message="Reused background removal")
    else:
        sink.stage("preprocess", status="running", progress=0.0, message="Preparing image")
        prepared_image = preprocess_image(settings.image_path, settings.remove_background)
        save_image(prepared_image, processed_path)
        sink.asset(
            "preprocess",
            kind="image",
            label="Background Removed",
            path=processed_path,
            mime_type="image/png",
        )
        sink.stage("preprocess", status="completed", progress=1.0, message="Background removed")

    if stage_can_resume("mesh_export", white_mesh_path):
        mesh = load_mesh_file(white_mesh_path)
        sink.stage("model_load", status="completed", progress=1.0, message="Skipped shape model load; reusing white mesh")
        sink.stage("diffusion", status="completed", progress=1.0, message="Reused saved shape sampling")
        sink.stage("volume_decode", status="completed", progress=1.0, message="Reused saved volume decode")
        sink.asset(
            "mesh_export",
            kind="model",
            label="White Mesh",
            path=white_mesh_path,
            mime_type="model/gltf-binary",
        )
        sink.stage("mesh_export", status="completed", progress=1.0, message="Reused white mesh")
    else:
        sink.stage("model_load", status="running", progress=0.0, message="Loading shape model")
        pipeline = _load_shape_pipeline(
            model_path=settings.model_path,
            subfolder=settings.subfolder,
            device=device,
            dtype=dtype,
            variant=variant,
        )
        sink.stage("model_load", status="completed", progress=1.0, message=f"Shape model ready on {device}")

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

        sink.stage("mesh_export", status="running", progress=0.0, message="Writing white mesh")
        mesh.export(white_mesh_path)
        sink.asset(
            "mesh_export",
            kind="model",
            label="White Mesh",
            path=white_mesh_path,
            mime_type="model/gltf-binary",
        )
        sink.stage("mesh_export", status="completed", progress=1.0, message="White mesh saved")

    sink.stage("texture_model_load", status="running", progress=0.0, message="Loading paint models")
    texture_pipeline = _load_texture_pipeline(
        settings.texgen_model_path,
        settings.texgen_subfolder,
        device,
    )
    texture_pipeline.config.delight_steps = settings.texgen_delight_steps
    texture_pipeline.config.multiview_steps = settings.texgen_multiview_steps
    sink.stage("texture_model_load", status="completed", progress=1.0, message=f"Paint models ready on {device}")
    relight_input = texture_pipeline.recenter_image(prepared_image)

    if stage_can_resume("delight", delighted_path):
        delighted_image = load_image_file(delighted_path, "RGB")
        sink.asset(
            "delight",
            kind="image",
            label="Light Cleaned Image",
            path=delighted_path,
            mime_type="image/png",
        )
        sink.stage("delight", status="completed", progress=1.0, message="Reused light cleanup")
    else:
        sink.stage("delight", status="running", progress=0.0, message="Cleaning input lighting")

        def update_delight_stage(progress: float, message: str) -> None:
            sink.stage("delight", status="running", progress=min(max(progress, 0.0), 1.0), message=message)

        logger.info(
            "Starting light cleanup on %s with %s delight steps",
            device,
            settings.texgen_delight_steps,
        )
        delight_started_at = time.monotonic()
        delighted_image = texture_pipeline.models["delight_model"](
            relight_input,
            progress_callback=update_delight_stage,
        )
        logger.info(
            "Finished light cleanup in %.1fs",
            time.monotonic() - delight_started_at,
        )
        save_image(delighted_image, delighted_path)
        sink.asset(
            "delight",
            kind="image",
            label="Light Cleaned Image",
            path=delighted_path,
            mime_type="image/png",
        )
        sink.stage("delight", status="completed", progress=1.0, message="Lighting cleaned")

    reusable_uv_mesh = False
    if stage_can_resume("uv_unwrap", uv_mesh_path):
        textured_mesh_input = load_mesh_file(uv_mesh_path)
        uv = getattr(getattr(textured_mesh_input, "visual", None), "uv", None)
        reusable_uv_mesh = uv is not None
        if reusable_uv_mesh:
            texture_pipeline.render.load_mesh(textured_mesh_input)
            sink.asset(
                "uv_unwrap",
                kind="model",
                label="UV Mesh",
                path=uv_mesh_path,
                mime_type="model/gltf-binary",
            )
            sink.stage("uv_unwrap", status="completed", progress=1.0, message="Reused UV unwrap")
        else:
            logger.warning("Discarding UV mesh without UV coordinates: %s", uv_mesh_path)

    if not reusable_uv_mesh:
        sink.stage("uv_unwrap", status="running", progress=0.0, message="Generating UV unwrap")
        run_uv_wrap_in_subprocess(white_mesh_path, uv_mesh_path, sink)
        textured_mesh_input = load_mesh_file(uv_mesh_path)
        ensure_mesh_has_uvs(textured_mesh_input, context="Generated UV mesh")
        texture_pipeline.render.load_mesh(textured_mesh_input)
        sink.asset(
            "uv_unwrap",
            kind="model",
            label="UV Mesh",
            path=uv_mesh_path,
            mime_type="model/gltf-binary",
        )
        sink.stage("uv_unwrap", status="completed", progress=1.0, message="UV unwrap complete")

    selected_camera_elevs = texture_pipeline.config.candidate_camera_elevs
    selected_camera_azims = texture_pipeline.config.candidate_camera_azims
    selected_view_weights = texture_pipeline.config.candidate_view_weights

    if stage_can_resume("multiview", normal_sheet_path, position_sheet_path, multiview_sheet_path) and any(
        multiview_dir.glob("*.png")
    ):
        multiviews = load_image_sequence(multiview_dir, mode="RGB")
        sink.asset(
            "multiview",
            kind="image",
            label="Normal Maps",
            path=normal_sheet_path,
            mime_type="image/png",
        )
        sink.asset(
            "multiview",
            kind="image",
            label="Position Maps",
            path=position_sheet_path,
            mime_type="image/png",
        )
        sink.asset(
            "multiview",
            kind="image",
            label="Painted Views",
            path=multiview_sheet_path,
            mime_type="image/png",
        )
        sink.stage("multiview", status="completed", progress=1.0, message="Reused painted views")
    else:
        sink.stage("multiview", status="running", progress=0.1, message="Rendering geometry guides")
        normal_maps = texture_pipeline.render_normal_multiview(
            selected_camera_elevs,
            selected_camera_azims,
            use_abs_coor=True,
        )
        position_maps = texture_pipeline.render_position_multiview(
            selected_camera_elevs,
            selected_camera_azims,
        )
        save_contact_sheet(normal_maps, normal_sheet_path)
        save_contact_sheet(position_maps, position_sheet_path)
        sink.asset(
            "multiview",
            kind="image",
            label="Normal Maps",
            path=normal_sheet_path,
            mime_type="image/png",
        )
        sink.asset(
            "multiview",
            kind="image",
            label="Position Maps",
            path=position_sheet_path,
            mime_type="image/png",
        )

        sink.stage("multiview", status="running", progress=0.55, message="Generating painted views")
        camera_info = [
            (((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev]
            + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev]
            for azim, elev in zip(selected_camera_azims, selected_camera_elevs)
        ]
        multiviews = texture_pipeline.models["multiview_model"](
            [delighted_image],
            normal_maps + position_maps,
            camera_info,
        )
        multiviews = [
            image.resize((texture_pipeline.config.render_size, texture_pipeline.config.render_size))
            for image in multiviews
        ]
        save_image_sequence(multiviews, multiview_dir)
        save_contact_sheet(multiviews, multiview_sheet_path)
        sink.asset(
            "multiview",
            kind="image",
            label="Painted Views",
            path=multiview_sheet_path,
            mime_type="image/png",
        )
        sink.stage("multiview", status="completed", progress=1.0, message="Painted views ready")

    if stage_can_resume("texture_bake", mask_path, texture_path):
        sink.asset(
            "texture_bake",
            kind="image",
            label="Texture Coverage",
            path=mask_path,
            mime_type="image/png",
        )
        sink.asset(
            "texture_bake",
            kind="image",
            label="Texture Map",
            path=texture_path,
            mime_type="image/png",
        )
        texture = load_texture_tensor(texture_path, device)
        texture_pipeline.render.set_texture(texture)
        textured_mesh = texture_pipeline.render.save_mesh()
        sink.stage("texture_bake", status="completed", progress=1.0, message="Reused baked texture")
    else:
        sink.stage("texture_bake", status="running", progress=0.2, message="Baking texture")
        texture, mask = texture_pipeline.bake_from_multiview(
            multiviews,
            selected_camera_elevs,
            selected_camera_azims,
            selected_view_weights,
            method=texture_pipeline.config.merge_method,
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        save_image(Image.fromarray(mask_np, mode="L"), mask_path)
        sink.asset(
            "texture_bake",
            kind="image",
            label="Texture Coverage",
            path=mask_path,
            mime_type="image/png",
        )

        sink.stage("texture_bake", status="running", progress=0.7, message="Inpainting texture")
        texture = texture_pipeline.texture_inpaint(texture, mask_np)
        save_tensor_image(texture, texture_path)
        sink.asset(
            "texture_bake",
            kind="image",
            label="Texture Map",
            path=texture_path,
            mime_type="image/png",
        )
        texture_pipeline.render.set_texture(texture)
        textured_mesh = texture_pipeline.render.save_mesh()
        sink.stage("texture_bake", status="completed", progress=1.0, message="Texture baked")

    if stage_can_resume("export", settings.output_path):
        sink.asset(
            "export",
            kind="model",
            label="Textured Mesh",
            path=settings.output_path,
            mime_type="model/gltf-binary",
        )
        sink.stage("export", status="completed", progress=1.0, message="Reused textured GLB")
        return settings.output_path

    sink.stage("export", status="running", progress=0.0, message="Writing textured GLB")
    textured_mesh.export(settings.output_path)
    sink.asset(
        "export",
        kind="model",
        label="Textured Mesh",
        path=settings.output_path,
        mime_type="model/gltf-binary",
    )
    sink.stage("export", status="completed", progress=1.0, message="Textured GLB saved")
    return settings.output_path
