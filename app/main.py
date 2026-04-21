from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import config, store
from .database import init_db
from .events import event_broker
from .jobs import job_manager


config.ensure_directories()
app = FastAPI(title="Meshify")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    config.ensure_directories()
    init_db()
    job_manager.start()


app.mount("/files", StaticFiles(directory=config.STORAGE_DIR), name="files")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return store.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = store.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    run = store.load_run(run_id, include_deleting=True)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    job_manager.request_delete(run_id)


@app.get("/api/events")
async def wait_for_events(since: int = 0, timeout: float = 25.0) -> dict:
    clamped_timeout = max(5.0, min(timeout, 55.0))
    result = await asyncio.to_thread(event_broker.wait_for_update, since, clamped_timeout)
    if result["changed"]:
        result["runs"] = store.list_runs()
    return result


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    config.ensure_directories()
    run_id = store.create_run(file.filename, {"keep_background": False})
    store.initialize_run_stages(run_id)

    upload_dir = config.UPLOADS_DIR / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        with Image.open(destination) as image:
            width, height = image.size
    except UnidentifiedImageError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    store.create_asset(
        run_id,
        "uploaded",
        kind="image",
        label="Uploaded Image",
        storage_path=destination.relative_to(config.STORAGE_DIR),
        mime_type=file.content_type or mimetypes.guess_type(destination.name)[0] or "application/octet-stream",
        metadata={"width": width, "height": height},
    )
    store.upsert_stage(
        run_id,
        "uploaded",
        status="completed",
        progress=1.0,
        message="Image uploaded",
        started=True,
        completed=True,
    )
    job_manager.submit(run_id)
    run = store.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Run was created but could not be loaded")
    return run


@app.get("/{full_path:path}")
def spa(full_path: str) -> FileResponse:
    requested = config.FRONTEND_DIST_DIR / full_path
    if full_path and requested.is_file():
        return FileResponse(requested)
    index_path = config.FRONTEND_DIST_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend has not been built yet")
    return FileResponse(index_path)
