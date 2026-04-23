from __future__ import annotations

import logging
import queue
import shutil
import threading
from pathlib import Path

from . import config, store
from .mesh_pipeline import PipelineSettings, RunDeletedError, StageSink, run_pipeline


logger = logging.getLogger(__name__)


class DatabaseStageSink(StageSink):
    def __init__(self, run_id: str, ensure_active):
        self.run_id = run_id
        self._ensure_active = ensure_active
        self._last_logged_progress: dict[str, int] = {}
        self._last_logged_message: dict[str, str | None] = {}
        self._last_logged_status: dict[str, str] = {}

    def stage(
        self,
        stage_key: str,
        *,
        status: str,
        progress: float,
        message: str | None = None,
    ) -> None:
        self._ensure_active()
        stage_index = store.STAGE_ORDER[stage_key]
        stage_count = len(store.STAGE_DEFINITIONS)
        overall_progress = min(((stage_index - 1) + progress) / stage_count, 1.0)
        store.upsert_stage(
            self.run_id,
            stage_key,
            status=status,
            progress=progress,
            message=message,
            started=status == "running",
            completed=status == "completed",
        )
        run_status = "running"
        completed = False
        if status == "completed" and stage_key == "export":
            run_status = "completed"
            completed = True
        elif status == "failed":
            run_status = "failed"
        store.update_run(
            self.run_id,
            status=run_status,
            current_stage=stage_key,
            progress=1.0 if completed else overall_progress,
            message=message,
            started=True,
            completed=completed,
        )
        progress_bucket = int(progress * 100)
        should_log = (
            self._last_logged_status.get(stage_key) != status
            or self._last_logged_progress.get(stage_key) != progress_bucket
            or self._last_logged_message.get(stage_key) != message
            or status in {"completed", "failed"}
        )
        if should_log:
            logger.info(
                "run=%s stage=%s status=%s progress=%s%% message=%s",
                self.run_id,
                stage_key,
                status,
                progress_bucket,
                message or "",
            )
            self._last_logged_status[stage_key] = status
            self._last_logged_progress[stage_key] = progress_bucket
            self._last_logged_message[stage_key] = message

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
        self._ensure_active()
        relative_path = path.relative_to(config.STORAGE_DIR)
        store.create_asset(
            self.run_id,
            stage_key,
            kind=kind,
            label=label,
            storage_path=relative_path,
            mime_type=mime_type,
            metadata=metadata,
        )


class JobManager:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.started = False
        self.lock = threading.Lock()
        self.active_runs: set[str] = set()
        self.cancel_requested: set[str] = set()

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread.start()
        for run_id in store.recover_incomplete_runs():
            self.queue.put(run_id)

    def submit(self, run_id: str) -> None:
        uploaded_progress = 1 / len(store.STAGE_DEFINITIONS)
        store.update_run(
            run_id,
            status="queued",
            current_stage="uploaded",
            progress=uploaded_progress,
            message="Queued",
        )
        self.queue.put(run_id)

    def ensure_active(self, run_id: str) -> None:
        with self.lock:
            if run_id in self.cancel_requested:
                raise RunDeletedError(f"Run {run_id} was deleted")

    def request_delete(self, run_id: str) -> bool:
        with self.lock:
            is_active = run_id in self.active_runs
            self.cancel_requested.add(run_id)
        if is_active:
            store.mark_run_deleting(run_id)
            return True
        self._cleanup_run(run_id)
        return False

    def _cleanup_run(self, run_id: str) -> None:
        with self.lock:
            self.cancel_requested.add(run_id)
        shutil.rmtree(config.UPLOADS_DIR / run_id, ignore_errors=True)
        shutil.rmtree(config.RUNS_DIR / run_id, ignore_errors=True)
        store.delete_run(run_id)
        with self.lock:
            self.cancel_requested.discard(run_id)

    def _worker(self) -> None:
        while True:
            run_id = self.queue.get()
            try:
                self._process(run_id)
            except RunDeletedError:
                logger.info("run=%s deleted during processing", run_id)
                self._cleanup_run(run_id)
            except Exception as exc:  # pragma: no cover - defensive background handling
                run = store.load_run(run_id, include_deleting=True)
                if run is not None:
                    store.update_run(
                        run_id,
                        status="failed",
                        error=str(exc),
                        message=str(exc),
                        progress=0.0,
                    )
            finally:
                with self.lock:
                    self.active_runs.discard(run_id)
                self.queue.task_done()

    def _process(self, run_id: str) -> None:
        with self.lock:
            self.active_runs.add(run_id)
        run = store.load_run(run_id)
        if run is None:
            return
        upload_asset = next((asset for asset in run["assets"] if asset["stage_key"] == "uploaded"), None)
        if upload_asset is None:
            raise RuntimeError("Missing uploaded source image")

        output_dir = config.RUNS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "textured_mesh.glb"

        sink = DatabaseStageSink(run_id, lambda: self.ensure_active(run_id))
        settings = PipelineSettings(
            image_path=config.STORAGE_DIR / upload_asset["storage_path"],
            output_path=output_path,
            remove_background=not run["settings"].get("keep_background", False),
            disable_paint=bool(run["settings"].get("disable_paint", False)),
        )
        store.update_run(
            run_id,
            status="running",
            current_stage=run["current_stage"],
            progress=run["progress"],
            message="Resuming" if run["status"] == "running" else "Starting",
            started=True,
        )
        run_pipeline(settings, sink, previous_run=run, ensure_active=lambda: self.ensure_active(run_id))


job_manager = JobManager()
