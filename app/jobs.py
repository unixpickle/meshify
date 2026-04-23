from __future__ import annotations

import logging
import multiprocessing
import queue
import shutil
import threading
import traceback
from pathlib import Path
from typing import Any

from . import config, store
from .mesh_pipeline import PipelineSettings, RunDeletedError, StageSink, run_pipeline


logger = logging.getLogger(__name__)


class DatabaseEventSink:
    def __init__(self, run_id: str):
        self.run_id = run_id
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


class WorkerStageSink(StageSink):
    def __init__(self, run_id: str, result_queue):
        self.run_id = run_id
        self.result_queue = result_queue

    def stage(
        self,
        stage_key: str,
        *,
        status: str,
        progress: float,
        message: str | None = None,
    ) -> None:
        self.result_queue.put(
            {
                "type": "stage",
                "run_id": self.run_id,
                "stage_key": stage_key,
                "status": status,
                "progress": progress,
                "message": message,
            }
        )

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
        self.result_queue.put(
            {
                "type": "asset",
                "run_id": self.run_id,
                "stage_key": stage_key,
                "kind": kind,
                "label": label,
                "path": str(path),
                "mime_type": mime_type,
                "metadata": metadata or {},
            }
        )


def _serialize_settings(settings: PipelineSettings) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in settings.__dataclass_fields__:
        value = getattr(settings, field_name)
        payload[field_name] = str(value) if isinstance(value, Path) else value
    return payload


def _deserialize_settings(payload: dict[str, Any]) -> PipelineSettings:
    return PipelineSettings(
        image_path=Path(payload["image_path"]),
        output_path=Path(payload["output_path"]),
        remove_background=payload["remove_background"],
        disable_paint=payload["disable_paint"],
        model_path=payload["model_path"],
        subfolder=payload["subfolder"],
        texgen_model_path=payload["texgen_model_path"],
        texgen_subfolder=payload["texgen_subfolder"],
        variant=payload["variant"],
        steps=payload["steps"],
        texgen_delight_steps=payload["texgen_delight_steps"],
        texgen_multiview_steps=payload["texgen_multiview_steps"],
        octree_resolution=payload["octree_resolution"],
        num_chunks=payload["num_chunks"],
        seed=payload["seed"],
        device=payload["device"],
    )


def _worker_command_loop(command_queue, pending_runs, canceled_runs: set[str], state_lock, stop_event) -> None:
    while not stop_event.is_set():
        command = command_queue.get()
        command_type = command["type"]
        if command_type == "enqueue_run":
            pending_runs.put(command)
        elif command_type == "cancel_run":
            with state_lock:
                canceled_runs.add(command["run_id"])
        elif command_type == "shutdown":
            stop_event.set()
            pending_runs.put(None)
            return


def _worker_main(command_queue, result_queue) -> None:
    pending_runs: queue.Queue[dict[str, Any] | None] = queue.Queue()
    canceled_runs: set[str] = set()
    state_lock = threading.Lock()
    stop_event = threading.Event()
    command_thread = threading.Thread(
        target=_worker_command_loop,
        args=(command_queue, pending_runs, canceled_runs, state_lock, stop_event),
        daemon=True,
    )
    command_thread.start()

    def is_canceled(run_id: str) -> bool:
        with state_lock:
            return run_id in canceled_runs

    def ensure_active(run_id: str) -> None:
        if is_canceled(run_id):
            raise RunDeletedError(f"Run {run_id} was deleted")

    while True:
        job = pending_runs.get()
        if job is None:
            break
        run_id = job["run_id"]
        if is_canceled(run_id):
            continue

        result_queue.put(
            {
                "type": "run_started",
                "run_id": run_id,
                "current_stage": job["current_stage"],
                "progress": job["progress"],
                "message": job["message"],
            }
        )

        try:
            run_pipeline(
                _deserialize_settings(job["settings"]),
                WorkerStageSink(run_id, result_queue),
                previous_run=job["previous_run"],
                ensure_active=lambda run_id=run_id: ensure_active(run_id),
            )
        except RunDeletedError:
            result_queue.put({"type": "run_deleted", "run_id": run_id})
        except Exception as exc:  # pragma: no cover - defensive background handling
            result_queue.put(
                {
                    "type": "run_failed",
                    "run_id": run_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            result_queue.put({"type": "run_completed", "run_id": run_id})


class JobManager:
    def __init__(self) -> None:
        self.started = False
        self.lock = threading.Lock()
        self.active_runs: set[str] = set()
        self.pending_runs: set[str] = set()
        self.cancel_requested: set[str] = set()
        self._event_sinks: dict[str, DatabaseEventSink] = {}
        self._context: multiprocessing.context.BaseContext | None = None
        self._command_queue = None
        self._result_queue = None
        self._process: multiprocessing.process.BaseProcess | None = None
        self._result_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self._context = multiprocessing.get_context("spawn")
        self._command_queue = self._context.Queue()
        self._result_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_main,
            args=(self._command_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()
        self._result_thread = threading.Thread(target=self._consume_results, daemon=True)
        self._result_thread.start()
        for run_id in store.recover_incomplete_runs():
            self.submit(run_id, recovered=True)

    def stop(self) -> None:
        if not self.started:
            return
        if self._command_queue is not None:
            self._command_queue.put({"type": "shutdown"})
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
        self.started = False

    def submit(self, run_id: str, *, recovered: bool = False) -> None:
        payload = self._build_run_payload(run_id)
        if payload is None:
            return
        if not recovered:
            uploaded_progress = 1 / len(store.STAGE_DEFINITIONS)
            store.update_run(
                run_id,
                status="queued",
                current_stage="uploaded",
                progress=uploaded_progress,
                message="Queued",
            )
        if self._command_queue is None:
            raise RuntimeError("Job manager has not been started")
        with self.lock:
            self.pending_runs.add(run_id)
        self._command_queue.put(payload)

    def request_delete(self, run_id: str) -> bool:
        if self._command_queue is not None:
            self._command_queue.put({"type": "cancel_run", "run_id": run_id})
        with self.lock:
            is_active = run_id in self.active_runs
            is_pending = run_id in self.pending_runs
            self.cancel_requested.add(run_id)
        if is_active or is_pending:
            store.mark_run_deleting(run_id)
            return True
        self._cleanup_run(run_id)
        return False

    def _build_run_payload(self, run_id: str) -> dict[str, Any] | None:
        run = store.load_run(run_id, include_deleting=True)
        if run is None:
            return None
        upload_asset = next((asset for asset in run["assets"] if asset["stage_key"] == "uploaded"), None)
        if upload_asset is None:
            raise RuntimeError("Missing uploaded source image")

        output_dir = config.RUNS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "textured_mesh.glb"
        settings = PipelineSettings(
            image_path=config.STORAGE_DIR / upload_asset["storage_path"],
            output_path=output_path,
            remove_background=not run["settings"].get("keep_background", False),
            disable_paint=bool(run["settings"].get("disable_paint", False)),
        )
        return {
            "type": "enqueue_run",
            "run_id": run_id,
            "settings": _serialize_settings(settings),
            "previous_run": run,
            "current_stage": run["current_stage"],
            "progress": run["progress"],
            "message": "Resuming" if recovered_status(run) == "running" else "Starting",
        }

    def _consume_results(self) -> None:
        if self._result_queue is None:
            return
        while True:
            try:
                event = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    return
                continue
            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        run_id = event["run_id"]
        run = store.load_run(run_id, include_deleting=True)
        if run is None and event_type != "run_deleted":
            with self.lock:
                self.active_runs.discard(run_id)
                self.pending_runs.discard(run_id)
                self.cancel_requested.discard(run_id)
                self._event_sinks.pop(run_id, None)
            return
        sink = self._event_sinks.setdefault(run_id, DatabaseEventSink(run_id))
        if event_type == "run_started":
            with self.lock:
                self.active_runs.add(run_id)
                self.pending_runs.discard(run_id)
            store.update_run(
                run_id,
                status="running",
                current_stage=event["current_stage"],
                progress=event["progress"],
                message=event["message"],
                started=True,
            )
            return

        if event_type == "stage":
            sink.stage(
                event["stage_key"],
                status=event["status"],
                progress=event["progress"],
                message=event.get("message"),
            )
            return

        if event_type == "asset":
            sink.asset(
                event["stage_key"],
                kind=event["kind"],
                label=event["label"],
                path=Path(event["path"]),
                mime_type=event["mime_type"],
                metadata=event.get("metadata"),
            )
            return

        with self.lock:
            self.active_runs.discard(run_id)
            self.pending_runs.discard(run_id)

        if event_type == "run_completed":
            with self.lock:
                self.cancel_requested.discard(run_id)
                self._event_sinks.pop(run_id, None)
            return

        if event_type == "run_deleted":
            with self.lock:
                self._event_sinks.pop(run_id, None)
            self._cleanup_run(run_id)
            return

        if event_type == "run_failed":
            logger.error(
                "run=%s failed in worker: %s\n%s",
                run_id,
                event["error"],
                event.get("traceback", ""),
            )
            if run is not None:
                store.update_run(
                    run_id,
                    status="failed",
                    error=event["error"],
                    message=event["error"],
                    progress=0.0,
                )
            with self.lock:
                self.cancel_requested.discard(run_id)
                self._event_sinks.pop(run_id, None)
            return

        logger.warning("Unhandled worker event type: %s", event_type)

    def _cleanup_run(self, run_id: str) -> None:
        with self.lock:
            self.cancel_requested.add(run_id)
        shutil.rmtree(config.UPLOADS_DIR / run_id, ignore_errors=True)
        shutil.rmtree(config.RUNS_DIR / run_id, ignore_errors=True)
        store.delete_run(run_id)
        with self.lock:
            self.cancel_requested.discard(run_id)


def recovered_status(run: dict[str, Any]) -> str:
    status = run.get("status")
    return status if isinstance(status, str) else ""


job_manager = JobManager()
