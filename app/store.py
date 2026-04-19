from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import get_connection
from .events import event_broker


DB_LOCK = threading.Lock()

STAGE_DEFINITIONS: list[tuple[str, str]] = [
    ("uploaded", "Uploaded"),
    ("preprocess", "Background Removal"),
    ("model_load", "Model Load"),
    ("diffusion", "Diffusion Sampling"),
    ("volume_decode", "Volume Decode"),
    ("mesh_export", "White Mesh Export"),
    ("texture_model_load", "Texture Model Load"),
    ("delight", "Light Cleanup"),
    ("uv_unwrap", "UV Unwrap"),
    ("multiview", "Multiview Paint"),
    ("texture_bake", "Texture Bake"),
    ("export", "Textured Mesh Export"),
]
STAGE_ORDER = {key: index for index, (key, _) in enumerate(STAGE_DEFINITIONS, start=1)}
STAGE_LABELS = {key: label for key, label in STAGE_DEFINITIONS}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def create_run(original_name: str, settings: dict[str, Any]) -> str:
    run_id = uuid.uuid4().hex
    now = utc_now()
    with DB_LOCK, get_connection() as connection:
        connection.execute(
            """
            INSERT INTO runs (
                id, original_name, status, current_stage, progress, message, error,
                created_at, updated_at, started_at, completed_at, settings_json
            ) VALUES (?, ?, 'queued', 'uploaded', 0, ?, NULL, ?, ?, NULL, NULL, ?)
            """,
            (run_id, original_name, "Waiting to start", now, now, json.dumps(settings)),
        )
    event_broker.publish("run.created", run_id)
    return run_id


def initialize_run_stages(run_id: str) -> None:
    now = utc_now()
    with DB_LOCK, get_connection() as connection:
        for key, label in STAGE_DEFINITIONS:
            connection.execute(
                """
                INSERT OR IGNORE INTO stages (
                    run_id, stage_key, stage_label, stage_order, status, progress,
                    message, started_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, NULL, ?, NULL)
                """,
                (run_id, key, label, STAGE_ORDER[key], now),
            )
    event_broker.publish("stages.initialized", run_id)


def update_run(
    run_id: str,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
    started: bool = False,
    completed: bool = False,
) -> None:
    now = utc_now()
    with DB_LOCK, get_connection() as connection:
        current = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if current is None:
            raise KeyError(f"Run {run_id} not found")

        next_status = status or current["status"]
        next_stage = current_stage or current["current_stage"]
        next_progress = progress if progress is not None else current["progress"]
        next_message = message if message is not None else current["message"]
        next_error = error if error is not None else current["error"]
        started_at = current["started_at"] or (now if started else None)
        completed_at = now if completed else current["completed_at"]

        connection.execute(
            """
            UPDATE runs
            SET status = ?, current_stage = ?, progress = ?, message = ?, error = ?,
                updated_at = ?, started_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                next_status,
                next_stage,
                next_progress,
                next_message,
                next_error,
                now,
                started_at,
                completed_at,
                run_id,
            ),
        )
    event_broker.publish("run.updated", run_id)


def upsert_stage(
    run_id: str,
    stage_key: str,
    *,
    status: str,
    progress: float = 0,
    message: str | None = None,
    started: bool = False,
    completed: bool = False,
) -> None:
    now = utc_now()
    label = STAGE_LABELS[stage_key]
    order = STAGE_ORDER[stage_key]
    with DB_LOCK, get_connection() as connection:
        existing = connection.execute(
            "SELECT * FROM stages WHERE run_id = ? AND stage_key = ?",
            (run_id, stage_key),
        ).fetchone()
        started_at = existing["started_at"] if existing else None
        completed_at = existing["completed_at"] if existing else None
        if started and started_at is None:
            started_at = now
        if completed and completed_at is None:
            completed_at = now
        connection.execute(
            """
            INSERT INTO stages (
                run_id, stage_key, stage_label, stage_order, status, progress,
                message, started_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, stage_key) DO UPDATE SET
                stage_label = excluded.stage_label,
                stage_order = excluded.stage_order,
                status = excluded.status,
                progress = excluded.progress,
                message = excluded.message,
                started_at = excluded.started_at,
                updated_at = excluded.updated_at,
                completed_at = excluded.completed_at
            """,
            (
                run_id,
                stage_key,
                label,
                order,
                status,
                progress,
                message,
                started_at,
                now,
                completed_at,
            ),
        )
    event_broker.publish("stage.updated", run_id)


def create_asset(
    run_id: str,
    stage_key: str,
    *,
    kind: str,
    label: str,
    storage_path: Path,
    mime_type: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    now = utc_now()
    with DB_LOCK, get_connection() as connection:
        storage_path_text = storage_path.as_posix()
        existing = connection.execute(
            """
            SELECT id FROM assets
            WHERE run_id = ? AND stage_key = ? AND storage_path = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id, stage_key, storage_path_text),
        ).fetchone()
        if existing is None:
            asset_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO assets (
                    id, run_id, stage_key, kind, label, storage_path, mime_type, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    run_id,
                    stage_key,
                    kind,
                    label,
                    storage_path_text,
                    mime_type,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
        else:
            asset_id = existing["id"]
            connection.execute(
                """
                UPDATE assets
                SET kind = ?, label = ?, mime_type = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    kind,
                    label,
                    mime_type,
                    json.dumps(metadata or {}),
                    asset_id,
                ),
            )
    event_broker.publish("asset.created", run_id)
    return asset_id


def recover_incomplete_runs() -> list[str]:
    with DB_LOCK, get_connection() as connection:
        rows = connection.execute(
            "SELECT id FROM runs WHERE status IN ('queued', 'running') ORDER BY created_at ASC"
        ).fetchall()
    return [row["id"] for row in rows]


def load_run(run_id: str) -> dict[str, Any] | None:
    with DB_LOCK, get_connection() as connection:
        run_row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run_row is None:
            return None
        stage_rows = connection.execute(
            "SELECT * FROM stages WHERE run_id = ? ORDER BY stage_order ASC",
            (run_id,),
        ).fetchall()
        asset_rows = connection.execute(
            "SELECT * FROM assets WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()

    run = _row_to_dict(run_row)
    run["settings"] = json.loads(run.pop("settings_json"))
    run["stages"] = [_row_to_dict(row) for row in stage_rows]
    run["assets"] = []
    deduped_assets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in asset_rows:
        item = _row_to_dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        item["url"] = f"/files/{item['storage_path']}"
        deduped_assets[(item["stage_key"], item["storage_path"])] = item
    run["assets"] = list(deduped_assets.values())

    image_assets = [asset for asset in run["assets"] if asset["kind"] == "image"]
    model_assets = [asset for asset in run["assets"] if asset["kind"] == "model"]
    run["preview_image_url"] = image_assets[0]["url"] if image_assets else None
    run["final_model_url"] = model_assets[-1]["url"] if model_assets else None
    return run


def list_runs() -> list[dict[str, Any]]:
    with DB_LOCK, get_connection() as connection:
        rows = connection.execute("SELECT id FROM runs ORDER BY created_at DESC").fetchall()
    runs: list[dict[str, Any]] = []
    for row in rows:
        run = load_run(row["id"])
        if run is not None:
            runs.append(run)
    return runs
