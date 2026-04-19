from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = DATA_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RUNS_DIR = STORAGE_DIR / "runs"
DB_PATH = DATA_DIR / "meshify.sqlite3"

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

HF_HOME = Path(os.environ.get("HF_HOME", BASE_DIR / ".hf")).resolve()
HY3DGEN_MODELS = Path(os.environ.get("HY3DGEN_MODELS", BASE_DIR / ".models")).resolve()
DEFAULT_DEVICE = os.environ.get("MESHIFY_DEVICE")

MODEL_PATH = os.environ.get("MESHIFY_MODEL_PATH", "tencent/Hunyuan3D-2.1")
MODEL_SUBFOLDER = os.environ.get("MESHIFY_MODEL_SUBFOLDER", "hunyuan3d-dit-v2-1")
MODEL_VARIANT = os.environ.get("MESHIFY_MODEL_VARIANT", "auto")
TEXGEN_MODEL_PATH = os.environ.get("MESHIFY_TEXGEN_MODEL_PATH", "tencent/Hunyuan3D-2")
TEXGEN_SUBFOLDER = os.environ.get("MESHIFY_TEXGEN_SUBFOLDER", "hunyuan3d-paint-v2-0-turbo")
TEXGEN_DELIGHT_STEPS = int(os.environ.get("MESHIFY_TEXGEN_DELIGHT_STEPS", "50"))
TEXGEN_MULTIVIEW_STEPS = int(os.environ.get("MESHIFY_TEXGEN_MULTIVIEW_STEPS", "30"))
DEFAULT_STEPS = int(os.environ.get("MESHIFY_STEPS", "30"))
DEFAULT_OCTREE_RESOLUTION = int(os.environ.get("MESHIFY_OCTREE_RESOLUTION", "256"))
DEFAULT_NUM_CHUNKS = int(os.environ.get("MESHIFY_NUM_CHUNKS", "4000"))
DEFAULT_SEED = int(os.environ.get("MESHIFY_SEED", "12345"))


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIST_DIR.mkdir(parents=True, exist_ok=True)
