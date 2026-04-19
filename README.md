# Meshify Control Room

Single-user web UI for uploading images, running the Hunyuan 3D mesh pipeline, tracking job progress, and reviewing stage outputs and generated `.glb` meshes.

## Stack

- Frontend: TypeScript browser app in `frontend/src`
- Backend: FastAPI app in `app/main.py`
- Pipeline: shared Hunyuan runner in `app/mesh_pipeline.py`
- Database: local SQLite database in `data/meshify.sqlite3`
- Assets: uploaded and generated files in `data/storage`

## Prerequisites

- macOS or Linux
- Python `3.12`
- `uv`
- A Hugging Face token with access to the required model files
- A local checkout of Tencent Hunyuan 3D at `Hunyuan3D-2/`

Clone the upstream dependency into this folder name before syncing Python dependencies:

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git Hunyuan3D-2
```

## Setup

Create and populate the virtual environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync
npm install
```

Set the runtime environment variables in your shell:

```bash
export HF_TOKEN=your_token_here
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
```

Notes:

- `HF_HUB_DISABLE_XET=1` is important here. The full `Hunyuan3D-2.1` download failed with `416 Range Not Satisfiable` without it in this workspace.
- The frontend build now expects `node_modules` to exist locally, so run `npm install` once after cloning.
- If `node` is not on your `PATH`, the build script can still fall back to the Node runtime bundled with the Codex desktop app.

## Build The Frontend

```bash
scripts/build_frontend.sh
```

This compiles `frontend/src` into `frontend/dist`.

## Run The Web App

For local-only access:

```bash
source .venv/bin/activate
export HF_TOKEN=your_token_here
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
scripts/build_frontend.sh
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

For access from other machines on your VPN:

```bash
source .venv/bin/activate
export HF_TOKEN=your_token_here
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
scripts/build_frontend.sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://<this-machine-vpn-ip>:8000
```

If you prefer a local helper script, keep it in a gitignored file such as `serve.sh` rather than committing tokens into the repo.

## Run The CLI Directly

The original command-line entrypoint still works and uses the same shared pipeline code:

```bash
source .venv/bin/activate
export HF_TOKEN=your_token_here
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
python run_hunyuan3d_watch.py
```

## What The App Does

- Queues an image upload immediately
- Stores run state, stages, and assets in SQLite
- Processes heavy mesh jobs through a single background worker
- Shows the full stage timeline for each run:
  `Uploaded`, `Background Removal`, `Model Load`, `Diffusion Sampling`, `Volume Decode`, `Mesh Export`
- Displays stage outputs including images and the final `glb`
- Uses server-driven long polling so clients update on status changes without fixed-interval refreshes

## Important Paths

- FastAPI entrypoint: `app/main.py`
- Job queue worker: `app/jobs.py`
- Pipeline implementation: `app/mesh_pipeline.py`
- Data layer: `app/store.py`
- Event broker: `app/events.py`
- Frontend app: `frontend/src/main.ts`
- Frontend API client: `frontend/src/api.ts`
- Frontend styles: `frontend/styles.css`
- Frontend build script: `scripts/build_frontend.sh`

## Repo Hygiene

This repo intentionally does not commit:

- tokens or local helper scripts containing secrets
- model cache downloads
- virtualenvs and package caches
- `node_modules`
- SQLite runtime state
- uploaded/generated assets
- the local `Hunyuan3D-2/` checkout
- vendored TypeScript tool binaries

That means a fresh clone should follow the setup steps above before the app is runnable.
