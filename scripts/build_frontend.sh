#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NODE_BIN="${NODE_BIN:-}"
if [[ -z "$NODE_BIN" ]]; then
  if command -v node >/dev/null 2>&1; then
    NODE_BIN="$(command -v node)"
  else
    echo "Could not find 'node' on your PATH." >&2
    echo "Install Node.js or set NODE_BIN=/path/to/node and rerun." >&2
    exit 1
  fi
fi

if [[ ! -f "node_modules/typescript/lib/tsc.js" ]]; then
  echo "Missing frontend dependencies." >&2
  echo "Run 'npm install' from the project root, then rerun this script." >&2
  exit 1
fi

"$NODE_BIN" node_modules/typescript/lib/tsc.js -p frontend/tsconfig.json
cp frontend/index.html frontend/styles.css frontend/dist/
