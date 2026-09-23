#!/usr/bin/env bash
# Download all raw datasets into data/raw/.
# HuggingFace datasets are pulled lazily by their loaders; this script handles GitHub clones.

set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

clone_if_missing() {
    local repo_url="$1"
    local target="$2"
    if [ -d "$target/.git" ]; then
        echo "[skip] $target already cloned"
    else
        echo "[clone] $repo_url -> $target"
        git clone --depth 1 "$repo_url" "$target"
    fi
}

clone_if_missing https://github.com/microsoft/BIPIA "$RAW_DIR/BIPIA"
clone_if_missing https://github.com/uiuc-kang-lab/InjecAgent "$RAW_DIR/InjecAgent"
clone_if_missing https://github.com/liu00222/Open-Prompt-Injection "$RAW_DIR/Open-Prompt-Injection"
clone_if_missing https://github.com/Sizhe-Chen/StruQ "$RAW_DIR/StruQ"

echo "[ok] github datasets ready. HF datasets are streamed on demand."
