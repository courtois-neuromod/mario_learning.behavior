#!/usr/bin/env bash
# Bootstrap mario_learning.behavior on a fresh clone.
#
# - Creates a uv-managed venv at .venv/
# - Installs the package + dependencies (pinned in pyproject.toml / uv.lock)
# - Generates config.yaml from the template by prompting for placeholder values.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "==> Creating venv with uv"
uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing project (editable) with dev extras"
uv pip install -e ".[dev]"

if [[ -f config.yaml ]]; then
    echo "==> config.yaml already exists; leaving as-is."
else
    echo "==> Generating config.yaml from template"
    read -r -p "Path to mario.scenes (canonical human dataset): " MARIO_SCENES_PATH
    read -r -p "SLURM username (leave blank if not using SLURM): " SLURM_USER
    SLURM_USER="${SLURM_USER:-none}"

    sed -e "s|<PLACEHOLDER_MARIO_SCENES_PATH>|${MARIO_SCENES_PATH}|" \
        -e "s|<PLACEHOLDER_SLURM_USER>|${SLURM_USER}|" \
        config.yaml.template > config.yaml

    echo "==> Wrote config.yaml"
fi

mkdir -p output logs

echo
echo "Done. Activate the venv with: source .venv/bin/activate"
echo "List tasks with:               inv --list"
