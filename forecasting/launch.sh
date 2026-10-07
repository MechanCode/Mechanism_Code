#!/usr/bin/env bash
# Compatibility entry point. Configure all paths in scripts/launch_common.sh.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_ROOT/scripts/launch_common.sh" forecasting "$@"
