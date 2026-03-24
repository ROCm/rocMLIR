#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNING_DATA="${1:?Usage: $0 /path/to/tuning-data [extra trainer args...]}"
shift

python3 "${SCRIPT_DIR}/trainClassifier.py" "$TUNING_DATA" --update "$@"
