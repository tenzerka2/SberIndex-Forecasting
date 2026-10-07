#!/usr/bin/env bash
# Current audit and improvements. Historical outputs are preserved, not silently regenerated.
set -euo pipefail
cd "$(dirname "$0")"
exec bash reproduce_research.sh
