#!/usr/bin/env bash
# Refresh the vendored code from the two source repos.
#   ./sync.sh /path/to/day-trader-skill
# Kronos's model/ comes from this repo's own ../model.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DTS="${1:?usage: ./sync.sh /path/to/day-trader-skill}"
rm -rf "$HERE/model" && cp -r "$HERE/../model" "$HERE/model" && cp "$HERE/../LICENSE" "$HERE/model/LICENSE"
rm -rf "$HERE/model/__pycache__"
cp "$DTS/trader-entry/entry.py" "$DTS/trader-exit/exit.py" "$DTS/trader-budget/size.py" "$HERE/trader/"
cp "$DTS/LICENSE" "$HERE/trader/LICENSE"
echo "synced model/ and trader/"
