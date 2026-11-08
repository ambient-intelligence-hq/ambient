"""Vendor vlm.fyi's capability data into ambient/artifacts/capabilities.json.

    uv run python scripts/sync_capabilities.py --from ../vlm-fyi/data/capabilities.json
    uv run python scripts/sync_capabilities.py --from https://vlm.fyi/v1/capabilities.json

The engine reads only the vendored copy (no network at runtime). Refresh it in a
reviewed PR so limit changes are visible in the diff.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "ambient" / "artifacts" / "capabilities.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="path or URL of capabilities.json")
    args = ap.parse_args()
    if args.src.startswith("http"):
        with urllib.request.urlopen(args.src, timeout=30) as r:
            data = json.load(r)
    else:
        data = json.loads(Path(args.src).read_text())
    assert data.get("version") == 1 and isinstance(data.get("entries"), list), "not a vlm.fyi capabilities v1 file"
    old = json.loads(DEST.read_text()) if DEST.exists() else {"entries": [], "updated": None}
    DEST.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"{DEST.relative_to(DEST.parents[2])}: {len(old['entries'])} -> {len(data['entries'])} entries, "
          f"updated {old.get('updated')} -> {data.get('updated')}")


if __name__ == "__main__":
    main()
