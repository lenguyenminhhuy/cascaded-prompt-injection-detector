"""Freeze cascade-pid dataset v1 (PLAN.md Chunk 7.4).

Computes a content hash over the split files + key reports, writes a freeze
manifest (data/VERSION.json), and logs a W&B artifact if W&B is available and
configured. Without W&B (not installed, or no WANDB_API_KEY) it falls back to a
local freeze so the build stays reproducible offline.

Usage:
  python scripts/freeze_dataset.py --version v1
  WANDB_API_KEY=... python scripts/freeze_dataset.py --version v1   # logs artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLIT_NAMES = ["train", "cal", "test_in_dist", "test_cross_channel", "test_cross_domain"]
ARTIFACT_FILES = [
    "splits/train.jsonl", "splits/cal.jsonl", "splits/test_in_dist.jsonl",
    "splits/test_cross_channel.jsonl", "splits/test_cross_domain.jsonl",
    "splits/manifest.json", "coverage_report.json",
    "audit/balance_report.json", "audit/structural_audit.json",
    "audit/leakage_report.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--project", default="cascade-pid")
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    files = {f: sha256(DATA / f) for f in ARTIFACT_FILES if (DATA / f).exists()}
    counts = {}
    for n in SPLIT_NAMES:
        p = DATA / "splits" / f"{n}.jsonl"
        counts[n] = sum(1 for _ in p.open()) if p.exists() else 0

    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()
    ).hexdigest()[:16]

    manifest = {
        "version": args.version,
        "content_hash": combined,
        "split_counts": counts,
        "total": sum(counts.values()),
        "file_hashes": files,
        "schema": "SCHEMA.md v1",
    }

    # W&B artifact logging (optional)
    wandb_status = "skipped (wandb not installed)"
    try:
        import wandb  # noqa: PLC0415

        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        run = wandb.init(project=args.project, entity=args.entity,
                         job_type="data-prep", group="data-prep",
                         mode=mode, config=manifest)
        art = wandb.Artifact(f"cascade-pid-dataset", type="dataset",
                             metadata=manifest)
        for f in files:
            art.add_file(str(DATA / f), name=f)
        run.log_artifact(art, aliases=[args.version, "latest"])
        run.finish()
        wandb_status = f"logged ({mode})"
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        wandb_status = f"failed: {str(e)[:120]}"

    manifest["wandb"] = wandb_status
    (DATA / "VERSION.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"version": args.version, "content_hash": combined,
                      "total": manifest["total"], "wandb": wandb_status}, indent=2))


if __name__ == "__main__":
    main()
