"""Download + cache the two HuggingFace positive sources to local JSONL.

HackAPrompt is large (~600k rows); TensorTrust is moderate. We stream, keep only
the fields we need, and cap rows so the build stays fast and reproducible.
"""

import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "hf"
OUT.mkdir(parents=True, exist_ok=True)

CAP = 20000  # generous cap; loaders sample down further


def dump(path: Path, rows):
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n >= CAP:
                break
    print(f"[ok] {path.name}: {n} rows")
    return n


# Canonical hackaprompt is gated; these public mirrors carry the same playground
# submissions (user_input / prompt + correctness flags).
HACKAPROMPT_CANDIDATES = [
    "hackaprompt/hackaprompt-dataset",
    "imoxto/prompt_injection_hackaprompt_gpt35",
    "reinforz/pi_hackaprompt_squad",
    "mxmh/hackaprompt",
]


def hackaprompt():
    path = OUT / "hackaprompt.jsonl"
    if path.exists():
        print("[skip] hackaprompt cached")
        return
    keep = ("user_input", "prompt", "text", "completion", "model_completion",
            "expected_completion", "correct", "level", "intention")
    last_err = None
    for hf_id in HACKAPROMPT_CANDIDATES:
        try:
            ds = load_dataset(hf_id, split="train", streaming=True)

            def rows():
                for r in ds:
                    rec = {k: r.get(k) for k in keep if k in r}
                    rec["_hf_id"] = hf_id
                    if any(rec.get(k) for k in ("user_input", "prompt", "text")):
                        yield rec
            n = dump(path, rows())
            if n:
                print(f"[ok] hackaprompt source = {hf_id}")
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[warn] {hf_id} unavailable: {str(e)[:120]}")
    print(f"[warn] hackaprompt: all mirrors failed ({last_err})")


def tensortrust():
    # Try a few known configs; the dataset exposes extraction + hijacking robustness sets.
    path = OUT / "tensortrust.jsonl"
    if path.exists():
        print("[skip] tensortrust cached")
        return
    last_err = None
    for cfg in (None,):
        try:
            ds = load_dataset("qxcv/tensor-trust", cfg) if cfg else load_dataset("qxcv/tensor-trust")
            written = 0
            for split in ds:
                with path.open("a", encoding="utf-8") as f:
                    for r in ds[split]:
                        rec = {"_split": split}
                        rec.update({k: v for k, v in r.items()
                                    if isinstance(v, (str, int, float, bool)) or v is None})
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                        if written >= CAP:
                            break
                if written >= CAP:
                    break
            print(f"[ok] tensortrust.jsonl: {written} rows (splits={list(ds.keys())})")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    print(f"[warn] tensortrust failed: {last_err}")


if __name__ == "__main__":
    hackaprompt()
    tensortrust()
    print("DONE_HF")
