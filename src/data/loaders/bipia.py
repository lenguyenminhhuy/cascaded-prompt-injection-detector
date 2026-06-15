import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample

logger = logging.getLogger(__name__)

LABEL = "injection"
CHANNEL = "document_embedded"


class BipiaLoader(DatasetLoader):
    name = "bipia"
    REPO_URL = "https://github.com/microsoft/BIPIA"

    def __init__(self, raw_dir: str = "data/raw", max_samples: int | None = None):
        self.repo_dir = Path(raw_dir) / self.name
        self.max_samples = max_samples

    def _ensure_clone(self) -> None:
        if self.repo_dir.exists() and any(self.repo_dir.iterdir()):
            return
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", self.REPO_URL, str(self.repo_dir)],
            check=True,
            capture_output=True,
        )

    def _extract_text(self, row: dict) -> str:
        for key in ("attack", "injected_prompt", "text"):
            value = row.get(key, "")
            if value and isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _parse(self) -> Iterator[Sample]:
        jsonl_files = sorted(self.repo_dir.rglob("*.jsonl"))
        if not jsonl_files:
            logger.warning("BipiaLoader: no .jsonl files found under %s", self.repo_dir)
            return

        count = 0
        for path in jsonl_files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "BipiaLoader: JSON decode error in %s line %d — skipping",
                                path,
                                lineno,
                            )
                            continue

                        text = self._extract_text(row)
                        if not text:
                            continue

                        task = row.get("task_name", row.get("task", ""))
                        yield Sample(
                            input=text,
                            label=LABEL,
                            channel=CHANNEL,
                            source=self.name,
                            metadata={"attack_success": True, "task": task},
                        )
                        count += 1
                        if self.max_samples is not None and count >= self.max_samples:
                            return
            except OSError as exc:
                logger.warning("BipiaLoader: could not read %s — %s", path, exc)
                continue

    def load(self) -> Iterator[Sample]:
        self._ensure_clone()
        yield from self._parse()
