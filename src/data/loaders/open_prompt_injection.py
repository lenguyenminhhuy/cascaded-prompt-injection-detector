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


class OpenPromptInjectionLoader(DatasetLoader):
    name = "open_prompt_injection"
    REPO_URL = "https://github.com/liu00222/Open-Prompt-Injection"

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
        for key in ("injected_prompt", "attack_str", "injection", "text"):
            value = row.get(key, "")
            if value and isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _rows_from_file(self, path: Path) -> list[dict]:
        """Load a JSON file and normalise to a list of dicts."""
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "OpenPromptInjectionLoader: could not read %s — %s", path, exc
            )
            return []

        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        logger.warning(
            "OpenPromptInjectionLoader: unexpected top-level type %s in %s — skipping",
            type(data).__name__,
            path,
        )
        return []

    def _parse(self) -> Iterator[Sample]:
        json_files = sorted(self.repo_dir.rglob("*.json"))
        if not json_files:
            logger.warning(
                "OpenPromptInjectionLoader: no .json files found under %s",
                self.repo_dir,
            )
            return

        count = 0
        for path in json_files:
            fname = str(path.relative_to(self.repo_dir))
            for row in self._rows_from_file(path):
                text = self._extract_text(row)
                if not text:
                    continue

                yield Sample(
                    input=text,
                    label=LABEL,
                    channel=CHANNEL,
                    source=self.name,
                    metadata={"task": row.get("task", ""), "source_file": fname},
                )
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return

    def load(self) -> Iterator[Sample]:
        self._ensure_clone()
        yield from self._parse()
