import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.preprocessing.struq_generator import generate_struq
from src.data.schema import Sample

logger = logging.getLogger(__name__)

LABEL = "injection"
CHANNEL = "app_structured"

_FALLBACK_INSTRUCTIONS: list[str] = [
    "Summarize the following:",
    "Translate to French:",
    "Answer the question based on the context:",
    "Extract the key facts from the text:",
    "Rewrite the paragraph in a formal tone:",
    "List the main points of the article:",
    "Explain the concept in simple terms:",
    "Classify the sentiment of the following review:",
    "Generate a brief title for the passage:",
    "Identify the named entities in the text:",
]


class StruQLoader(DatasetLoader):
    name = "struq"
    REPO_URL = "https://github.com/Sizhe-Chen/StruQ"

    def __init__(
        self,
        raw_dir: str = "data/raw",
        max_samples: int | None = None,
        split: str = "train",
    ):
        self.repo_dir = Path(raw_dir) / self.name
        self.max_samples = max_samples
        self.split = split

    def _ensure_clone(self) -> None:
        if self.repo_dir.exists() and any(self.repo_dir.iterdir()):
            return
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", self.REPO_URL, str(self.repo_dir)],
            check=True,
            capture_output=True,
        )

    def _collect_instructions(self) -> list[str]:
        """Walk the cloned repo and extract raw instruction strings."""
        instructions: list[str] = []

        candidate_files = sorted(
            list(self.repo_dir.rglob("*.json"))
            + list(self.repo_dir.rglob("*.jsonl"))
        )

        for path in candidate_files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    if path.suffix == ".jsonl":
                        lines = fh.readlines()
                        for lineno, line in enumerate(lines, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "StruQLoader: JSON decode error in %s line %d — skipping",
                                    path,
                                    lineno,
                                )
                                continue
                            self._extract_from_obj(obj, instructions)
                    else:
                        try:
                            data = json.load(fh)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "StruQLoader: could not parse %s — %s", path, exc
                            )
                            continue
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    self._extract_from_obj(item, instructions)
                                elif isinstance(item, str) and item.strip():
                                    instructions.append(item.strip())
                        elif isinstance(data, dict):
                            self._extract_from_obj(data, instructions)
            except OSError as exc:
                logger.warning("StruQLoader: could not read %s — %s", path, exc)
                continue

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for instr in instructions:
            if instr not in seen:
                seen.add(instr)
                unique.append(instr)
        return unique

    @staticmethod
    def _extract_from_obj(obj: dict, out: list[str]) -> None:
        """Pull instruction-like strings from a dict into *out*."""
        for key in ("instruction", "prompt", "input", "task", "text"):
            value = obj.get(key, "")
            if value and isinstance(value, str) and value.strip():
                out.append(value.strip())
                return  # One instruction per dict is enough.

    def _parse(self) -> Iterator[Sample]:
        instructions = self._collect_instructions()

        if not instructions:
            logger.warning(
                "StruQLoader: no usable instruction files found under %s — "
                "falling back to %d hardcoded instructions",
                self.repo_dir,
                len(_FALLBACK_INSTRUCTIONS),
            )
            instructions = _FALLBACK_INSTRUCTIONS

        samples = generate_struq(instructions, split=self.split)

        count = 0
        for sample in samples:
            yield sample
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                return

    def load(self) -> Iterator[Sample]:
        self._ensure_clone()
        yield from self._parse()
