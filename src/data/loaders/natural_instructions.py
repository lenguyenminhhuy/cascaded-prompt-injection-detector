from collections.abc import Iterator

from datasets import load_dataset

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample


class NaturalInstructionsLoader(DatasetLoader):
    name = "natural_instructions"

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    def load(self) -> Iterator[Sample]:
        ds = load_dataset(
            "Muennighoff/natural-instructions",
            split="train",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        count = 0
        for row in ds:
            text = self._extract_input(row)
            if not text or not text.strip():
                continue
            yield Sample(
                input=text.strip(),
                label="benign",
                channel="app_structured",
                source=self.name,
                metadata=self._extract_metadata(row),
            )
            count += 1
            if self.max_samples and count >= self.max_samples:
                break

    def _extract_input(self, row) -> str:
        # Primary schema variant: "inputs" field
        if "inputs" in row and row["inputs"]:
            return row["inputs"]
        # Alternate schema variant: "Definition" field (list or string)
        definition = row.get("Definition")
        if definition:
            if isinstance(definition, list) and definition:
                return definition[0]
            if isinstance(definition, str):
                return definition
        return ""

    def _extract_metadata(self, row) -> dict:
        return {}
