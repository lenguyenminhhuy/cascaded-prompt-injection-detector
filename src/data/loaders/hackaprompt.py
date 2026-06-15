from collections.abc import Iterator

from datasets import load_dataset

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample


class HackapromptLoader(DatasetLoader):
    name = "hackaprompt"

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    def load(self) -> Iterator[Sample]:
        ds = load_dataset(
            "hackaprompt/hackaprompt-dataset",
            split="train",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        seen: set[str] = set()
        count = 0
        for row in ds:
            if not row.get("correct", False):
                continue
            text = (row.get("user_input") or "").strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            yield Sample(
                input=text,
                label="injection",
                channel="direct",
                source=self.name,
                metadata={"level": row.get("level"), "model": row.get("model")},
            )
            count += 1
            if self.max_samples and count >= self.max_samples:
                break
