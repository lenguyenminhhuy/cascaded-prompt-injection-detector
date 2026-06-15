from collections.abc import Iterator

from datasets import load_dataset

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample


class WildguardLoader(DatasetLoader):
    name = "wildguard"

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    def load(self) -> Iterator[Sample]:
        ds = load_dataset(
            "allenai/wildguardmix",
            split="test",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        count = 0
        for row in ds:
            harm_label = (row.get("prompt_harm_label") or "").lower()
            is_adversarial = row.get("adversarial", False)
            if harm_label != "harmful" or not is_adversarial:
                continue
            text = (row.get("prompt") or "").strip()
            if not text:
                continue
            yield Sample(
                input=text,
                label="injection",
                channel="direct",
                source=self.name,
                metadata={"harm_label": harm_label},
            )
            count += 1
            if self.max_samples and count >= self.max_samples:
                break
