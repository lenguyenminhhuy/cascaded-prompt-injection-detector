from collections.abc import Iterator

from datasets import load_dataset

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample


class LmsysChatLoader(DatasetLoader):
    name = "lmsys_chat"

    def __init__(self, max_samples: int | None = None, cache_dir: str | None = None):
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    def load(self) -> Iterator[Sample]:
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        count = 0
        for row in ds:
            text = self._first_human_turn(row)
            if not text:
                continue
            yield Sample(
                input=text,
                label="benign",
                channel="conversational",
                source=self.name,
                metadata={"language": row.get("language", "")},
            )
            count += 1
            if self.max_samples and count >= self.max_samples:
                break

    def _first_human_turn(self, row) -> str:
        for turn in row.get("conversation") or []:
            role = (turn.get("role") or "").lower()
            content = (turn.get("content") or "").strip()
            if role in ("user", "human") and content:
                return content
        return ""
