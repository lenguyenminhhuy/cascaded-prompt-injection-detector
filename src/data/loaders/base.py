from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.data.schema import Sample


class DatasetLoader(ABC):
    name: str

    @abstractmethod
    def load(self) -> Iterator[Sample]:
        """Yield samples in the unified schema."""
