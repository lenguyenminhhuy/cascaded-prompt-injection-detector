from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.data.schema import Sample


class DatasetLoader(ABC):
    """A source loader yields validated positive Samples in the v1 contract.

    Loaders emit *un-rendered* payloads: ``rendered_input`` holds the raw attack
    text and structural_features is empty. Channel-faithful rendering happens in
    Chunk 3 (src/data/rendering). Loaders set channel, label, payload_family,
    domain, source, difficulty.
    """

    name: str

    @abstractmethod
    def load(self) -> Iterator[Sample]:
        """Yield Samples in the unified v1 schema (validated by the caller)."""
