from dataclasses import dataclass, field, asdict
from typing import Literal, Any

Label = Literal["benign", "injection"]
Channel = Literal["document_embedded", "tool_output", "direct", "app_structured", "conversational"]


@dataclass
class Sample:
    input: str
    label: Label
    channel: Channel
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
