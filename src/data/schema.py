"""cascade-pid v1 data contract.

This module is the single source of truth for the per-example schema defined in
SCHEMA.md. Every example in every split file conforms to ``Sample``.

Core domain is Text-to-SQL / data-assistant (``domain == "sql"``). Non-``sql``
domains only appear in the held-out cross-domain test slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, get_args

# --------------------------------------------------------------------------- #
# Controlled vocabularies (keep in lock-step with SCHEMA.md)
# --------------------------------------------------------------------------- #

Label = Literal["injected", "benign"]
Channel = Literal["direct", "document_embedded", "tool_output"]
Domain = Literal["sql", "browsing", "calendar", "email"]
Difficulty = Literal["easy", "hard", "ambiguous"]
Split = Literal[
    "train",
    "cal",
    "test_in_dist",
    "test_cross_channel",
    "test_cross_domain",
    # "unassigned" is a working state before Chunk 6 stamps the real split.
    "unassigned",
]

# Attack intent, independent of channel. ``None`` only when label == "benign".
PayloadFamily = Literal[
    "instruction_override",
    "exfiltration",
    "tool_misuse",
    "system_prompt_extraction",
    "rbac_bypass",
]

Source = Literal[
    "agentdojo",
    "injecagent",
    "bipia",
    "hackaprompt",
    "tensortrust",
    "synthetic",
    "authored",
]

# Structural concealment / framing tricks for document/tool channels. Audit aid
# only; never a model feature. We enumerate the known vocabulary so coverage can
# be measured per SCHEMA.md §3.
STRUCTURAL_FEATURES = frozenset(
    {
        "html_comment",
        "zero_width",
        "white_on_white",
        "tiny_font",
        "chunk_position",
        "json_field",
        "db_row",
        "metadata_description",
        "code_block",
        "quoted_text",
        "ddl_comment",
        "csv_cell",
    }
)

LABELS = frozenset(get_args(Label))
CHANNELS = frozenset(get_args(Channel))
DOMAINS = frozenset(get_args(Domain))
DIFFICULTIES = frozenset(get_args(Difficulty))
SPLITS = frozenset(get_args(Split))
PAYLOAD_FAMILIES = frozenset(get_args(PayloadFamily))
SOURCES = frozenset(get_args(Source))


class SchemaError(ValueError):
    """Raised when a Sample violates the v1 data contract."""


@dataclass
class Sample:
    """One example in the cascade-pid dataset.

    ``rendered_input`` is the ONLY field the model sees. All other fields are
    stratification / provenance / audit metadata and must never leak into it.
    """

    id: str
    rendered_input: str
    channel: str
    label: str
    payload_family: str | None
    domain: str
    source: str
    difficulty: str
    structural_features: list[str] = field(default_factory=list)
    split: str = "unassigned"
    notes: str = ""

    # ------------------------------------------------------------------ #

    def validate(self) -> "Sample":
        """Enforce the v1 contract. Returns self so calls can be chained."""
        if not self.id:
            raise SchemaError("id is required and must be non-empty")
        if not self.rendered_input or not self.rendered_input.strip():
            raise SchemaError(f"{self.id}: rendered_input must be non-empty")
        if self.channel not in CHANNELS:
            raise SchemaError(f"{self.id}: bad channel {self.channel!r}")
        if self.label not in LABELS:
            raise SchemaError(f"{self.id}: bad label {self.label!r}")
        if self.domain not in DOMAINS:
            raise SchemaError(f"{self.id}: bad domain {self.domain!r}")
        if self.difficulty not in DIFFICULTIES:
            raise SchemaError(f"{self.id}: bad difficulty {self.difficulty!r}")
        if self.source not in SOURCES:
            raise SchemaError(f"{self.id}: bad source {self.source!r}")
        if self.split not in SPLITS:
            raise SchemaError(f"{self.id}: bad split {self.split!r}")

        # Label <-> payload_family coupling (SCHEMA.md §4 rule 6).
        if self.label == "benign":
            if self.payload_family is not None:
                raise SchemaError(
                    f"{self.id}: benign examples must have payload_family=None, "
                    f"got {self.payload_family!r}"
                )
        else:  # injected
            if self.payload_family not in PAYLOAD_FAMILIES:
                raise SchemaError(
                    f"{self.id}: injected examples need a valid payload_family, "
                    f"got {self.payload_family!r}"
                )

        # direct channel carries no structural concealment (SCHEMA.md §3).
        if self.channel == "direct" and self.structural_features:
            raise SchemaError(
                f"{self.id}: direct channel must not carry structural_features"
            )
        bad = set(self.structural_features) - STRUCTURAL_FEATURES
        if bad:
            raise SchemaError(f"{self.id}: unknown structural_features {sorted(bad)}")

        # No metadata leak into the rendered text.
        lowered = self.rendered_input.lower()
        for leak in ("channel:", "payload_family:", "difficulty:", "label:"):
            if leak in lowered:
                raise SchemaError(f"{self.id}: possible metadata leak ({leak!r})")

        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        return cls(
            id=d["id"],
            rendered_input=d["rendered_input"],
            channel=d["channel"],
            label=d["label"],
            payload_family=d.get("payload_family"),
            domain=d["domain"],
            source=d["source"],
            difficulty=d["difficulty"],
            structural_features=list(d.get("structural_features") or []),
            split=d.get("split", "unassigned"),
            notes=d.get("notes", ""),
        )

    @property
    def cell(self) -> tuple[str, str]:
        """The (payload_family, channel) cell used for cross-channel holdout."""
        return (self.payload_family or "benign", self.channel)


def make_id(source: str, payload_family: str | None, n: int) -> str:
    """Stable id, e.g. ``bipia-exfil-000123``. Abbreviates families for brevity."""
    fam = {
        "instruction_override": "ovr",
        "exfiltration": "exfil",
        "tool_misuse": "tool",
        "system_prompt_extraction": "spx",
        "rbac_bypass": "rbac",
        None: "benign",
    }.get(payload_family, "x")
    return f"{source}-{fam}-{n:06d}"
