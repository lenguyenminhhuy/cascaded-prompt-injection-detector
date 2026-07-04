"""AgentDojo -> tool_output positives (raw payloads).

AgentDojo's injection objectives live as ``GOAL = "..."`` / ``GOAL = f"..."``
assignments in each suite's ``injection_tasks.py``. We extract those goal
strings (the attacker's intent) and map each to a payload_family. Placeholder
tokens like ``{_RECIPIENT}`` are replaced with readable generics so the payload
is self-contained. SQL-domain tool wrapping happens in Chunk 3.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample, make_id
from src.data.taxonomy import agentdojo_family, difficulty_for

# GOAL = "..."  or  GOAL = f"""...""" (single- or triple-quoted, one logical line)
_GOAL_RE = re.compile(
    r"""GOAL\s*=\s*f?(['"]{1,3})(.*?)\1""",
    re.DOTALL,
)
_PLACEHOLDER = re.compile(r"\{_?([A-Za-z0-9_]+)\}")

_FILLERS = {
    "recipient": "attacker@example.com",
    "email": "attacker@example.com",
    "sender": "alerts@bank.example",
    "text": "the requested information",
    "subject": "Account update",
    "title": "Sync meeting",
    "participant": "attacker@example.com",
    "file_id": "13",
    "start_time": "2024-05-30 10:00",
    "amount": "100",
    "iban": "DE89370400440532013000",
}


def _fill(token: str) -> str:
    key = token.lower().strip("_")
    return _FILLERS.get(key, f"<{key}>")


class AgentDojoLoader(DatasetLoader):
    name = "agentdojo"

    def __init__(self, local: str = "data/raw/agentdojo", max_samples: int | None = None):
        self.root = Path(local)
        self.max_samples = max_samples

    def _goal_files(self) -> list[Path]:
        base = self.root / "src" / "agentdojo" / "default_suites"
        return sorted(base.rglob("injection_tasks.py")) if base.exists() else []

    def load(self) -> Iterator[Sample]:
        seen: set[str] = set()
        n = 0
        for path in self._goal_files():
            suite = path.parent.name
            src = path.read_text(encoding="utf-8")
            for _, raw in _GOAL_RE.findall(src):
                goal = _PLACEHOLDER.sub(lambda m: _fill(m.group(1)), raw).strip()
                goal = re.sub(r"\s+", " ", goal)
                # Skip f-string fragments that didn't resolve to a sentence.
                if len(goal) < 15 or "{" in goal or goal in seen:
                    continue
                seen.add(goal)
                family = agentdojo_family(goal)
                yield Sample(
                    id=make_id(self.name, family, n),
                    rendered_input=goal,
                    channel="tool_output",
                    label="injected",
                    payload_family=family,
                    domain="sql",
                    source=self.name,
                    difficulty=difficulty_for(goal),
                    structural_features=[],
                    notes=f"agentdojo:{suite}",
                ).validate()
                n += 1
                if self.max_samples and n >= self.max_samples:
                    return
