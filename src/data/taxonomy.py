"""payload_family mapping tables + difficulty heuristics (SCHEMA.md §2).

The external benchmarks describe attacks in their own vocabularies. These tables
project those vocabularies onto our channel-independent payload_family taxonomy
so a family seen in one channel can be recognized in another.
"""

from __future__ import annotations

import re

# --- BIPIA -------------------------------------------------------------------
# text_attack categories. Most are "do this other task instead" => override;
# a few solicit data / spread content => exfiltration-adjacent.
BIPIA_TEXT_FAMILY = {
    "Task Automation": "instruction_override",
    "Business Intelligence": "instruction_override",
    "Conversational Agent": "instruction_override",
    "Research Assistance": "instruction_override",
    "Sentiment Analysis": "instruction_override",
    "Substitution Ciphers": "instruction_override",
    "Base Encoding": "instruction_override",
    "Reverse Text": "instruction_override",
    "Emoji Substitution": "instruction_override",
    "Language Translation": "instruction_override",
    "Information Dissemination": "exfiltration",
    "Marketing & Advertising": "instruction_override",
    "Entertainment": "instruction_override",
    "Scams & Fraud": "exfiltration",
    "Misinformation & Propaganda": "instruction_override",
}
# code_attack categories are all "make the agent run harmful code/actions".
BIPIA_CODE_FAMILY_DEFAULT = "tool_misuse"
BIPIA_CODE_EXFIL = {"Data Eavesdropping", "Traffic Analysis", "Keylogging",
                    "Screen Scraping"}

# Categories that conceal intent behind output transforms read as harder.
BIPIA_HARD_TEXT = {"Substitution Ciphers", "Base Encoding", "Reverse Text",
                   "Emoji Substitution", "Language Translation"}

# --- InjecAgent --------------------------------------------------------------
# dh (direct harm) test cases -> unintended tool action; ds (data stealing) -> exfil.
INJECAGENT_SPLIT_FAMILY = {"dh": "tool_misuse", "ds": "exfiltration"}

# --- AgentDojo ---------------------------------------------------------------
# Heuristic over the injection GOAL text.
def agentdojo_family(goal: str) -> str:
    g = goal.lower()
    if any(w in g for w in ("send", "forward", "email", "leak", "post the", "share")):
        # sending data outward = exfiltration; sending a *new* action = tool_misuse.
        if any(w in g for w in ("security code", "body of the email", "contents", "balance", "iban")):
            return "exfiltration"
        return "exfiltration" if "to '" in goal or "to the" in g else "tool_misuse"
    if any(w in g for w in ("delete", "create", "transfer", "pay", "grant", "add", "change", "update", "schedule")):
        return "tool_misuse"
    if any(w in g for w in ("ignore", "stop what", "instead", "do not")):
        return "instruction_override"
    return "tool_misuse"


# --- HackAPrompt / TensorTrust (direct) --------------------------------------
# Must reference *revealing/repeating* the prompt/instructions, not merely
# contain the words "you are" (which leaks from benign app framing).
_SPX_PAT = re.compile(
    r"(reveal|repeat|print|show|output|tell me|what (is|are|were)|give me|"
    r"say).{0,40}"
    r"(system prompt|the prompt|your (instructions|prompt|rules|guidelines)|"
    r"initial prompt|developer message|everything above|the (text|words) above|"
    r"the password|access code|secret)",
    re.I,
)
_SPX_SIMPLE = re.compile(
    r"(repeat everything above|what is your system prompt|"
    r"ignore.{0,30}(repeat|print).{0,30}above)",
    re.I,
)


def direct_family(text: str, intention: str | None = None) -> str:
    """Distinguish system-prompt extraction from generic instruction override."""
    blob = f"{text} {intention or ''}"
    if _SPX_PAT.search(blob) or _SPX_SIMPLE.search(blob):
        return "system_prompt_extraction"
    return "instruction_override"


# --- difficulty heuristics ---------------------------------------------------
_OBFUSCATION = re.compile(r"(base64|rot13|hex|unicode|zero.?width|reverse|cipher|"
                          r"emoji|leet|\\u00|%[0-9a-f]{2})", re.I)


def difficulty_for(text: str, *, enhanced: bool = False, level: int | None = None) -> str:
    """easy | hard (ambiguous is reserved for hand-tagged near-boundary cases)."""
    if enhanced or _OBFUSCATION.search(text or ""):
        return "hard"
    if level is not None and level >= 4:
        return "hard"
    if len(text or "") > 400:
        return "hard"
    return "easy"
