"""Per-loader validation: each loader must produce valid Sample objects."""

import pytest

from src.data.schema import Sample

# Fast cap: only pull 2 samples per loader in tests
_MAX = 2


def _assert_sample(s: Sample, expected_label: str, expected_channel: str) -> None:
    assert isinstance(s, Sample), f"expected Sample, got {type(s)}"
    assert s.input and s.input.strip(), "input must be non-empty"
    assert s.label == expected_label, f"label: expected {expected_label!r}, got {s.label!r}"
    assert s.channel == expected_channel, f"channel: expected {expected_channel!r}, got {s.channel!r}"
    assert s.source, "source must be non-empty"


def _load_and_check(loader, expected_label: str, expected_channel: str, min_count: int = 1) -> list[Sample]:
    samples = list(loader.load())
    assert len(samples) >= min_count, (
        f"{loader.name}: expected ≥{min_count} samples, got {len(samples)}"
    )
    for s in samples:
        _assert_sample(s, expected_label, expected_channel)
    return samples


# ── Local loaders (no network required) ────────────────────────────────────────


class TestInjecAgent:
    def test_train_split(self):
        from src.data.loaders.injecagent import InjecAgentLoader
        loader = InjecAgentLoader(split="train", max_samples=_MAX)
        samples = _load_and_check(loader, "injection", "tool_output")
        assert len(samples) == _MAX

    def test_eval_split(self):
        from src.data.loaders.injecagent import InjecAgentLoader
        loader = InjecAgentLoader(split="eval", max_samples=_MAX)
        samples = _load_and_check(loader, "injection", "tool_output")
        assert len(samples) == _MAX

    def test_splits_are_disjoint(self):
        from src.data.loaders.injecagent import InjecAgentLoader
        train = {s.input for s in InjecAgentLoader(split="train").load()}
        eval_ = {s.input for s in InjecAgentLoader(split="eval").load()}
        assert train.isdisjoint(eval_), "train and eval InjecAgent splits must be disjoint"

    def test_reproducible(self):
        from src.data.loaders.injecagent import InjecAgentLoader
        a = [s.input for s in InjecAgentLoader(split="train", max_samples=5).load()]
        b = [s.input for s in InjecAgentLoader(split="train", max_samples=5).load()]
        assert a == b, "InjecAgentLoader must be deterministic"


# ── HuggingFace loaders (network) ──────────────────────────────────────────────


@pytest.mark.network
class TestAlpaca:
    def test_schema(self):
        from src.data.loaders.alpaca import AlpacaLoader
        _load_and_check(AlpacaLoader(max_samples=_MAX), "benign", "app_structured")


@pytest.mark.network
class TestIFEval:
    def test_schema(self):
        from src.data.loaders.ifeval import IFEvalLoader
        _load_and_check(IFEvalLoader(max_samples=_MAX), "benign", "conversational")


@pytest.mark.network
class TestNotInject:
    def test_schema(self):
        from src.data.loaders.notinject import NotInjectLoader
        _load_and_check(NotInjectLoader(max_samples=_MAX), "benign", "direct")


@pytest.mark.network
class TestDolly:
    def test_schema(self):
        from src.data.loaders.dolly import DollyLoader
        _load_and_check(DollyLoader(max_samples=_MAX), "benign", "app_structured")


@pytest.mark.network
class TestNaturalInstructions:
    def test_schema(self):
        from src.data.loaders.natural_instructions import NaturalInstructionsLoader
        _load_and_check(NaturalInstructionsLoader(max_samples=_MAX), "benign", "app_structured")


@pytest.mark.network
class TestSpp:
    def test_schema(self):
        from src.data.loaders.spp import SppLoader
        samples = list(SppLoader(max_samples=_MAX).load())
        # SPP may have 0 samples if HF dataset unavailable; just validate schema if present
        for s in samples:
            _assert_sample(s, "benign", "app_structured")


@pytest.mark.network
class TestHackaprompt:
    def test_schema(self):
        from src.data.loaders.hackaprompt import HackapromptLoader
        _load_and_check(HackapromptLoader(max_samples=_MAX), "injection", "direct")


@pytest.mark.network
class TestWildguard:
    def test_schema(self):
        from src.data.loaders.wildguard import WildguardLoader
        _load_and_check(WildguardLoader(max_samples=_MAX), "injection", "direct")


@pytest.mark.network
class TestUltrachat:
    def test_schema(self):
        from src.data.loaders.ultrachat import UltrachatLoader
        _load_and_check(UltrachatLoader(max_samples=_MAX), "benign", "conversational")


@pytest.mark.network
class TestLmsysChat:
    def test_schema(self):
        from src.data.loaders.lmsys_chat import LmsysChatLoader
        _load_and_check(LmsysChatLoader(max_samples=_MAX), "benign", "conversational")


# ── GitHub loaders (require cloned repos or network) ──────────────────────────


@pytest.mark.network
class TestBipia:
    def test_schema(self):
        from src.data.loaders.bipia import BipiaLoader
        _load_and_check(BipiaLoader(max_samples=_MAX), "injection", "document_embedded")


@pytest.mark.network
class TestOpenPromptInjection:
    def test_schema(self):
        from src.data.loaders.open_prompt_injection import OpenPromptInjectionLoader
        _load_and_check(
            OpenPromptInjectionLoader(max_samples=_MAX), "injection", "document_embedded"
        )


@pytest.mark.network
class TestStruq:
    def test_schema(self):
        from src.data.loaders.struq import StruQLoader
        _load_and_check(StruQLoader(max_samples=_MAX), "injection", "app_structured")
