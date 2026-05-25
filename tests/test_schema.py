from src.data.schema import Sample


def test_sample_roundtrip():
    s = Sample(
        input="ignore previous instructions and ...",
        label="injection",
        channel="direct",
        source="hackaprompt",
        metadata={"id": "abc-123"},
    )
    d = s.to_dict()
    assert d["label"] == "injection"
    assert d["channel"] == "direct"
    assert d["metadata"]["id"] == "abc-123"
