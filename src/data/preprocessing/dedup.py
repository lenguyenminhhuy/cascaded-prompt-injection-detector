from src.data.schema import Sample


def dedup(samples: list[Sample]) -> list[Sample]:
    """Exact-match dedup on the input field. First occurrence wins, stable order."""
    seen: set[str] = set()
    result: list[Sample] = []
    for sample in samples:
        if sample.input not in seen:
            seen.add(sample.input)
            result.append(sample)
    return result
