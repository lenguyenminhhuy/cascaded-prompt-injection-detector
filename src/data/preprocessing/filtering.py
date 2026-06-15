from src.data.schema import Sample


def filter_success(samples: list[Sample], field: str = "attack_success") -> list[Sample]:
    """Keep samples where metadata[field] is truthy. If field is missing, keep the sample (assume success)."""
    return [s for s in samples if s.metadata.get(field, True)]


def filter_benign_only(samples: list[Sample]) -> list[Sample]:
    """Keep only samples with label == 'benign'."""
    return [s for s in samples if s.label == "benign"]


def filter_injection_only(samples: list[Sample]) -> list[Sample]:
    """Keep only samples with label == 'injection'."""
    return [s for s in samples if s.label == "injection"]
