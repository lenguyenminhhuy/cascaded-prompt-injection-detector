import random

from src.data.schema import Sample


def balanced_sample(
    samples: list[Sample],
    ratio: float = 0.5,
    seed: int = 42,
) -> list[Sample]:
    """Downsample the majority class so injection_count / total == ratio.

    Only the majority class is trimmed; the minority class is never upsampled.
    Returns a shuffled list.

    Args:
        samples: Input samples containing both benign and injection labels.
        ratio: Target fraction of injection samples in the output (0 < ratio < 1).
        seed: Random seed for reproducibility.

    Returns:
        Balanced, shuffled list of samples.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    rng = random.Random(seed)

    injections = [s for s in samples if s.label == "injection"]
    benigns = [s for s in samples if s.label == "benign"]

    n_inj = len(injections)
    n_ben = len(benigns)

    if n_inj == 0 or n_ben == 0:
        result = list(samples)
        rng.shuffle(result)
        return result

    # injection / total = ratio  →  total = injection / ratio
    # benign / total = 1 - ratio  →  total = benign / (1 - ratio)
    # We only trim; pick the scenario that requires fewer total samples.
    target_total_from_inj = int(n_inj / ratio)
    target_total_from_ben = int(n_ben / (1 - ratio))

    if target_total_from_inj <= target_total_from_ben:
        # Injections are the minority or exactly balanced; trim benigns.
        target_ben = int(n_inj * (1 - ratio) / ratio)
        selected_ben = rng.sample(benigns, min(target_ben, n_ben))
        selected_inj = injections
    else:
        # Benigns are the minority; trim injections.
        target_inj = int(n_ben * ratio / (1 - ratio))
        selected_inj = rng.sample(injections, min(target_inj, n_inj))
        selected_ben = benigns

    result = selected_inj + selected_ben
    rng.shuffle(result)
    return result
