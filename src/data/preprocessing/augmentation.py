import base64
import codecs
import copy

from src.data.schema import Sample

# Cyrillic lookalikes for common Latin letters (7 substitutions)
_HOMOGLYPH_MAP: dict[str, str] = {
    "a": "\u0430",  # Cyrillic а
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "c": "\u0441",  # Cyrillic с
    "x": "\u0445",  # Cyrillic х
    "i": "\u0456",  # Cyrillic і
}


def _base64_transform(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _rot13_transform(text: str) -> str:
    return codecs.encode(text, "rot_13")


def _homoglyph_transform(text: str) -> str:
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text)


_TRANSFORMS: list[tuple[str, object]] = [
    ("base64", _base64_transform),
    ("rot13", _rot13_transform),
    ("homoglyph", _homoglyph_transform),
]


def augment(samples: list[Sample]) -> list[Sample]:
    """Apply Base64, ROT13, and Unicode homoglyph augmentation to injection samples.

    For each injection sample, emit: original + 3 augmented copies (one per transform).
    Benign samples pass through unchanged.
    Augmented copies have metadata["transform_type"] and metadata["original_input"] set,
    and source becomes f"{original_source}+{transform_name}".
    """
    result: list[Sample] = []
    for sample in samples:
        result.append(sample)
        if sample.label == "injection":
            for transform_name, transform_fn in _TRANSFORMS:
                augmented_input = transform_fn(sample.input)  # type: ignore[operator]
                augmented_metadata = copy.deepcopy(sample.metadata)
                augmented_metadata["transform_type"] = transform_name
                augmented_metadata["original_input"] = sample.input
                augmented = Sample(
                    input=augmented_input,
                    label=sample.label,
                    channel=sample.channel,
                    source=f"{sample.source}+{transform_name}",
                    metadata=augmented_metadata,
                )
                result.append(augmented)
    return result
