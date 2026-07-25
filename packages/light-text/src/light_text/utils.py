"""Shared character-level utilities for CJK text detection."""


def is_cjk(ch: str) -> bool:
    """Check whether a character is a CJK (Chinese/Japanese/Korean) character.

    Covers:
        - CJK Unified Ideographs (4E00–9FFF)
        - CJK Extension A (3400–4DBF)
        - Kangxi Radicals / CJK Radicals Supplement (2E80–2FDF)
        - CJK Compatibility Ideographs (F900–FAFF)
    """
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF  # Extension A
        or 0x4E00 <= cp <= 0x9FFF  # Unified Ideographs
        or 0x2E80 <= cp <= 0x2FDF  # Kangxi / CJK Radicals
        or 0xF900 <= cp <= 0xFAFF
    )  # Compatibility Ideographs
