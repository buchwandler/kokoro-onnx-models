"""Small model-specific frontend normalization helpers for local gates."""


def thorsten_cleanup(phonemes: str) -> str:
    """Apply Thorsten's training-time short-ü normalization."""
    return phonemes.replace("ʏ", "y")
