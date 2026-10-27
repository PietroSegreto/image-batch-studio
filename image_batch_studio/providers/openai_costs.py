"""Image output estimates, excluding input tokens, taxes and additional attempts.

Formula and USD rates checked 2026-09-15 against the official calculator:
https://developers.openai.com/api/docs/guides/image-generation#cost-and-latency
The calculator groups Sunburst and Flare together; actual usage can differ.
"""
import math


OUTPUT_USD_PER_MILLION = 30
QUALITY_GRID = {
    "gpt-image-2": {"low": 16, "medium": 48, "high": 96},
    "gpt-image-2.5-sunburst": {"low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96},
    "gpt-image-2.5-flare": {"low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96},
}


def estimate_output_tokens(model_id: str, quality: str, width: int, height: int) -> int:
    """Mirror the official calculator, including ties-to-even grid rounding."""
    if (min(width, height) <= 0 or width % 16 or height % 16
            or max(width, height) > 3840
            or max(width, height) / min(width, height) > 3
            or not 655_360 <= width * height <= 8_294_400):
        raise ValueError("Dimensioni non valide per GPT Image.")
    grid = QUALITY_GRID[model_id][quality]
    short_grid = round(grid * min(width, height) / max(width, height))
    return math.ceil(grid * short_grid * (2_000_000 + width * height) / 4_000_000)


def estimate_output_cost(model_id: str, quality: str, width: int, height: int,
                         count: int = 1) -> tuple[float, float]:
    """Return USD low/high estimates; auto spans explicit quality settings.

    This interval is an indicative scenario, not a guaranteed billing bound.
    """
    if count < 0:
        raise ValueError("Il numero di immagini non può essere negativo.")
    qualities = QUALITY_GRID[model_id] if quality == "auto" else [quality]
    costs = [estimate_output_tokens(model_id, q, width, height)
             * OUTPUT_USD_PER_MILLION / 1_000_000 * count for q in qualities]
    return min(costs), max(costs)
