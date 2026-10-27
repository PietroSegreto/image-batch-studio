"""Factory dei provider: crea l'implementazione giusta per un ModelSpec."""
from image_batch_studio.config import BFL, GEMINI, OPENAI, ModelSpec
from image_batch_studio.providers.base import GeneratedImage, GenerationParams, ImageProvider
from image_batch_studio.providers.bfl_provider import BFLProvider
from image_batch_studio.providers.gemini_provider import GeminiProvider
from image_batch_studio.providers.openai_provider import OpenAIProvider

__all__ = [
    "GeneratedImage",
    "GenerationParams",
    "ImageProvider",
    "create_provider",
]


def create_provider(spec: ModelSpec, api_keys: dict[str, str | None]) -> ImageProvider:
    api_key = api_keys.get(spec.provider)
    if not api_key:
        raise ValueError(f"Chiave API mancante per il provider '{spec.provider}'.")
    if spec.provider == GEMINI:
        return GeminiProvider(api_key, spec)
    if spec.provider == BFL:
        return BFLProvider(api_key, spec)
    if spec.provider == OPENAI:
        return OpenAIProvider(api_key, spec)
    raise ValueError(f"Provider sconosciuto: {spec.provider}")
