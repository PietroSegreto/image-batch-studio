"""Provider per i modelli Nano Banana (Gemini API, SDK google-genai)."""
import io

import PIL.Image
from google import genai
from google.genai import types

from config import ModelSpec
from providers.base import GeneratedImage, GenerationParams, ImageProvider
from providers.retry import RejectedGenerationError, RetryableGenerationError


class GeminiProvider(ImageProvider):
    def __init__(self, api_key: str, spec: ModelSpec):
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )
        self._spec = spec

    def generate(
        self,
        prompt: str,
        reference_images: list[bytes],
        params: GenerationParams,
    ) -> GeneratedImage:
        contents: list = [prompt]
        for raw in reference_images:
            img = PIL.Image.open(io.BytesIO(raw))
            img.load()
            contents.append(img)

        config = types.GenerateContentConfig(
            response_modalities=list(self._spec.response_modalities or ("Image",)),
            image_config=types.ImageConfig(
                aspect_ratio=params.aspect_ratio,
                image_size=params.resolution,
            ),
        )

        try:
            response = self._client.models.generate_content(
                model=self._spec.model_id, contents=contents, config=config,
            )
        except genai.errors.APIError as exc:
            if exc.code == 429:
                raise RetryableGenerationError(str(exc)) from exc
            if exc.code and 400 <= exc.code < 500 and exc.code != 408:
                raise RejectedGenerationError(str(exc)) from exc
            raise

        image_bytes = _extract_image_bytes(response)
        if image_bytes is None:
            detail = _safe_text(response) or "nessuna immagine nella risposta"
            raise RuntimeError(f"Gemini non ha restituito immagini: {detail}")

        meta = {}
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_token_count", None)
            meta["output_tokens"] = getattr(usage, "candidates_token_count", None)
        return GeneratedImage(image_bytes=image_bytes, extension="png", meta=meta)


def _extract_image_bytes(response) -> bytes | None:
    """Estrae i byte della prima immagine della risposta (inline_data)."""
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                return inline.data
    return None


def _safe_text(response) -> str:
    try:
        return (response.text or "").strip()
    except Exception:
        return ""
