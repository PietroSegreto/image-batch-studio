"""Provider per gpt-image-2 (OpenAI Image API).

Flusso (developers.openai.com/api/docs/guides/image-generation):
- senza immagini di riferimento: POST https://api.openai.com/v1/images/generations
  con body JSON;
- con immagini di riferimento: POST https://api.openai.com/v1/images/edits in
  multipart/form-data, passando i file nel campo ripetuto `image[]`.

In entrambi i casi la risposta è sincrona e contiene l'immagine base64 in
`data[0].b64_json` (gpt-image-2 non restituisce URL).

L'API non ha un parametro aspect_ratio: si passa `size` in pixel, con vincoli
stretti (lati multipli di 16, lato massimo 3840, rapporto lungo/corto <= 3:1,
pixel totali fra 655.360 e 8.294.400): dims_for_aspect li rispetta partendo da
aspect ratio + target in megapixel scelti nella UI.
"""
import base64
import math

import requests

from config import ModelSpec
from providers.base import GeneratedImage, GenerationParams, ImageProvider
from providers.retry import RejectedGenerationError, RetryableGenerationError

BASE_URL = "https://api.openai.com/v1"
TIMEOUT_S = 600
MAX_INPUT_IMAGES = 10

_MP_VALUES = {"1 MP": 1e6, "2 MP": 2e6, "4 MP": 4e6, "8 MP": 8e6}

MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_EDGE = 3840
STEP = 16

_MIME_BY_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"RIFF", "image/webp", "webp"),
)


def dims_for_aspect(aspect_ratio: str, resolution_label: str) -> tuple[int, int]:
    """Converte aspect ratio + target megapixel in (width, height) valida per l'API.

    Le dimensioni sono arrotondate al multiplo di 16 e poi corrette finché non
    rientrano nei limiti di lato massimo e di pixel totali.
    """
    target_px = _MP_VALUES.get(resolution_label, 1e6)
    w_ratio, h_ratio = (int(x) for x in aspect_ratio.split(":"))
    width = math.sqrt(target_px * w_ratio / h_ratio)
    height = width * h_ratio / w_ratio

    # Rientra nei limiti prima dello snap: lato massimo e pixel totali.
    scale = min(
        1.0,
        MAX_EDGE / max(width, height),
        math.sqrt(MAX_PIXELS / (width * height)),
    )
    if width * height * scale ** 2 < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (width * height))
    width, height = width * scale, height * scale

    def snap(value: float) -> int:
        return max(STEP * 4, min(MAX_EDGE, int(round(value / STEP)) * STEP))

    w, h = snap(width), snap(height)
    # Lo snap può far uscire di pochissimo dai limiti di pixel totali: correggi.
    while w * h < MIN_PIXELS and max(w, h) + STEP <= MAX_EDGE:
        w, h = w + STEP, h + STEP
    while w * h > MAX_PIXELS and min(w, h) > STEP * 4:
        w, h = w - STEP, h - STEP
    return w, h


def _image_part(index: int, raw: bytes) -> tuple[str, tuple[str, bytes, str]]:
    """Costruisce la tupla multipart per un'immagine di riferimento."""
    mime, ext = "image/png", "png"
    for magic, candidate_mime, candidate_ext in _MIME_BY_MAGIC:
        if raw.startswith(magic):
            mime, ext = candidate_mime, candidate_ext
            break
    return "image[]", (f"reference_{index + 1}.{ext}", raw, mime)


class OpenAIProvider(ImageProvider):
    def __init__(self, api_key: str, spec: ModelSpec):
        self._api_key = api_key
        self._spec = spec

    def generate(
        self,
        prompt: str,
        reference_images: list[bytes],
        params: GenerationParams,
    ) -> GeneratedImage:
        width, height = dims_for_aspect(params.aspect_ratio, params.resolution)
        refs = reference_images[:MAX_INPUT_IMAGES]

        fields: dict[str, str] = {
            "model": self._spec.model_id,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "quality": params.quality,
            "n": "1",
            "output_format": params.output_format,
            "background": params.background,
            "moderation": params.moderation,
        }
        if params.output_format in ("jpeg", "webp"):
            fields["output_compression"] = str(params.output_compression)

        headers = {"Authorization": f"Bearer {self._api_key}"}
        if refs:
            # Endpoint di editing: tutto in multipart, file nel campo image[].
            response = requests.post(
                f"{BASE_URL}/images/edits",
                headers=headers,
                data=fields,
                files=[_image_part(i, raw) for i, raw in enumerate(refs)],
                timeout=TIMEOUT_S,
            )
        else:
            body = dict(fields, n=1)
            if "output_compression" in fields:
                body["output_compression"] = params.output_compression
            response = requests.post(
                f"{BASE_URL}/images/generations",
                headers={**headers, "Content-Type": "application/json"},
                json=body,
                timeout=TIMEOUT_S,
            )

        if response.status_code != 200:
            error_type = RetryableGenerationError if response.status_code == 429 else (RejectedGenerationError if 400 <= response.status_code < 500 and response.status_code != 408 else RuntimeError)
            raise error_type(
                f"OpenAI {'edits' if refs else 'generations'} fallito "
                f"(HTTP {response.status_code}): {_error_detail(response)}"
            )

        payload = response.json()
        data = payload.get("data") or []
        b64 = data[0].get("b64_json") if data else None
        if not b64:
            raise RuntimeError(f"Risposta OpenAI senza immagine: {str(payload)[:300]}")

        meta = {
            "width": width,
            "height": height,
            "quality": params.quality,
            "endpoint": "edits" if refs else "generations",
        }
        if refs:
            meta["riferimenti"] = len(refs)
        usage = payload.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if usage.get(key) is not None:
                meta[key] = usage[key]

        return GeneratedImage(
            image_bytes=base64.b64decode(b64),
            extension="jpg" if params.output_format == "jpeg" else params.output_format,
            meta=meta,
        )


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("error", response.json()))[:300]
    except ValueError:
        return response.text[:300]
