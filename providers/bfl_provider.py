"""Provider per la famiglia FLUX.2 (Black Forest Labs REST API).

Flusso (docs.bfl.ai): POST https://api.bfl.ai/v1/<endpoint> con header `x-key`
-> risposta con `polling_url` -> GET periodico finché status == "Ready"
-> download dell'immagine dall'URL firmato `result.sample` (valido 10 minuti).

L'API non ha un parametro aspect_ratio: si passano width/height in pixel,
calcolati da aspect ratio + target megapixel (dims_for_aspect). Le immagini di
riferimento si passano base64 nei campi input_image .. input_image_8.
"""
import base64
import math
import time

import requests

from config import ModelSpec
from providers.base import GeneratedImage, GenerationParams, ImageProvider
from providers.retry import RejectedGenerationError, RetryableGenerationError, retry_call, retryable_read

BASE_URL = "https://api.bfl.ai/v1"
POLL_INTERVAL_S = 1.5
POLL_TIMEOUT_S = 360
MAX_INPUT_IMAGES = 8

_MP_VALUES = {"0.5 MP": 0.5e6, "1 MP": 1e6, "2 MP": 2e6, "4 MP": 4e6}


def dims_for_aspect(aspect_ratio: str, resolution_label: str) -> tuple[int, int]:
    """Converte aspect ratio + target megapixel in (width, height).

    Le dimensioni sono arrotondate al multiplo di 32 (requisito tipico dei
    modelli FLUX) con minimo 64 px per lato.
    """
    target_px = _MP_VALUES.get(resolution_label, 1e6)
    w_ratio, h_ratio = (int(x) for x in aspect_ratio.split(":"))
    width = math.sqrt(target_px * w_ratio / h_ratio)
    height = width * h_ratio / w_ratio

    def snap(value: float) -> int:
        return max(64, int(round(value / 32)) * 32)

    return snap(width), snap(height)


class BFLProvider(ImageProvider):
    def __init__(self, api_key: str, spec: ModelSpec):
        self._api_key = api_key
        self._spec = spec

    def generate(
        self,
        prompt: str,
        reference_images: list[bytes],
        params: GenerationParams,
    ) -> GeneratedImage:
        return self.generate_with_retries(prompt, reference_images, params, 3)

    def generate_with_retries(self, prompt, reference_images, params, attempts, *, checkpoint=None, resume=None, on_attempt=None):
        width, height = dims_for_aspect(params.aspect_ratio, params.resolution)

        body: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "output_format": params.output_format,
            "safety_tolerance": params.safety_tolerance,
            "prompt_upsampling": params.prompt_upsampling,
        }
        if params.seed is not None:
            body["seed"] = params.seed
        if self._spec.supports_guidance_steps:
            if params.guidance is not None:
                body["guidance"] = params.guidance
            if params.steps is not None:
                body["steps"] = params.steps

        for i, raw in enumerate(reference_images[:MAX_INPUT_IMAGES]):
            key = "input_image" if i == 0 else f"input_image_{i + 1}"
            body[key] = base64.b64encode(raw).decode("ascii")

        headers = {
            "accept": "application/json",
            "x-key": self._api_key,
            "Content-Type": "application/json",
        }
        submission = resume or retry_call(
            lambda: self._submit(body, headers), attempts,
            lambda exc: isinstance(exc, RetryableGenerationError),
            on_attempt=lambda n: on_attempt("generation", n) if on_attempt else None,
        )
        if checkpoint:
            checkpoint(submission)
        polling_url = submission.get("polling_url")
        if not polling_url:
            raise RuntimeError(f"Risposta BFL senza polling_url: {submission}")

        result = self._poll(polling_url, attempts, on_attempt)
        sample_url = result.get("result", {}).get("sample")
        if not sample_url:
            raise RuntimeError(f"Risultato BFL senza URL immagine: {result}")

        image = retry_call(lambda: self._get(sample_url, timeout=120), attempts, retryable_read,
                           on_attempt=lambda n: on_attempt("download", n) if on_attempt else None)

        meta = {"width": width, "height": height}
        for key in ("cost", "seed"):
            value = submission.get(key) or result.get("result", {}).get(key)
            if value is not None:
                meta[key] = value
        return GeneratedImage(
            image_bytes=image.content,
            extension=params.output_format,
            meta=meta,
        )

    def _submit(self, body, headers):
        submit = requests.post(f"{BASE_URL}/{self._spec.model_id}",
                               headers=headers, json=body, timeout=60)
        if submit.status_code != 200:
            error_type = RetryableGenerationError if submit.status_code == 429 else (RejectedGenerationError if 400 <= submit.status_code < 500 and submit.status_code != 408 else RuntimeError)
            raise error_type(f"BFL submit fallito (HTTP {submit.status_code}): {submit.text[:300]}")
        return submit.json()

    @staticmethod
    def _get(url, **kwargs):
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        return response

    def _poll(self, polling_url: str, attempts: int = 3, on_attempt=None) -> dict:
        headers = {"accept": "application/json", "x-key": self._api_key}
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_S)
            response = retry_call(
                lambda: self._get(polling_url, headers=headers, timeout=30),
                attempts, retryable_read,
                on_attempt=lambda n: on_attempt("polling", n) if on_attempt else None,
            )
            payload = response.json()
            status = payload.get("status")
            if status == "Ready":
                return payload
            if status in ("Error", "Failed", "Content Moderated", "Request Moderated"):
                raise RejectedGenerationError(f"Generazione BFL fallita ({status}): {payload}")
        raise TimeoutError(f"Timeout dopo {POLL_TIMEOUT_S}s in attesa del risultato BFL.")
