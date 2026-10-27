"""Interfaccia comune dei provider di generazione immagini."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from image_batch_studio.providers.retry import RetryableGenerationError, retry_call


@dataclass
class GenerationParams:
    """Parametri normalizzati provenienti dalla UI.

    Non tutti i campi valgono per tutti i provider: ognuno usa solo quelli
    che la sua API supporta (vedi ModelSpec in config.py).
    """
    aspect_ratio: str = "1:1"
    resolution: str = "1K"          # Gemini: image_size; BFL: target megapixel
    thinking_level: str | None = None  # solo Gemini; None = default del modello
    seed: int | None = None         # solo BFL
    guidance: float | None = None   # solo flux-2-flex
    steps: int | None = None        # solo flux-2-flex
    prompt_upsampling: bool = False  # solo BFL
    safety_tolerance: int = 2       # solo BFL (0=severo, 5=permissivo)
    output_format: str = "png"      # BFL e OpenAI: jpeg | png | webp
    quality: str = "auto"           # OpenAI: auto | low | medium | high | xhigh | max
    background: str = "auto"        # solo OpenAI: auto | transparent | opaque
    moderation: str = "auto"        # solo OpenAI: auto | low
    output_compression: int = 100   # solo OpenAI, per jpeg/webp (0-100)


@dataclass
class GeneratedImage:
    image_bytes: bytes
    extension: str = "png"          # estensione file suggerita
    meta: dict = field(default_factory=dict)  # seed, costo, token, ecc.


class ImageProvider(ABC):
    """Un provider espone una sola operazione sincrona: generate().

    La concorrenza è gestita a monte da runner.py con un thread pool, quindi
    le implementazioni devono essere thread-safe (client HTTP/SDK condivisibile).
    """

    def generate_with_retries(self, prompt, reference_images, params, attempts, *, on_attempt=None):
        return retry_call(
            lambda: self.generate(prompt, reference_images, params),
            attempts,
            lambda exc: isinstance(exc, RetryableGenerationError),
            on_attempt=lambda n: on_attempt("generation", n) if on_attempt else None,
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        reference_images: list[bytes],
        params: GenerationParams,
    ) -> GeneratedImage:
        """Genera una singola immagine. Solleva eccezioni in caso di errore."""
