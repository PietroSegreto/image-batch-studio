"""Esecuzione batch dei job di generazione con thread pool e retry.

Riprende la logica di generator.py (concorrenza limitata, retry con backoff
esponenziale, salvataggio progressivo) ma in forma provider-agnostica, così
funziona sia per Gemini sia per BFL.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

from prompt_files import PromptJob, unique_filename
from providers.base import GeneratedImage, GenerationParams, ImageProvider
from providers.retry import retry_call


def _save_image(path: Path, data: bytes):
    # Creazione esclusiva: anche un altro processo non può sovrascrivere il file.
    with path.open("xb") as stream:
        try:
            stream.write(data)
            stream.flush()
        except OSError:
            stream.close()
            path.unlink(missing_ok=True)
            raise


@dataclass
class JobResult:
    job: PromptJob
    path: Path | None = None
    image: GeneratedImage | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.path is not None


def run_jobs(
    provider: ImageProvider,
    jobs: list[PromptJob],
    reference_images: list[bytes],
    params: GenerationParams,
    output_dir: Path,
    *,
    concurrency: int = 4,
    max_retries: int = 3,
    on_progress=None,
) -> list[JobResult]:
    """Esegue i job in parallelo salvando ogni immagine appena pronta.

    on_progress(done, total, result) è chiamato dal thread principale a ogni
    completamento (utile per aggiornare la UI Streamlit).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names = {p.stem.casefold() for p in output_dir.iterdir()}
    jobs = [replace(job, name=unique_filename(job.name, used_names)) for job in jobs]
    results: list[JobResult] = []

    def worker(job: PromptJob) -> JobResult:
        start = time.monotonic()
        image = None
        try:
            image = provider.generate_with_retries(
                job.prompt, reference_images, params, max_retries,
            )
            if image.extension not in {"png", "jpeg", "jpg", "webp"}:
                raise ValueError(f"Estensione immagine non valida: {image.extension}")
            path = output_dir / f"{job.name}.{image.extension}"
            retry_call(
                lambda: _save_image(path, image.image_bytes), max_retries,
                lambda exc: isinstance(exc, OSError) and not isinstance(exc, FileExistsError),
            )
            return JobResult(job=job, path=path, image=image,
                             elapsed_s=time.monotonic() - start, meta=image.meta)
        except Exception as exc:
            return JobResult(job=job, image=image, error=str(exc),
                             elapsed_s=time.monotonic() - start,
                             meta=image.meta if image else {})

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for done_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if on_progress is not None:
                on_progress(done_count, len(jobs), result)

    order = {id(job): i for i, job in enumerate(jobs)}
    results.sort(key=lambda r: order[id(r.job)])
    return results
