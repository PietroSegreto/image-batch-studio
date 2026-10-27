"""Retry espliciti: le richieste di generazione possono avere esito ambiguo."""
import time

import requests


class RejectedGenerationError(RuntimeError):
    """Il servizio ha rifiutato la richiesta o dichiarato il job fallito."""


class RetryableGenerationError(RejectedGenerationError):
    """Richiesta rifiutata temporaneamente prima della generazione (HTTP 429)."""


def retry_call(operation, attempts: int, retryable, on_attempt=None):
    for attempt in range(max(1, attempts)):
        if on_attempt:
            on_attempt(attempt + 1)
        try:
            return operation()
        except Exception as exc:
            if attempt + 1 >= max(1, attempts) or not retryable(exc):
                raise
            time.sleep(2.0 * 2 ** attempt)


def retryable_read(exc: Exception) -> bool:
    """GET: ripetibili anche dopo timeout o errori temporanei del server."""
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    return (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and (exc.response.status_code == 429 or 500 <= exc.response.status_code < 600)
    )
