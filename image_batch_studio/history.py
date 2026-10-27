"""Storico su disco e ripresa conservativa delle generazioni."""
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from image_batch_studio.config import ModelSpec
from image_batch_studio.prompt_files import PromptJob, unique_filename
from image_batch_studio.providers import GenerationParams, create_provider
from image_batch_studio.providers.bfl_provider import BFLProvider
from image_batch_studio.providers.retry import RejectedGenerationError, retry_call
from image_batch_studio.runner import JobResult, _save_image


STATES = {'pending', 'running', 'recoverable', 'saving', 'completed', 'failed', 'uncertain'}


def now():
    return datetime.now(timezone.utc).isoformat()


def local_path(folder, relative):
    path = (folder / relative).resolve()
    if path == folder.resolve() or not path.is_relative_to(folder.resolve()):
        raise ValueError('Percorso fuori dalla cartella della run.')
    return path


def atomic_json(path, data):
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(folder):
    """Lock OS rilasciato anche se il processo termina improvvisamente."""
    with (folder / '.run.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Questa run è già in esecuzione in un’altra sessione.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def create_run(folder, spec, params, jobs, references, concurrency=4, attempts=3):
    folder.mkdir(parents=True, exist_ok=False)
    names = set()
    refs = []
    if references:
        (folder / 'references').mkdir()
    for i, raw in enumerate(references):
        relative = f'references/{i + 1}.bin'
        local_path(folder, relative).write_bytes(raw)
        refs.append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest()})
    data = {
        'version': 1, 'label': folder.name, 'created_at': now(), 'updated_at': now(),
        'model': asdict(spec), 'params': asdict(params), 'references': refs,
        'concurrency': concurrency, 'attempts_per_phase': attempts,
        'jobs': [dict(asdict(job), id=str(i), name=unique_filename(job.name, names),
                      status='pending', executions=0, phase_attempts={}, events=[], error=None, meta={})
                 for i, job in enumerate(jobs)],
    }
    atomic_json(folder / 'run.json', data)
    return data


def load_run(folder):
    try:
        data = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        if data['version'] != 1 or not isinstance(data['jobs'], list):
            raise ValueError('Versione o struttura non supportata.')
        if not isinstance(data['label'], str) or not isinstance(data['created_at'], str):
            raise ValueError('Metadati run non validi.')
        if not isinstance(data['concurrency'], int) or not 1 <= data['concurrency'] <= 16:
            raise ValueError('Parallelismo non valido.')
        if not isinstance(data['attempts_per_phase'], int) or not 1 <= data['attempts_per_phase'] <= 5:
            raise ValueError('Numero di tentativi non valido.')
        ModelSpec(**data['model'])
        GenerationParams(**data['params'])
        ids = set()
        for job in data['jobs']:
            if job['status'] not in STATES or job['id'] in ids:
                raise ValueError('Stato o identificativo job non valido.')
            if (not isinstance(job['id'], str) or not isinstance(job['name'], str)
                    or not isinstance(job['prompt'], str) or not isinstance(job['events'], list)
                    or not isinstance(job['executions'], int)):
                raise ValueError('Struttura job non valida.')
            if job['status'] in {'completed', 'saving'} and not (job.get('path') and job.get('sha256')):
                raise ValueError('Job salvato senza percorso o checksum.')
            ids.add(job['id'])
            PromptJob(job['name'], job['prompt'], job['source_row'])
            if job.get('path'):
                local_path(folder, job['path'])
        for ref in data['references']:
            if not isinstance(ref['sha256'], str):
                raise ValueError('Checksum riferimento non valido.')
            local_path(folder, ref['path'])
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'Manifesto non leggibile in {folder.name}: {exc}') from exc


def effective_status(folder, job):
    if job.get('path') and job.get('sha256'):
        path = local_path(folder, job['path']) if job.get('path') else None
        if path and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == job.get('sha256'):
            return 'completed'
        return 'recoverable' if job.get('remote') else ('failed' if job['status'] in {'completed', 'failed'} else 'uncertain')
    if job['status'] == 'running':
        return 'recoverable' if job.get('remote') else 'uncertain'
    return job['status']


def is_running(folder):
    try:
        with run_lock(folder):
            return False
    except ValueError:
        return True


def run_saved(folder, api_keys, *, selected=None, regenerate=False, on_progress=None, provider=None):
    """Default: pending + recuperabili. Rigenerazioni solo su selezione esplicita."""
    with run_lock(folder):
        data = load_run(folder)
        mutex = threading.Lock()

        def update(job, **fields):
            with mutex:
                job.update(fields)
                job['updated_at'] = now()
                job['events'].append({'at': job['updated_at'], 'status': job['status'],
                                      'execution': job['executions'], 'phase': job.get('phase'),
                                      'remote_id': (job.get('remote') or {}).get('id'),
                                      'phase_attempts': dict(job.get('phase_attempts', {})), 'error': job.get('error')})
                data['updated_at'] = now()
                atomic_json(folder / 'run.json', data)

        for job in data['jobs']:
            state = effective_status(folder, job)
            if state != job['status']:
                update(job, status=state, error=None if state == 'completed' else 'Interruzione o file risultato mancante/non integro.')
        selected_ids = set(selected) if selected is not None else None
        work = [j for j in data['jobs'] if j['status'] != 'completed' and
                ((selected_ids is None and j['status'] in {'pending', 'recoverable'}) or
                 (selected_ids is not None and j['id'] in selected_ids and
                  (j['status'] != 'uncertain' or regenerate)))]
        if not work:
            return data
        refs = []
        for ref in data['references']:
            raw = local_path(folder, ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('Immagine di riferimento modificata: ripresa interrotta.')
            refs.append(raw)
        params = GenerationParams(**data['params'])
        provider = provider or create_provider(ModelSpec(**data['model']), api_keys)
        attempts = data['attempts_per_phase']
        names = {p.stem.casefold() for p in folder.iterdir()}
        names.update(j['name'].casefold() for j in data['jobs'])

        def worker(job):
            original = PromptJob(job['name'], job['prompt'], job['source_row'])
            image = None

            def on_attempt(phase, number):
                counts = dict(job.get('phase_attempts', {}))
                counts[phase] = counts.get(phase, 0) + 1
                update(job, phase=phase, phase_attempts=counts)

            try:
                if regenerate:
                    update(job, remote=None, path=None, sha256=None)
                update(job, status='running', executions=job['executions'] + 1, error=None)
                if isinstance(provider, BFLProvider):
                    image = provider.generate_with_retries(
                        job['prompt'], refs, params, attempts, resume=job.get('remote'),
                        checkpoint=lambda remote: update(job, status='recoverable', remote=remote),
                        on_attempt=on_attempt,
                    )
                else:
                    image = provider.generate_with_retries(job['prompt'], refs, params, attempts, on_attempt=on_attempt)
                if image.extension not in {'png', 'jpeg', 'jpg', 'webp'}:
                    raise ValueError('Estensione immagine non valida.')
                with mutex:
                    # Conserva il nome se libero, altrimenti riserva un nuovo suffisso.
                    filename = f"{job['name']}.{image.extension}"
                    if any(p.name.casefold() == filename.casefold() for p in folder.iterdir()):
                        filename = f"{unique_filename(job['name'], names)}.{image.extension}"
                path = local_path(folder, filename)
                update(job, status='saving', path=filename, sha256=hashlib.sha256(image.image_bytes).hexdigest(), meta=image.meta)
                retry_call(lambda: _save_image(path, image.image_bytes), attempts,
                           lambda exc: isinstance(exc, OSError) and not isinstance(exc, FileExistsError),
                           on_attempt=lambda n: on_attempt('saving', n))
                update(job, status='completed', error=None)
                return JobResult(original, path=path, meta=image.meta)
            except Exception as exc:
                state = 'failed' if isinstance(exc, RejectedGenerationError) else (
                    'recoverable' if job.get('remote') else 'uncertain')
                update(job, status=state, error=str(exc))
                return JobResult(original, error=str(exc))

        with ThreadPoolExecutor(max_workers=max(1, min(16, data['concurrency']))) as pool:
            futures = [pool.submit(worker, job) for job in work]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if on_progress:
                    on_progress(count, len(work), result)
        return data
