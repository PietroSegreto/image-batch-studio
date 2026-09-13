import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import requests
from config import MODELS
from history import create_run, load_run, run_saved, effective_status, atomic_json, run_lock, is_running
from prompt_files import PromptJob
from providers.base import GeneratedImage, GenerationParams
from providers.bfl_provider import BFLProvider
from providers.retry import RejectedGenerationError
from test_regressions import FakeProvider, response


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'run'
        self.jobs = [PromptJob('first', 'first prompt', 1), PromptJob('second', 'second prompt', 2)]
        self.spec = MODELS['nano-banana-2-flash']
        self.params = GenerationParams(resolution='2K')
        patch('providers.retry.time.sleep').start()
        self.addCleanup(patch.stopall)

    def create(self, **kwargs):
        return create_run(self.folder, self.spec, self.params, self.jobs, kwargs.get('refs', []), concurrency=1)

    def save(self, data):
        atomic_json(self.folder / 'run.json', data)

    def test_manifest_contains_configuration_and_reference_bytes(self):
        self.create(refs=[b'reference'])
        data = load_run(self.folder)
        self.assertEqual(data['params'], asdict(self.params))
        self.assertEqual(data['model']['model_id'], self.spec.model_id)
        self.assertEqual((self.folder / data['references'][0]['path']).read_bytes(), b'reference')
        self.assertNotIn('api_key', json.dumps(data))

    def test_restart_skips_completed_jobs_and_restores_settings(self):
        self.create(refs=[b'reference'])
        first = FakeProvider()
        run_saved(self.folder, {}, selected=['0'], provider=first)
        second = FakeProvider()
        data = run_saved(self.folder, {}, provider=second)
        self.assertEqual(second.generate.call_count, 1)
        prompt, refs, params = second.generate.call_args.args
        self.assertEqual((prompt, refs, params.resolution), ('second prompt', [b'reference'], '2K'))
        self.assertTrue(all(j['status'] == 'completed' for j in data['jobs']))
        self.assertEqual([j['executions'] for j in data['jobs']], [1, 1])

    def test_interrupted_job_not_automatically_regenerated(self):
        data = self.create()
        data['jobs'][0]['status'] = 'running'
        self.save(data)
        provider = FakeProvider()
        data = run_saved(self.folder, {}, provider=provider)
        self.assertEqual(data['jobs'][0]['status'], 'uncertain')
        self.assertEqual(provider.generate.call_count, 1)
        run_saved(self.folder, {}, selected=['0'], provider=provider)
        self.assertEqual(provider.generate.call_count, 1)
        data = run_saved(self.folder, {}, selected=['0'], regenerate=True, provider=provider)
        self.assertEqual(provider.generate.call_count, 2)
        self.assertEqual(data['jobs'][0]['status'], 'completed')

    def test_timeout_is_uncertain_and_rejected_is_failed(self):
        self.create()
        provider = FakeProvider([requests.ReadTimeout('uncertain'), RejectedGenerationError('invalid')])
        data = run_saved(self.folder, {}, provider=provider)
        self.assertEqual([j['status'] for j in data['jobs']], ['uncertain', 'failed'])
        retry = FakeProvider()
        data = run_saved(self.folder, {}, selected=['1'], provider=retry)
        self.assertEqual(retry.generate.call_count, 1)
        self.assertEqual(data['jobs'][1]['executions'], 2)

    def test_existing_saved_image_reconciled_after_interruption(self):
        data = self.create()
        (self.folder / 'first.png').write_bytes(b'image')
        data['jobs'][0].update(status='saving', path='first.png', sha256=hashlib.sha256(b'image').hexdigest())
        self.save(data)
        provider = FakeProvider()
        data = run_saved(self.folder, {}, provider=provider)
        self.assertEqual(provider.generate.call_count, 1)
        self.assertEqual(data['jobs'][0]['status'], 'completed')

    def test_missing_completed_image_requires_explicit_retry(self):
        self.create()
        data = run_saved(self.folder, {}, provider=FakeProvider())
        (self.folder / data['jobs'][0]['path']).unlink()
        provider = FakeProvider()
        data = run_saved(self.folder, {}, provider=provider)
        self.assertEqual(provider.generate.call_count, 0)
        self.assertEqual(data['jobs'][0]['status'], 'failed')
        data = run_saved(self.folder, {}, selected=['0'], provider=provider)
        self.assertEqual(provider.generate.call_count, 1)
        self.assertEqual(data['jobs'][0]['status'], 'completed')

    def test_bfl_checkpoint_survives_poll_failure_and_resumes_same_request(self):
        self.spec = MODELS['flux-2-pro']
        self.jobs = self.jobs[:1]
        self.create()
        provider = BFLProvider('test', self.spec)
        with patch('providers.bfl_provider.requests.post', return_value=response(payload={
            'id': 'original', 'polling_url': 'https://example.test/poll'})) as post, patch(
            'providers.bfl_provider.requests.get', side_effect=requests.ReadTimeout()):
            data = run_saved(self.folder, {}, provider=provider)
        self.assertEqual(data['jobs'][0]['remote']['id'], 'original')
        self.assertEqual(data['jobs'][0]['status'], 'recoverable')
        with patch('providers.bfl_provider.requests.post') as post, patch(
            'providers.bfl_provider.requests.get', side_effect=[
                response(payload={'status': 'Ready', 'result': {'sample': 'https://example.test/image'}}), response()]):
            data = run_saved(self.folder, {}, provider=BFLProvider('test', self.spec))
        post.assert_not_called()
        self.assertEqual(data['jobs'][0]['status'], 'completed')
        self.assertEqual(data['jobs'][0]['phase_attempts'], {'generation': 1, 'polling': 4, 'download': 1, 'saving': 1})

    def test_corrupt_manifest_never_starts_provider(self):
        self.create()
        (self.folder / 'run.json').write_text('{broken', encoding='utf-8')
        provider = FakeProvider()
        with self.assertRaises(ValueError):
            run_saved(self.folder, {}, provider=provider)
        provider.generate.assert_not_called()

    def test_modified_reference_blocks_generation(self):
        data = self.create(refs=[b'reference'])
        (self.folder / data['references'][0]['path']).write_bytes(b'changed')
        provider = FakeProvider()
        with self.assertRaises(ValueError):
            run_saved(self.folder, {}, provider=provider)
        provider.generate.assert_not_called()

    def test_manifest_replacement_failure_preserves_previous_json(self):
        data = self.create()
        with patch('history.os.replace', side_effect=OSError('busy')):
            with self.assertRaises(OSError):
                atomic_json(self.folder / 'run.json', {'broken': True})
        self.assertEqual(load_run(self.folder), json.loads(json.dumps(data)))
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_run_lock_blocks_second_session_and_releases(self):
        self.create()
        with run_lock(self.folder):
            self.assertTrue(is_running(self.folder))
            with self.assertRaises(ValueError):
                run_saved(self.folder, {}, provider=FakeProvider())
        self.assertFalse(is_running(self.folder))

    def test_history_ui_with_saved_run(self):
        patch.stopall()
        from streamlit.testing.v1 import AppTest
        app = Path(__file__).resolve().parents[1] / 'app.py'
        root = Path(self.temp.name)
        self.folder = root / 'output' / 'saved-run'
        self.create()
        previous = Path.cwd()
        try:
            os.chdir(root)
            at = AppTest.from_file(str(app)).run(timeout=30)
            self.assertFalse(at.exception, [e.message for e in at.exception])
            at.selectbox(key='history_choice').set_value('saved-run').run(timeout=30)
            self.assertFalse(at.exception, [e.message for e in at.exception])
            self.assertTrue(any('Riprendi batch' in b.label for b in at.button))
        finally:
            os.chdir(previous)

    def test_end_to_end_ui_generation_and_new_session(self):
        patch.stopall()
        import io
        from PIL import Image
        from streamlit.testing.v1 import AppTest
        app = Path(__file__).resolve().parents[1] / 'app.py'
        stream = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(stream, format='PNG')
        provider = FakeProvider()
        provider.generate.return_value = GeneratedImage(stream.getvalue())
        previous = Path.cwd()
        try:
            os.chdir(self.temp.name)
            with patch('providers.create_provider', return_value=provider):
                at = AppTest.from_file(str(app)).run(timeout=30)
                next(t for t in at.text_area if t.label == 'Prompt').set_value('saved prompt')
                next(b for b in at.button if b.label.endswith(' Genera')).click().run(timeout=30)
            self.assertFalse(at.exception, [e.message for e in at.exception])
            manifests = list(Path('output').glob('*/run.json'))
            self.assertEqual(len(manifests), 1)
            folder = manifests[0].parent
            self.assertEqual(load_run(folder)['jobs'][0]['status'], 'completed')
            fresh = AppTest.from_file(str(app)).run(timeout=30)
            fresh.selectbox(key='history_choice').set_value(folder.name).run(timeout=30)
            self.assertFalse(fresh.exception, [e.message for e in fresh.exception])
            self.assertEqual(provider.generate.call_count, 1)
            next(b for b in fresh.button if b.label == 'Chiudi visualizzazione').click().run(timeout=30)
            self.assertFalse(fresh.exception, [e.message for e in fresh.exception])
            self.assertTrue(manifests[0].exists())
        finally:
            os.chdir(previous)

    def test_invalid_schema_and_escaping_path_rejected(self):
        original = self.create()
        import copy
        for mutate in [lambda d: d.pop('created_at'),
                       lambda d: d['jobs'][0].update(status='completed'),
                       lambda d: d['jobs'][0].update(path='../outside.png')]:
            data = copy.deepcopy(original)
            mutate(data)
            self.save(data)
            with self.assertRaises(ValueError):
                load_run(self.folder)

    def test_completed_checkpoint_failure_is_reconciled_without_generation(self):
        self.jobs = self.jobs[:1]
        self.create()
        original = atomic_json
        def interrupted(path, data):
            if data['jobs'][0]['status'] == 'completed':
                raise OSError('manifest write interrupted')
            original(path, data)
        with patch('history.atomic_json', side_effect=interrupted):
            run_saved(self.folder, {}, provider=FakeProvider())
        provider = FakeProvider()
        data = run_saved(self.folder, {}, provider=provider)
        provider.generate.assert_not_called()
        self.assertEqual(data['jobs'][0]['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
