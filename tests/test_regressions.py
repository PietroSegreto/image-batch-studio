import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from config import MODELS
from prompt_files import PromptJob, build_jobs, sanitize_filename
from providers.base import GeneratedImage, GenerationParams, ImageProvider
from providers.bfl_provider import BFLProvider
from providers.openai_provider import OpenAIProvider
from providers.gemini_provider import GeminiProvider
from providers.retry import RetryableGenerationError
from runner import _save_image, run_jobs


class FakeProvider(ImageProvider):
    generate = Mock()

    def __init__(self, effects=None):
        self.generate = Mock(side_effect=effects, return_value=GeneratedImage(b'image'))


def response(status=200, payload=None, content=b'image'):
    r = requests.Response()
    r.status_code = status
    r._content = content
    r.json = Mock(return_value=payload)
    return r


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.sleep = patch('providers.retry.time.sleep').start()
        self.addCleanup(patch.stopall)
        self.jobs = [PromptJob('image', 'a prompt', 1)]
        self.params = GenerationParams()

    def run_batch(self, provider, jobs=None, **kwargs):
        return run_jobs(provider, jobs or self.jobs, [], self.params, self.folder, **kwargs)

    def test_parser_resolves_suffix_and_case_collisions(self):
        rows = [[name, 'a prompt'] for name in ['a', 'a', 'a_2', 'A']]
        jobs, issues = build_jobs(rows, has_header=False, name_col=0, prompt_cols=[1])
        self.assertEqual(len({j.name.casefold() for j in jobs}), 4)
        self.assertEqual(len(issues), 3)

    def test_windows_reserved_names_and_truncation(self):
        for name in ['CON', 'nul.png', 'COM1', 'LPT9', 'COM¹', 'CONIN$']:
            self.assertTrue(sanitize_filename(name).startswith('_'), name)
        self.assertFalse(sanitize_filename('a' * 99 + '.' + 'b').endswith('.'))

    def test_concurrent_duplicate_jobs_and_existing_file(self):
        existing = self.folder / 'IMAGE.png'
        existing.write_bytes(b'keep')
        jobs = [PromptJob(n, n, i) for i, n in enumerate(['image', 'image', 'image_2', 'CON'])]
        results = self.run_batch(FakeProvider(), jobs, concurrency=4)
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(len({r.path.name.casefold() for r in results}), 4)
        self.assertEqual(existing.read_bytes(), b'keep')
        self.assertEqual(len(list(self.folder.glob('*.png'))), 5)
        self.assertEqual([r.job.prompt for r in results], [j.prompt for j in jobs])

    def test_exclusive_write_preserves_existing_file(self):
        path = self.folder / 'image.png'
        path.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            _save_image(path, b'new')
        self.assertEqual(path.read_bytes(), b'keep')

    def test_failed_partial_write_is_removed(self):
        path = self.folder / 'image.png'
        real_open = Path.open
        def broken_open(target, *args, **kwargs):
            stream = real_open(target, *args, **kwargs)
            wrapper = Mock(wraps=stream)
            def fail(data):
                stream.write(data[:2])
                raise OSError(errno.ENOSPC, 'disk full')
            wrapper.write.side_effect = fail
            context = Mock()
            context.__enter__ = Mock(return_value=wrapper)
            context.__exit__ = Mock(side_effect=lambda *args: stream.close())
            return context
        with patch.object(Path, 'open', broken_open):
            with self.assertRaises(OSError):
                _save_image(path, b'image')
        self.assertFalse(path.exists())

    def test_save_retry_never_regenerates(self):
        provider = FakeProvider()
        with patch('runner._save_image', side_effect=[OSError('busy'), None]) as save:
            results = self.run_batch(provider)
        self.assertTrue(results[0].ok)
        self.assertEqual(save.call_count, 2)
        self.assertEqual(provider.generate.call_count, 1)

    def test_disk_failure_retains_generated_bytes(self):
        provider = FakeProvider()
        with patch('runner._save_image', side_effect=OSError('disk full')) as save:
            result = self.run_batch(provider)[0]
        self.assertFalse(result.ok)
        self.assertEqual(result.image.image_bytes, b'image')
        self.assertEqual(save.call_count, 3)
        self.assertEqual(provider.generate.call_count, 1)

    def test_generation_retry_only_for_explicit_transient_rejection(self):
        provider = FakeProvider([RetryableGenerationError('429'), GeneratedImage(b'ok')])
        self.assertTrue(self.run_batch(provider)[0].ok)
        self.assertEqual(provider.generate.call_count, 2)
        for exc in [ValueError('invalid'), RuntimeError('moderated'), requests.ReadTimeout('uncertain')]:
            provider = FakeProvider([exc])
            self.assertFalse(self.run_batch(provider)[0].ok)
            self.assertEqual(provider.generate.call_count, 1)

    def test_bfl_poll_and_download_retry_without_resubmission(self):
        provider = BFLProvider('test', MODELS['flux-2-pro'])
        submit = response(payload={'id': 'job-1', 'polling_url': 'https://example.test/poll'})
        ready = response(payload={'status': 'Ready', 'result': {'sample': 'https://example.test/image'}})
        with patch('providers.bfl_provider.requests.post', return_value=submit) as post, patch(
            'providers.bfl_provider.requests.get', side_effect=[
                requests.ReadTimeout(), ready, response(503), response(),
            ],
        ) as get:
            result = self.run_batch(provider)[0]
        self.assertTrue(result.ok)
        self.assertEqual(post.call_count, 1)
        self.assertEqual([c.args[0] for c in get.call_args_list],
                         ['https://example.test/poll'] * 2 + ['https://example.test/image'] * 2)

    def test_bfl_exhausted_poll_and_moderation_do_not_resubmit(self):
        for effect, expected_calls in [(requests.ReadTimeout(), 3),
                (response(payload={'status': 'Content Moderated'}), 1)]:
            provider = BFLProvider('test', MODELS['flux-2-pro'])
            with patch('providers.bfl_provider.requests.post', return_value=response(payload={
                'polling_url': 'https://example.test/poll'})) as post, patch(
                'providers.bfl_provider.requests.get', side_effect=effect if isinstance(effect, Exception) else None,
                return_value=effect,
            ) as get:
                self.assertFalse(self.run_batch(provider)[0].ok)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(get.call_count, expected_calls)

    def test_openai_http_errors_are_classified(self):
        for status, expected_calls in [(400, 1), (401, 1), (429, 3), (500, 1)]:
            provider = OpenAIProvider('test', MODELS['gpt-image-2'])
            with patch('providers.openai_provider.requests.post', return_value=response(status, {})) as post:
                self.assertFalse(self.run_batch(provider)[0].ok)
            self.assertEqual(post.call_count, expected_calls)

    def test_bfl_submit_retries_only_explicit_rejection(self):
        provider = BFLProvider('test', MODELS['flux-2-pro'])
        with patch('providers.bfl_provider.requests.post', side_effect=[
            response(429), response(payload={'polling_url': 'https://example.test/poll'}),
        ]) as post, patch('providers.bfl_provider.requests.get', side_effect=[
            response(payload={'status': 'Ready', 'result': {'sample': 'https://example.test/image'}}),
            response(),
        ]):
            self.assertTrue(self.run_batch(provider)[0].ok)
        self.assertEqual(post.call_count, 2)

    def test_bfl_download_failure_does_not_resubmit(self):
        provider = BFLProvider('test', MODELS['flux-2-pro'])
        with patch('providers.bfl_provider.requests.post', return_value=response(payload={
            'polling_url': 'https://example.test/poll'})) as post, patch(
            'providers.bfl_provider.requests.get', side_effect=[
                response(payload={'status': 'Ready', 'result': {'sample': 'https://example.test/image'}}),
                requests.ReadTimeout(), requests.ReadTimeout(), requests.ReadTimeout(),
            ],
        ):
            self.assertFalse(self.run_batch(provider)[0].ok)
        self.assertEqual(post.call_count, 1)

    def test_gemini_error_classification(self):
        from google.genai.errors import ClientError, ServerError
        for error, calls in [(ClientError(429, {'message': 'busy'}), 3),
                             (ClientError(400, {'message': 'invalid'}), 1),
                             (ServerError(503, {'message': 'unavailable'}), 1)]:
            with patch('providers.gemini_provider.genai.Client') as client:
                provider = GeminiProvider('test', MODELS['nano-banana-2-flash'])
                generate = client.return_value.models.generate_content
                generate.side_effect = error
                self.assertFalse(self.run_batch(provider)[0].ok)
                self.assertEqual(generate.call_count, calls)

    def test_gemini_sdk_retry_disabled(self):
        with patch('providers.gemini_provider.genai.Client') as client:
            GeminiProvider('test', MODELS['nano-banana-2-flash'])
        self.assertEqual(client.call_args.kwargs['http_options'].retry_options.attempts, 1)


if __name__ == '__main__':
    unittest.main()
