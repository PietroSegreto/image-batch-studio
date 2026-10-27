import base64
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from image_batch_studio.config import MODELS, OPENAI
from image_batch_studio.providers import create_provider
from image_batch_studio.providers.base import GenerationParams
from image_batch_studio.providers.openai_costs import estimate_output_cost, estimate_output_tokens
from image_batch_studio.providers.openai_provider import dims_for_aspect
from image_batch_studio.providers.retry import RejectedGenerationError

NEW_MODELS = ['gpt-image-2.5-sunburst', 'gpt-image-2.5-flare']


class OpenAIImageTests(unittest.TestCase):
    def test_new_models_generate_and_edit(self):
        for model in NEW_MODELS:
            for refs in ([], [b'\xff\xd8\xffreference']):
                for quality in ('xhigh', 'max'):
                    with self.subTest(model=model, refs=bool(refs), quality=quality):
                        response = Mock(status_code=200)
                        usage = {'output_tokens': 1000, 'input_tokens': 80,
                                 'input_tokens_details': {'text_tokens': 30, 'image_tokens': 50}}
                        response.json.return_value = {
                            'data': [{'b64_json': base64.b64encode(b'image').decode()}], 'usage': usage}
                        provider = create_provider(MODELS[model], {OPENAI: 'test'})
                        with patch('image_batch_studio.providers.openai_provider.requests.post', return_value=response) as post:
                            result = provider.generate('prompt', refs, GenerationParams(
                                quality=quality, resolution='1 MP', output_format='webp',
                                background='transparent', output_compression=85))
                        args, kwargs = post.call_args
                        self.assertTrue(args[0].endswith('/images/edits' if refs else '/images/generations'))
                        fields = kwargs['data' if refs else 'json']
                        self.assertEqual(fields['model'], model)
                        self.assertEqual(fields['quality'], quality)
                        self.assertEqual(fields['size'], '992x992')
                        self.assertEqual(fields['output_compression'], '85' if refs else 85)
                        self.assertEqual(fields['n'], '1' if refs else 1)
                        if refs:
                            self.assertEqual(kwargs['files'][0][1][2], 'image/jpeg')
                        self.assertEqual(result.image_bytes, b'image')
                        self.assertEqual(result.extension, 'webp')
                        self.assertEqual(result.meta['usage'], usage)
                        self.assertEqual(result.meta['output_cost_usd'], 0.03)

    def test_rejects_invalid_options_before_request(self):
        with patch('image_batch_studio.providers.openai_provider.requests.post') as post:
            provider = create_provider(MODELS['gpt-image-2'], {OPENAI: 'test'})
            for params in (GenerationParams(quality='max'),
                           GenerationParams(background='transparent', output_format='jpeg')):
                with self.assertRaises(RejectedGenerationError):
                    provider.generate('prompt', [], params)
            post.assert_not_called()

    def test_official_calculator_examples_and_model_difference(self):
        for model in NEW_MODELS:
            self.assertEqual(estimate_output_tokens(model, 'low', 1024, 1024), 196)
            self.assertEqual(estimate_output_tokens(model, 'medium', 1024, 1024), 439)
            self.assertEqual(estimate_output_tokens(model, 'high', 1024, 1024), 1756)
            self.assertAlmostEqual(estimate_output_cost(model, 'low', 1024, 1024)[0], 0.00588)
        self.assertEqual(estimate_output_tokens('gpt-image-2', 'medium', 1024, 1024), 1756)

    def test_auto_batch_scaling_and_orientation(self):
        for model in NEW_MODELS + ['gpt-image-2']:
            low, high = estimate_output_cost(model, 'auto', 1536, 1024, 6)
            self.assertAlmostEqual(low, estimate_output_cost(model, 'low', 1024, 1536)[0] * 6)
            max_quality = MODELS[model].qualities[-1]
            self.assertAlmostEqual(high, estimate_output_cost(model, max_quality, 1536, 1024)[0] * 6)
            self.assertGreater(high, low)

    def test_all_ui_sizes_are_valid(self):
        for model in NEW_MODELS + ['gpt-image-2']:
            spec = MODELS[model]
            for ratio in spec.aspect_ratios:
                for resolution in spec.resolutions:
                    width, height = dims_for_aspect(ratio, resolution)
                    self.assertGreater(estimate_output_tokens(model, 'low', width, height), 0)
        with self.assertRaises(ValueError):
            estimate_output_tokens(NEW_MODELS[0], 'low', 1000, 1000)

    def test_ui_model_switch_and_estimate_updates(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        for model in NEW_MODELS:
            app.sidebar.selectbox[0].select(model).run()
            self.assertFalse(app.exception)
            quality = next(w for w in app.sidebar.selectbox if w.label == 'Qualità')
            self.assertIn('max', quality.options)
            quality.select('max').run()
            next(w for w in app.number_input if w.label == 'Numero di varianti').set_value(3).run()
            self.assertFalse(app.exception)
            estimates = [w.value for w in app.info if 'Stima costo output' in w.value]
            self.assertEqual(len(estimates), 1)
            self.assertIn('per 3 immagini', estimates[0])
        app.sidebar.selectbox[0].select('gpt-image-2').run()
        self.assertFalse(app.exception)
        quality = next(w for w in app.sidebar.selectbox if w.label == 'Qualità')
        self.assertNotIn('max', quality.options)
