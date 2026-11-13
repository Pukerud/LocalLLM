import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('v0213_speed_bench', Path(__file__).with_name('v0213_speed_bench.py'))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class QuietThread:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def join(self, **kwargs): pass


class Reply(io.BytesIO):
    status = 200


class Connection:
    sock = None
    sent = []
    cache = 0
    def __init__(self, *args, **kwargs): pass
    def request(self, *args, **kwargs): self.sent.append((args, kwargs))
    def close(self): pass
    def getresponse(self):
        chunks = [dict(choices=[dict(delta={'content': 'A complete test reply.'}, finish_reason=None)]),
                  dict(choices=[dict(delta={}, finish_reason='length')],
                       usage={'prompt_tokens': 256, 'completion_tokens': 128},
                       timings={'cache_n': self.cache, 'prompt_n': 256, 'prompt_per_second': 1234.5,
                                'predicted_n': 128, 'predicted_per_second': 99.5})]
        return Reply(b''.join(b'data: ' + json.dumps(x).encode() + b'\n\n' for x in chunks) + b'data: [DONE]\n\n')


class SpeedBench(unittest.TestCase):
    def test_nonce_is_first_and_distinct_and_reproducible(self):
        one = B.make_messages('short-r1', 'same body')
        two = B.make_messages('short-r2', 'same body')
        self.assertEqual(one, B.make_messages('short-r1', 'same body'))
        self.assertNotEqual(one[0]['content'][:64], two[0]['content'][:64])
        self.assertEqual(one[1], two[1])
        self.assertEqual(len(one[0]['content'].splitlines()[0]), 64)

    def test_counting_route_is_cpu_only_disabled_thinking(self):
        messages = B.make_messages('count', 'text')
        with patch.object(B, 'api', return_value={'input_tokens': 131072}) as api:
            self.assertEqual(B.token_count(messages, '0.1.46'), 131072)
        path, body = api.call_args.args
        self.assertEqual(path, '/v1/messages/count_tokens')
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['messages'], messages[1:])
        self.assertEqual(body['system'], messages[0]['content'])

    def test_unknown_occupancy_refuses(self):
        for value in (None, True, '0', -1):
            with self.assertRaises(RuntimeError):
                B.running({'live': {'running': value}})
        self.assertEqual(B.running({'live': {'running': 0}}), 0)

    def test_target_uses_actual_context_schema_and_refuses_wrong_model(self):
        status = {'loaded': True, 'model': B.OLD.MODEL_ID, 'engine': '0.1.46',
                  'context': {'native': 262144, 'max_positions': 262144}, 'concurrency': {'serving': 2}}
        with patch.object(B, 'api', return_value=status):
            self.assertEqual(B.target('0.1.46'), status)
            with self.assertRaises(RuntimeError): B.target('0.1.59')
        status['context']['max_positions'] = 524288
        with patch.object(B, 'api', return_value=status), self.assertRaises(RuntimeError): B.target('0.1.46')

    def test_candidate_permission_precedes_every_http_request(self):
        with patch.object(B.NEW, 'authorize_activation', side_effect=RuntimeError('locked')), \
             patch.object(B, 'api') as api, self.assertRaisesRegex(RuntimeError, 'locked'):
            B.run('0.1.59', Path('/nonexistent'))
        api.assert_not_called()

    def test_sampling_is_request_local_projection_real_key(self):
        self.assertEqual(B.SAMPLING['max_tokens'], 128)
        self.assertEqual(B.SAMPLING['experimental_speed_projection'], True)
        self.assertNotIn('strata_projection', B.SAMPLING)
        self.assertEqual(B.SAMPLING['chat_template_kwargs'], {'enable_thinking': False})

    def test_exact_fit_and_hash_with_mock_cpu_counter(self):
        with patch.object(B, 'token_count', side_effect=lambda messages, version: 99 + messages[1]['content'].count(' q')):
            case = B.fit('short-r1', 256, 'neutral')
        self.assertEqual(case['input_tokens'], 256)
        self.assertEqual(case['messages_sha256'], B.digest(case['messages']))

    def test_mock_stream_complete_and_cached_sample_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(B, 'idle'), \
             patch.object(B.http.client, 'HTTPConnection', Connection), \
             patch.object(B.threading, 'Thread', QuietThread):
            root = Path(folder)
            Connection.cache = 0
            result = B.stream(B.make_messages('s', 'text'), '0.1.46', root, 'ok', root/'abort')
            self.assertEqual(result['usage']['completion_tokens'], 128)
            self.assertEqual(result['timings']['cache_n'], 0)
            Connection.cache = 12
            try:
                with self.assertRaisesRegex(RuntimeError, 'KV reuse'):
                    B.stream(B.make_messages('s2', 'text'), '0.1.46', root, 'cached', root/'abort')
                self.assertTrue((root/'cached.json').exists())
            finally:
                Connection.cache = 0


if __name__ == '__main__': unittest.main()
