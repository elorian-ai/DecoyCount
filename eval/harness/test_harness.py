import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
from PIL import Image

from normalize_images import one
from providers import PROVIDERS, ProviderError, build_request, parse_model_spec
from run_counting import load_tasks, parse_int
from score import load, summarize


class HarnessTests(unittest.TestCase):
    def test_answer_extraction(self):
        for text, expected in [('5', 5), ('**13**', 13), ('1,599', 1599),
                               ('`42`', 42), ('0', 0), ('', None),
                               ('I cannot count them.', None),
                               ('Total: **11**\nNeed 16 - 11 = 5\n\n**5**', 5),
                               ('The answer is 12.\nLet me know!', 12)]:
            with self.subTest(text=text):
                self.assertEqual(parse_int(text)[0], expected)

    def test_provider_identity_and_dropped_effort(self):
        self.assertEqual(parse_model_spec('test=gateway:model'), ('test', 'gateway', 'model'))
        with self.assertRaises(ProviderError):
            parse_model_spec('test=model')
        kwargs, dropped, effective = build_request(PROVIDERS['dashscope'], 'model', [], 'high', 20, 30)
        self.assertNotIn('reasoning_effort', kwargs)
        self.assertEqual((dropped, effective), (['reasoning_effort'], 'none'))

    def test_local_evaluation_and_scoring(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                content = body['messages'][0]['content']
                calls.append(content)
                answer = '2' if content[0]['text'] == 'Scenario question' else '5'
                data = json.dumps({'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
                                   'model': 'synthetic', 'choices': [{'index': 0, 'finish_reason': 'stop',
                                   'message': {'role': 'assistant', 'content': answer}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            images = root / 'images'
            images.mkdir()
            for sub, role in [('image', 'primary'), ('image_secondary', 'secondary')]:
                folder = root / 'tasks' / 'Task-synthetic' / sub
                folder.mkdir(parents=True)
                Image.new('RGB', (12, 8), 'white').save(folder / 'synthetic.png')
                self.assertEqual(one((str(folder.parent), sub, role, str(images)))['status'], 'ok')
            row = dict(task_name='Task synthetic', studio_id='synthetic', neutral_prompt='Count objects',
                       cued_prompt='Count actual objects', contextual_prompt='Scenario question',
                       true_count=5, contextual_count=2, signature_count=json.dumps([
                           {'sig_count': 7, 'sig_category': 'Text', 'sig_note': 'Synthetic decoy'}]),
                       scene_domain='synthetic', image_origin='synthetic')
            pd.DataFrame([row]).to_csv(root / 'index.csv', index=False)
            self.assertEqual(len(load_tasks(str(root), str(images))), 1)
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                env = dict(os.environ, GATEWAY_API_KEY='synthetic-test-key',
                           GATEWAY_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1')
                runner = pathlib.Path(__file__).with_name('run_counting.py')
                command = [sys.executable, str(runner), '--repo', str(root), '--imgdir', str(images),
                           '--models', 'synthetic=gateway:synthetic', '--out', str(root / 'results'),
                           '--workers', '1', '--retries', '0', '--effort', 'none']
                subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=40)
                records = [json.loads(line) for line in (root / 'results/responses.jsonl').read_text().splitlines()]
                self.assertEqual([r['n_images'] for r in records], [1, 1, 2])
                self.assertEqual([r['condition'] for r in records], ['neutral', 'cued', 'contextual'])
                self.assertTrue(all(r['correct'] for r in records))
                self.assertEqual(len(calls), 3)
                self.assertEqual(summarize(load([str(root / 'results')]))['acc'].tolist(), [100, 100, 100])
                subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=40)
                self.assertEqual(len(calls), 3, 'Resume must not repeat settled requests')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            (images / 'Task-synthetic__primary.jpg').unlink()
            with self.assertRaises(FileNotFoundError):
                load_tasks(str(root), str(images))

    def test_inapplicable_rows_are_not_counted_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {'model': 'synthetic', 'task_dir': 'Task-synthetic', 'condition': 'contextual',
                   'applicable': False, 'raw': '', 'gt': None}
            pathlib.Path(directory, 'responses.jsonl').write_text(json.dumps(row) + '\n')
            self.assertTrue(load([directory]).empty)


if __name__ == '__main__':
    unittest.main()
