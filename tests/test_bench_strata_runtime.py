"""CPU HTTP/SSE benchmark-client checks, not model performance validation."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import sys
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bench_strata_runtime as bench


class Client(unittest.TestCase):
    def setUp(self):
        owner=self;self.seen=[];self.usage=True
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));owner.seen.append(body)
                if not body['stream']:
                    data=json.dumps({'choices':[{'message':{'content':'ANSWER=42'}}]}).encode()
                    self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
                events=[{'choices':[{'delta':{'role':'assistant'}}]},
                        {'choices':[{'delta':{'reasoning_content':'Think.'}}]},
                        {'choices':[{'delta':{'content':'ANSWER=42'},'finish_reason':'stop'}]}]
                if owner.usage:events.append({'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':10},
                                           'timings':{'predicted_per_second':100}})
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for event in events:
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush();time.sleep(.01)
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.client=bench.Client(f'http://127.0.0.1:{self.server.server_port}','fixture')
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
    def test_per_request_smoke_effort_override_leaves_sampling_normal(self):
        self.client.post({'messages':[],'reasoning_effort':'none','max_tokens':64})
        self.assertEqual(self.seen[0]['reasoning_effort'],'none');self.assertEqual(self.seen[0]['temperature'],1)
        self.assertEqual(self.seen[0]['top_p'],.95);self.assertEqual(self.seen[0]['top_k'],20)
    def test_ttft_requires_actual_content_or_thinking_not_role_header(self):
        d=self.client.post({'messages':[],'max_tokens':64},stream=True)
        self.assertGreater(d['ttft_seconds'],.005);self.assertLess(d['ttft_seconds'],d['seconds'])
        self.assertEqual(d['usage']['completion_tokens'],10);self.assertEqual(d['reasoning_chars'],6)
    def test_missing_usage_is_not_presented_as_measured_throughput(self):
        self.usage=False
        with self.assertRaisesRegex(RuntimeError,'usage'):self.client.post({'messages':[],'max_tokens':64},stream=True)
    def test_swapped_images_have_same_shape_and_different_pixels(self):
        import base64
        red=base64.b64decode(bench.image_uri('red','blue').split(',')[1]);blue=base64.b64decode(bench.image_uri('blue','red').split(',')[1])
        self.assertEqual(struct.unpack('>II',red[16:24]),(256,128));self.assertEqual(red[:24],blue[:24]);self.assertNotEqual(red,blue)


if __name__=='__main__':unittest.main()
