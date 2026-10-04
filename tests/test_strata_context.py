"""Option 3 native-context regression checks using small CPU-only artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform != 'linux':
    raise unittest.SkipTest('Linux controller imports')
import strata_launcher as launcher
from engine_safety import SafetyError


class OriginalContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source-99f3dbd0b21d'
        self.pack = self.root / 'data/packs/iq3_s'
        def put(p):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
            return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        runtime = [put(self.source / 'engine/strata'), put(self.source / 'engine/strata-vision')]
        self.relatives = ['data/models/IQ3_S/shard1.gguf','data/models/IQ3_S/shard2.gguf','data/models/projector.gguf']
        assets = [put(self.root / p) for p in self.relatives]
        for p in [self.source / '.venv/bin/python',self.pack / 'dense.bin',self.pack / 'tokenizer/tokenizer.json',
                  self.root / 'data/mtp/rt/experts.bin']: put(p)
        (self.root / 'prepared.json').write_text(json.dumps({'source_commit':launcher.SOURCE_COMMIT,
                                                          'runtime_assets':runtime,'assets':assets}))
        self.cfg = {'exe':str(self.source / 'engine/strata'),'tokenizer':str(self.pack / 'tokenizer'),
                    'gpu':[0,1,2,3],'layer_split':'auto',
                    'vision':{'exe':str(self.source / 'engine/strata-vision'),'gpu':True,
                              'model':str(self.root / self.relatives[0]),'mmproj':str(self.root / self.relatives[2])},
                    'args':['--max-context','262144','--kv','int8','--kv-resident','32768','--spec','4',
                            '--spec-min-p','0.5','--vram-reserve-mib','2048','--vision','--pack',str(self.pack),
                            '--native',str(self.root / self.relatives[0]),'--ple-gguf',str(self.root / self.relatives[1]),
                            '--mtp',str(self.root / 'data/mtp/rt')]}
        self.path = self.source / 'strata-iq3_s.json'; self.save()
        self.patchers = [mock.patch.object(launcher,'data_root',return_value=self.root),
                         mock.patch.object(launcher,'command_output',return_value=launcher.SOURCE_COMMIT),
                         mock.patch.object(launcher,'DIGESTS',dict(zip(self.relatives,[a['sha256'] for a in assets])))]
        for p in self.patchers: p.start()

    def tearDown(self):
        for p in reversed(self.patchers): p.stop()
        self.tmp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.cfg))

    def test_option_three_remains_262144_with_32k_resident_window(self):
        args = launcher.ready('iq3_s')[2]['args']
        self.assertEqual(args[args.index('--max-context')+1],'262144')
        self.assertEqual(args[args.index('--kv-resident')+1],'32768')

    def test_128k_downgrade_refused(self):
        self.cfg['args'][self.cfg['args'].index('--max-context')+1]='131072'; self.save()
        with self.assertRaisesRegex(SafetyError,'max-context'): launcher.ready('iq3_s')


if __name__ == '__main__':
    unittest.main()
