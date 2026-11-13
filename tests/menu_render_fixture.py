"""Pure display fixture; no subprocess, GPU query, API or real state."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hostllm_menu as M

DATA = {'active': 'strata', 'running': 'Unsloth V3.1 IQ4_XS / Strata 0.1.46',
        'selected': 'unsloth-v31-iq4xs-q8-cvec-2slot-v0146', 'docker': 'clear', 'admission': 'active',
        'compute': '1 process (strata)', 'gpus': [(0, 23748, 24576), (1, 8480, 24576), (2, 22002, 24576), (3, 22950, 24576)],
        'prepared': True}
if __name__ == '__main__':
    print(M.render_menu(DATA, width=80, color=False, unicode=False))
