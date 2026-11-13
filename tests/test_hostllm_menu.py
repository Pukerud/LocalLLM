"""Offline UI regressions; no real lifecycle operation or model generation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostllm_menu as M
from menu_render_fixture import DATA


class Menu(unittest.TestCase):
    def test_unique_process_not_one_line_per_gpu(self):
        raw = '\n'.join(['110987, /some/private/runtime/strata'] * 4)
        self.assertEqual(M.compute_summary(raw), '1 process (strata)')
        self.assertNotIn('110987', M.compute_summary(raw))
        self.assertNotIn('/some/private', M.compute_summary(raw))
        self.assertEqual(M.compute_summary(None), 'UNKNOWN')
        self.assertEqual(M.compute_summary(''), 'idle')

    def test_two_processes_stay_two(self):
        self.assertEqual(M.compute_summary('1, /bin/strata\n1, /bin/strata\n2, /bin/renderer'), '2 processes (renderer, strata)')

    def test_terminal_escape_from_process_name_cannot_inject_colour(self):
        self.assertNotIn('\x1b', M.compute_summary('1, /bin/evil\x1b[31m'))

    def test_modern_and_plain_layouts_have_exact_width(self):
        for width in (42, 54, 64, 80, 92):
            for color in (False, True):
                for unicode in (False, True):
                    with self.subTest(width=width, color=color, unicode=unicode):
                        text = M.render_menu(DATA, width=width, color=color, unicode=unicode)
                        self.assertTrue(all(M.cells(line) == width for line in text.splitlines()))
                        if not color:
                            self.assertNotIn('\x1b', text)

    def test_all_stable_ids_and_clear_lock_are_visible(self):
        for width in (54, 80, 92):
            text = M.render_menu(DATA, width=width, color=False, unicode=False)
            for n in (1, 7, 9, 10, 11, 12, 18, 21, 22):
                self.assertIn(f'[{n}]', text)
            self.assertIn('LOCKED', text)
            self.assertIn('SELECTED', text)
            self.assertNotIn('0.1.46/source/engine', text)

    def test_selected_is_hive_selection_not_claim_about_running_model(self):
        d = copy.deepcopy(DATA)
        d['selected'] = 'swift15u-q8'
        text = M.render_menu(d, color=False, unicode=False)
        selected = next(line for line in text.splitlines() if 'SELECTED' in line)
        self.assertIn('[12]', selected)
        self.assertIn(DATA['running'], text)

    def test_busy_and_unknown_states_are_not_hidden(self):
        for state in ('BUSY', 'UNKNOWN'):
            d = {**DATA, 'docker': state, 'admission': 'unknown', 'gpus': None}
            text = M.render_menu(d, color=False, unicode=False)
            self.assertIn(state, text)
            self.assertIn('VRAM    UNKNOWN', text)

    def test_authorized_trial_is_not_mislabeled_locked_or_qualified(self):
        d = {**DATA, 'candidate_authorized': True, 'selected': M.CANDIDATE}
        for width in (54, 66, 70, 80):
            text = M.render_menu(d, width=width, color=False, unicode=False)
            self.assertIn('TRIAL AUTHORIZED', text)
            self.assertIn('SELECTED', text)
            self.assertNotIn('LOCKED', text)
            self.assertNotIn('QUALIFIED', text)

    def test_draft_is_not_mislabeled_prepared(self):
        text = M.render_menu({**DATA, 'prepared': False}, color=False, unicode=False)
        self.assertIn('DRAFT', text)
        self.assertNotIn('PREPARED', text)

    def test_render_is_pure_no_external_command_or_live_state(self):
        with patch.object(M, 'query', side_effect=AssertionError('No real command')), patch.object(M, 'selected_profile', side_effect=AssertionError('No real state')):
            M.render_menu(DATA)

    def test_colour_clipping_preserves_width_and_resets(self):
        text = '\x1b[31m' + 'A' * 100 + '\x1b[0m'
        result = M.clip(text, 10)
        self.assertEqual(M.cells(result), 10)
        self.assertTrue(result.endswith('\x1b[0m'))


if __name__ == '__main__':
    unittest.main()
