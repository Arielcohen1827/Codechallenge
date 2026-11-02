"""Historical audit inputs must travel with a clean/shallow checkout."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import deep_clipping_replay as replay


class HistoricalSupportTests(unittest.TestCase):
    def test_corrupt_support_is_rejected_before_loading_historical_code(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / 'broken.zip'
            with zipfile.ZipFile(broken, 'w') as archive:
                archive.writestr('provenance.json', json.dumps({'files': {'candidate.py': '0' * 64}}))
                archive.writestr('candidate.py', 'raise RuntimeError("must not execute")')
            with patch.object(replay, 'FROZEN_SUPPORT', broken), \
                    patch.object(replay, '_support_directory', None), \
                    self.assertRaisesRegex(ValueError, 'Historical support hash mismatch'):
                replay.historical_support()

    def test_exact_historical_sources_weights_and_manifest_are_portable(self):
        root = replay.historical_support()
        external = root / 'external'
        reference = external / 'deep_clipping_phase1/move_search.baseline.py'
        candidate = external / 'deep_clipping_phase1/move_search.candidate.py'
        self.assertEqual(replay.digest(reference.read_bytes()), replay.REFERENCE_SEARCH)
        self.assertEqual(replay.digest(candidate.read_bytes()),
                         'f9b86822a12dffd3941ba2f66c6d963997aa1a522cf5328af4049e457b6f3b07')
        frozen = json.loads((external / 'baseline_manifest.json').read_text())
        self.assertEqual(replay.digest((external / 'informe_partidas_v7_20261011.md').read_bytes()),
                         frozen['report_sha256'])
        self.assertEqual(len(replay.manifest_names(external / 'informe_partidas_v7_20261011.md')), 86)

    def test_only_certified_real_line_excerpts_are_bundled(self):
        root = replay.historical_support()
        provenance = json.loads((root / 'provenance.json').read_text())
        cases = json.loads((Path(__file__).parent / 'fixtures/deep_clipping_real_v7.json').read_text())['cases']
        for case in cases:
            with self.subTest(case=case['provenance']):
                source = case['provenance']
                excerpt = root / 'logs' / source['filename']
                lines = excerpt.read_text(encoding='utf-8').splitlines()
                self.assertEqual(replay.digest(lines[source['debug_line'] - 1].encode()), source['debug_sha256'])
                self.assertEqual(provenance['original_log_hashes'][source['filename']], source['log_sha256'])
                # An excerpt is explicitly different from the complete original log.
                self.assertNotEqual(replay.digest(excerpt.read_bytes()), source['log_sha256'])


if __name__ == '__main__':
    unittest.main()
