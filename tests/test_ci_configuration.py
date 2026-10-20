import unittest
import ast
import re
import runpy
import tempfile
from pathlib import Path

from coverage import Coverage


ROOT = Path(__file__).resolve().parent.parent


class TestCiPythonCompatibility(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / '.github' / 'workflows' / 'tests.yml').read_text(encoding='utf-8')

    def test_matrix_covers_minimum_supported_python_and_current_target(self):
        matrix = re.search(r'^\s+python-version:\s*(\[.*\])\s*$',
                           self.workflow, re.MULTILINE)
        self.assertIsNotNone(matrix, 'The CI matrix must declare Python versions')
        versions = ast.literal_eval(matrix.group(1))
        self.assertIn('3.10', versions, 'Exercise the minimum version supporting dataclass slots')
        self.assertIn('3.12', versions, 'Keep the existing Python 3.12 target')
        for version in versions:
            with self.subTest(version=version):
                self.assertGreaterEqual(tuple(map(int, version.split('.'))), (3, 10))
        self.assertIn('python-version: ${{ matrix.python-version }}', self.workflow)

    def test_each_ci_target_runs_suite_and_enforces_coverage_threshold(self):
        self.assertIn('python -m coverage run -m unittest discover -v', self.workflow)
        self.assertIn('python -m coverage report --fail-under=90', self.workflow)


class TestCiCoverageGate(unittest.TestCase):
    def test_shared_configuration_measures_all_own_source_and_enforces_gate(self):
        cov = Coverage(config_file=str(ROOT / '.coveragerc'))
        self.assertEqual(cov.get_option('run:source'), ['.'])
        self.assertEqual(cov.get_option('report:fail_under'), 90)
        self.assertEqual(cov.get_option('report:precision'), 2)

    def test_measurement_excludes_tests_and_environments_but_keeps_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = ['run.py', 'launch_bot.py', 'snake_bot/state.py', 'tools/lab.py']
            excluded = ['tests/test_behavior.py', 'tests/helpers.py',
                        'test_root.py', 'venv/library.py', '.venv/library.py']
            for relative in included + excluded:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('value = 42\n', encoding='utf-8')
            cov = Coverage(config_file=str(ROOT / '.coveragerc'),
                           source=[str(root)], data_file=None)
            cov.start()
            try:
                for relative in included + excluded:
                    runpy.run_path(str(root / relative))
            finally:
                cov.stop()
            measured = {Path(path).relative_to(root).as_posix()
                        for path in cov.get_data().measured_files()}
            self.assertEqual(measured, set(included))


if __name__ == '__main__':
    unittest.main()
