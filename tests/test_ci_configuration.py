import unittest
import runpy
import tempfile
from pathlib import Path

from coverage import Coverage


ROOT = Path(__file__).resolve().parent.parent


class TestCiCoverageGate(unittest.TestCase):
    def test_github_actions_runs_coverage_with_required_threshold(self):
        workflow = (ROOT / '.github' / 'workflows' / 'tests.yml').read_text(encoding='utf-8')

        self.assertIn('python -m coverage run -m unittest discover -v', workflow)
        self.assertIn('python -m coverage report --fail-under=90', workflow)

    def test_local_coverage_script_uses_required_threshold(self):
        script = (ROOT / 'coverage.bat').read_text(encoding='utf-8')

        self.assertIn('-m coverage run -m unittest discover -v', script)
        self.assertIn('-m coverage report --fail-under=90', script)

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
