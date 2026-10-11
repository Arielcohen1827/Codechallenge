import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent


class TestRepositoryLayout(unittest.TestCase):
    def test_weight_loading_keeps_default_root_and_explicit_paths(self):
        weights = importlib.import_module('snake_bot.bot_weights')
        for explicit in (None, Path('weights') / 'candidate.json'):
            with self.subTest(explicit=explicit), \
                    patch.object(Path, 'exists', return_value=True), \
                    patch.object(weights, 'load_weights', return_value={}) as load, \
                    patch.object(weights, 'set_active_weights') as activate:
                loaded = (weights.load_active_weights() if explicit is None
                          else weights.load_active_weights(explicit))
                self.assertTrue(loaded)
                expected = ROOT / (explicit or 'weights/active_weights.json')
                self.assertEqual(load.call_args.args[0].resolve(), expected)
                activate.assert_called_once_with({})

    def test_packaged_state_preserves_move_rules(self):
        state_module = importlib.import_module('snake_bot.snake_state')
        for direction, head in [('right', (0, 1)), ('down', (1, 0))]:
            with self.subTest(direction=direction):
                state = state_module.parse_state({
                    'board': '|A  |\n|   |\n|  B|', 'side': 'A',
                    'rows': 3, 'cols': 3,
                    'remaining_moves': 20, 'score_1': 0, 'score_2': 0,
                })
                moved = state_module.apply_move(state, direction, 'A')
                self.assertEqual(moved.snakes['A'][0], head)

    def test_root_entrypoint_runs_without_token_from_other_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / 'run.py')],
                                    cwd=directory, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('please provide your auth_token', result.stdout)
        run = importlib.import_module('run')
        brain = importlib.import_module('snake_bot.snake_brain')
        self.assertIsInstance(run.BOT, brain.SnakeBrain)

    def test_tools_module_entrypoints(self):
        for name in ('bot_lab', 'benchmark_bot', 'analyze_matches',
                     'optimize_weights', 'match_viewer'):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-m', 'tools.' + name, '--help'],
                                        cwd=ROOT, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_viewer_resource_and_live_paths_remain_at_root(self):
        viewer = importlib.import_module('tools.match_viewer')
        self.assertEqual(viewer.LIVE_DIR, ROOT / 'games' / 'live')
        self.assertEqual(viewer.VIEWER_DIR, ROOT / 'viewer')
        for name in ('index.html', 'app.js', 'styles.css'):
            self.assertTrue((viewer.VIEWER_DIR / name).read_bytes())

    def test_viewer_serves_health_and_static_resources(self):
        import json
        import threading
        import urllib.request
        viewer = importlib.import_module('tools.match_viewer')
        with viewer.ThreadingHTTPServer(('127.0.0.1', 0), viewer.ViewerHandler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = 'http://127.0.0.1:' + str(server.server_port)
                with urllib.request.urlopen(base + '/api/health', timeout=5) as response:
                    self.assertEqual(json.load(response), {'ok': True})
                for name in ('index.html', 'app.js', 'styles.css'):
                    with self.subTest(name=name), urllib.request.urlopen(
                            base + '/' + name, timeout=5) as response:
                        self.assertEqual(response.read(), (ROOT / 'viewer' / name).read_bytes())
            finally:
                server.shutdown()
                worker.join(timeout=5)

    def test_launcher_starts_viewer_module_from_root(self):
        import launch_bot
        with patch.object(launch_bot, 'viewer_healthy', side_effect=[False, True]), \
                patch.object(launch_bot.subprocess, 'Popen') as process, \
                patch.object(launch_bot.webbrowser, 'open'), \
                patch.object(launch_bot.time, 'sleep'):
            launch_bot.start_viewer()
        self.assertEqual(process.call_args.args[0],
                         [sys.executable, '-m', 'tools.match_viewer'])
        self.assertEqual(process.call_args.kwargs['cwd'], ROOT)

    def test_benchmark_manifest_hashes_moved_sources(self):
        benchmark = importlib.import_module('tools.benchmark_bot')
        from types import SimpleNamespace
        result = SimpleNamespace(winner=None, score_a=0, score_b=0, turns=0,
                                 no_legal_side=None)
        for attribute in benchmark.EVENTS.values():
            for side in 'ab':
                setattr(result, attribute + '_' + side, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            reference.write_text('{}')
            with patch.object(benchmark, 'simulate_game', return_value=result):
                report = benchmark.run_benchmark(reference, reference, 1, 1, 1,
                                                 root / 'report.json')
        hashes = report['manifest']['source_sha256']
        for name in ('snake_bot/snake_state.py', 'tools/bot_lab.py',
                     'tools/benchmark_bot.py'):
            self.assertEqual(hashes[name], benchmark.hashlib.sha256(
                (ROOT / name).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
