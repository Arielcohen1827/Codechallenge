import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import gzip
import shutil

from tools import deep_clipping_replay as replay

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = replay.historical_support()
EXTERNAL = SUPPORT / 'external'
LOG_EXCERPTS = SUPPORT / 'logs'


class CorpusTests(unittest.TestCase):
    def test_manifest_is_exact_frozen_86(self):
        names = replay.manifest_names(EXTERNAL / 'informe_partidas_v7_20261011.md')
        frozen = json.loads((EXTERNAL / 'baseline_manifest.json').read_text())
        self.assertEqual(names, sorted(frozen['filenames']))
        self.assertEqual(len(names), 86)

    def test_real_known_loss_is_not_terminal(self):
        cases = replay.read_log(LOG_EXCERPTS / 'game_8d25279e-c5d3-11f1-96f5-a2aacf8fcf85.log')
        case = next(c for c in cases if c['provenance']['debug_line'] == 498)
        analysis = replay.analyze_case(case)
        self.assertEqual(analysis['terminal_class'], 'normal')
        self.assertEqual(analysis['raw_gap'], 106688)
        raw = {r['direction']: r['search_score'] for r in case['diagnostic']['deep_moves']}
        self.assertEqual(raw, {'up': -431010, 'down': -322732, 'left': -429420})
        self.assertTrue(analysis['all_clip_identical'])
        self.assertEqual(set(analysis['old_adjustments'].values()), {0})


class LegacyTests(unittest.TestCase):
    def test_historical_snapshot_retains_original_hash_and_rejects_wrong_reference(self):
        raw = replay.frozen_source('snake_bot/food_planner.py', replay.REFERENCE_X)
        self.assertEqual(replay.digest(raw), replay.REFERENCE_X)
        with self.assertRaisesRegex(ValueError, 'historical source hash mismatch'):
            replay.frozen_source('snake_bot/food_planner.py', '0' * 64)

    def test_legacy_clips_before_integer_mean(self):
        rows = [{'direction': 'up', 'search_score': 240582},
                {'direction': 'down', 'search_score': 240512}]
        self.assertEqual(replay.legacy_adjustments(rows, 4), {'up': 0, 'down': 0})

    def test_forced_loss_and_depth_gate(self):
        rows = [{'direction': 'up', 'search_score': 500000100, 'forced_loss': True},
                {'direction': 'down', 'search_score': -5000}]
        self.assertEqual(replay.legacy_adjustments(rows, 4), {'up': 0, 'down': 0})
        self.assertEqual(replay.legacy_adjustments(rows, 3), {'up': 0, 'down': 0})

    def test_terminal_provenance_is_conservative(self):
        self.assertEqual(replay.terminal_class([500019051, 540300], 175, 5),
                         'terminal-compatible/extreme')
        self.assertEqual(replay.terminal_class([0, 0], 3, 6), 'terminal-horizon')
        self.assertEqual(replay.terminal_class([-300046212], 3, 6), 'terminal-horizon')
        self.assertEqual(replay.terminal_class([299999999], 100, 4), 'normal')


class HypothesisTests(unittest.TestCase):
    def test_approved_normal_formula_centers_raw(self):
        rows = [
            {'direction': 'up', 'search_score': 240582},
            {'direction': 'down', 'search_score': 240512},
        ]
        self.assertEqual(
            replay.hypothetical_adjustments(rows, 4, 67),
            {'up': 12, 'down': -13},
        )
        rows[1]['search_score'] = -240512
        self.assertEqual(
            replay.hypothetical_adjustments(rows, 4, 67),
            {'up': 25000, 'down': -25000},
        )

    def test_whole_group_guard_excludes_forced_loss(self):
        rows = [
            {
                'direction': 'up',
                'search_score': 500019051,
                'forced_loss': True,
            },
            {'direction': 'down', 'search_score': 240582},
            {'direction': 'left', 'search_score': 240512},
        ]
        self.assertEqual(
            replay.hypothetical_adjustments(rows, 4, 67),
            {'up': 0, 'down': 12, 'left': -13},
        )
        rows[0]['forced_loss'] = False
        self.assertEqual(
            replay.hypothetical_adjustments(rows, 4, 67),
            replay.legacy_adjustments(rows, 4),
        )
        self.assertEqual(
            replay.hypothetical_adjustments(rows[1:], 4, 4),
            replay.legacy_adjustments(rows[1:], 4),
        )
        self.assertEqual(
            replay.hypothetical_adjustments(
                rows[1:], 4, 67, root_terminal=True
            ),
            replay.legacy_adjustments(rows[1:], 4),
        )

    def test_cap_metrics_have_specific_counterfactual_controls(self):
        source = replay.load_baseline(
            EXTERNAL / 'deep_clipping_phase1/move_search.baseline.py'
        )
        rows = []
        for direction, raw, base, old in [
            ('down', 900000, 20, 25000),
            ('up', 1000000, 10, 25000),
            ('left', -10000000, 0, -25000),
        ]:
            rows.append(
                {
                    'direction': direction,
                    'search_score': raw,
                    'search_depth': 4,
                    'score': base + old,
                    'search_adjustment': old,
                    'food_now': False,
                    'our_replies': 3,
                    'exits': 3,
                    'region': 100,
                    'voronoi_score': 0,
                    'safety': 'SAFE',
                }
            )
        case = {
            'provenance': {'game_id': 'unit-control', 'debug_line': 1},
            'diagnostic': {
                'deep_moves': rows,
                'scores': {'A': 0, 'B': 0},
                'remaining_moves': 100,
                'legal': ['up', 'down', 'left'],
            },
        }
        summary = replay.measure_hypothesis([case], source)
        self.assertEqual(summary['counts']['cap_a_top1_top2_gap_lost'], 1)
        self.assertEqual(
            summary['counts'][
                'cap_b_uncapped_order_changes_but_capped_stays_old'
            ],
            1,
        )
        self.assertEqual(
            summary['counts'][
                'cap_c_uncapped_mapping_changes_but_capped_stays_old'
            ],
            1,
        )


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(
            (ROOT / 'tests/fixtures/deep_clipping_real_v7.json').read_text()
        )['cases']

    def test_real_provenance_raw_and_context_all_fixtures(self):
        self.assertGreaterEqual(len(self.fixtures), 20)
        for case in self.fixtures:
            with self.subTest(provenance=case['provenance']):
                line = (
                    (LOG_EXCERPTS / case['provenance']['filename'])
                    .read_text(encoding='utf-8-sig')
                    .splitlines()[case['provenance']['debug_line'] - 1]
                )
                recorded = json.loads(line[2:])
                self.assertEqual(
                    recorded['deep_moves'], case['diagnostic']['deep_moves']
                )
                self.assertEqual(
                    replay.digest(line.encode()),
                    case['provenance']['debug_sha256'],
                )
                state, verification = replay.state_for(case)
                self.assertTrue(verification['observable_state_exact'])
                self.assertEqual(
                    state.remaining_moves, recorded['remaining_moves']
                )

    def test_frozen_rerank_legacy_control_preserves_order(self):
        source = replay.load_baseline(EXTERNAL / 'deep_clipping_phase1/move_search.baseline.py')
        for case in self.fixtures:
            rows = case['diagnostic'].get('deep_moves', [])
            if not rows:
                continue
            with self.subTest(provenance=case['provenance']):
                ranked = replay.rerank_recorded(case, source, candidate=False)
                self.assertEqual(ranked, rows)

    def test_fixture_categories_really_observed(self):
        classes = {f['analysis']['terminal_class'] for f in self.fixtures}
        self.assertEqual(classes, {'normal', 'terminal-horizon', 'terminal-compatible/extreme'})
        self.assertTrue(any(f['analysis']['forced_loss_present'] for f in self.fixtures))
        self.assertTrue(any(f['analysis']['SAFE2'] for f in self.fixtures))

    def test_compact_history_and_full_brain_control(self):
        case = next(f for f in self.fixtures if f['provenance']['debug_line'] == 498)
        self.assertIn('signatures', case['history'])
        self.assertLessEqual(len(case['history']['signatures']), 80)
        source = replay.load_baseline(EXTERNAL / 'deep_clipping_phase1/move_search.baseline.py')
        result = replay.replay_current_reference(case, source)
        self.assertEqual(result['status'], 'controlled_historical_raw_replay')
        self.assertTrue(result['raw_result_preserved'])
        self.assertTrue(result['repeat_count_exact'])
        self.assertTrue(result['hunger_exact'])
        self.assertEqual(result['control_final_direction'], result['current_x_final_direction'])
        self.assertTrue(result['safe_direction_applied'])

    def test_actual_candidate_brain_uses_exact_shared_result(self):
        case = next(
            f for f in self.fixtures if f['provenance']['debug_line'] == 498
        )
        source = replay.load_baseline(
            EXTERNAL / 'deep_clipping_phase1/move_search.baseline.py'
        )
        candidate = replay.load_source(
            EXTERNAL / 'deep_clipping_phase1/move_search.candidate.py',
            'f9b86822a12dffd3941ba2f66c6d963997aa1a522cf5328af4049e457b6f3b07',
        )
        result = replay.replay_comparison(case, source, candidate)
        self.assertTrue(result['raw_result_preserved'])
        self.assertTrue(result['shared_request_and_result_exact'])
        self.assertTrue(result['shallow_bases_and_info_exact'])
        self.assertEqual(result['current_x_final_direction'], 'up')
        self.assertEqual(result['candidate_final_direction'], 'down')
        self.assertTrue(result['candidate_final_changed'])

    def test_changed_x_target_is_excluded_not_retimed(self):
        case = next(
            f
            for f in self.fixtures
            if f['provenance']['game_id'].startswith('5580bbd6')
            and f['provenance']['debug_line'] == 576
        )
        source = replay.load_baseline(
            EXTERNAL / 'deep_clipping_phase1/move_search.baseline.py'
        )
        candidate = replay.load_source(
            EXTERNAL / 'deep_clipping_phase1/move_search.candidate.py',
            'f9b86822a12dffd3941ba2f66c6d963997aa1a522cf5328af4049e457b6f3b07',
        )
        result = replay.replay_comparison(case, source, candidate)
        self.assertEqual(result['status'], 'unsupported_changed_target')
        self.assertNotIn('candidate_final_direction', result)


class FrozenEngineTests(unittest.TestCase):
    setUpClass = FixtureTests.__dict__['setUpClass']

    def test_exact_result_and_request_reuse_rejects_target_change(self):
        case = next(
            f for f in self.fixtures if f['analysis']['eligible_differing_raw']
        )
        state, _ = replay.state_for(case)
        tape = replay.FrozenEngine(case)
        engine, compact = tape.from_game_state(state, 10, 6)
        target = tuple(case['diagnostic']['plan']['food'])
        first = engine.search(
            compact, state.side, case['diagnostic']['legal'], target=target
        )
        second = engine.search(
            compact, state.side, case['diagnostic']['legal'], target=target
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first.scores,
            {
                r['direction']: r['search_score']
                for r in case['diagnostic']['deep_moves']
                if r['search_score'] is not None
            },
        )
        with self.assertRaisesRegex(ValueError, 'request mismatch'):
            engine.search(
                compact, state.side, case['diagnostic']['legal'], target=(0, 0)
            )
        with self.assertRaisesRegex(ValueError, 'state'):
            tape.from_game_state(
                replay.replace(
                    state, remaining_moves=state.remaining_moves - 1
                ),
                10,
                6,
            )


class ComparisonCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        # Exercise the CLI's real input reads using an isolated, explicit audit
        # environment, never ignored files left beside a developer's checkout.
        for name, source in (
                ('snake_bot/search_engine.py', ROOT / 'snake_bot/search_engine.py'),
                ('weights/active_weights.json', EXTERNAL / 'baseline_active_weights.json'),
                ('games/game_8d25279e-c5d3-11f1-96f5-a2aacf8fcf85.log',
                 LOG_EXCERPTS / 'game_8d25279e-c5d3-11f1-96f5-a2aacf8fcf85.log')):
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        root_patch = patch.object(replay, 'ROOT', root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_real_two_case_cli_reports_final_moves_and_exclusions(self):
        fixtures = json.loads(
            (ROOT / 'tests/fixtures/deep_clipping_real_v7.json').read_text()
        )['cases']
        chosen = [
            c
            for c in fixtures
            if c['provenance']['debug_line'] == 498
            or c['provenance']['game_id'].startswith('5580bbd6')
            and c['provenance']['debug_line'] == 576
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(replay, 'load_cohort', return_value=chosen),
        ):
            replay.main(
                [
                    'compare',
                    '--report',
                    str(EXTERNAL / 'informe_partidas_v7_20261011.md'),
                    '--frozen-manifest',
                    str(EXTERNAL / 'baseline_manifest.json'),
                    '--fixtures',
                    str(ROOT / 'tests/fixtures/deep_clipping_real_v7.json'),
                    '--baseline-search',
                    str(
                        EXTERNAL
                        / 'deep_clipping_phase1/move_search.baseline.py'
                    ),
                    '--candidate-search',
                    str(
                        EXTERNAL
                        / 'deep_clipping_phase1/move_search.candidate.py'
                    ),
                    '--candidate-sha256',
                    'f9b86822a12dffd3941ba2f66c6d963997aa1a522cf5328af4049e457b6f3b07',
                    '--output',
                    directory,
                ]
            )
            result = json.loads(
                (Path(directory) / 'comparison.json').read_text()
            )
            self.assertEqual(result['counts']['eligible_historical_pool'], 2)
            self.assertEqual(result['counts']['supported_common_input'], 1)
            self.assertEqual(result['counts']['unsupported_changed_target'], 1)
            self.assertEqual(result['counts']['candidate_final_changed'], 1)
            self.assertEqual(
                result['games_with_at_least_one']['candidate_final_changed'], 1
            )
            self.assertEqual(result['timed_searches'], 0)

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            replay.write_new(path, {'first': 1})
            with self.assertRaises(FileExistsError):
                replay.write_new(path, {'second': 2})
            self.assertEqual(json.loads(path.read_text()), {'first': 1})

    def test_source_hash_and_frozen_manifest_mismatch_fail_before_replay(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            replay.load_source(ROOT / 'snake_bot/move_search.py', 'bad-hash')
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            replay.write_new(
                manifest, {'filenames': [], 'report_sha256': 'bad-hash'}
            )
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                replay.load_cohort(
                    EXTERNAL / 'informe_partidas_v7_20261011.md',
                    manifest,
                    ROOT / 'games',
                )

    def test_analysis_details_and_hypothesis_cli_real_subset(self):
        fixtures = json.loads(
            (ROOT / 'tests/fixtures/deep_clipping_real_v7.json').read_text()
        )['cases']
        for command, filename in [
            ('analyze', 'corpus.json'),
            ('details', 'corpus_details.json'),
            ('hypothesis', 'hypothesis.json'),
        ]:
            with (
                self.subTest(command=command),
                tempfile.TemporaryDirectory() as directory,
            ):
                with (
                    patch.object(replay, 'load_cohort', return_value=fixtures),
                    patch('builtins.print'),
                ):
                    replay.main(
                        [
                            command,
                            '--report',
                            str(EXTERNAL / 'informe_partidas_v7_20261011.md'),
                            '--frozen-manifest',
                            str(EXTERNAL / 'baseline_manifest.json'),
                            '--fixtures',
                            str(Path(directory) / 'fixtures.json'),
                            '--baseline-search',
                            str(
                                EXTERNAL
                                / 'deep_clipping_phase1/move_search.baseline.py'
                            ),
                            '--output',
                            directory,
                        ]
                    )
                result = json.loads((Path(directory) / filename).read_text())
                if command == 'hypothesis':
                    self.assertEqual(
                        result['counts']['legacy_order_control_mismatch'], 0
                    )
                    self.assertEqual(
                        result['counts']['guarded_mapping_changed'], 0
                    )
                    self.assertGreater(
                        result['counts']['adjustment_changed'], 0
                    )
                else:
                    self.assertEqual(result['counts']['decisions'], 31)
                    self.assertEqual(
                        result['counts']['actual_adjustment_mismatch'], 0
                    )
                    self.assertEqual(
                        result['raw_top1_top2_gap'][
                            'terminal-compatible/extreme'
                        ]['n'],
                        2,
                    )
                self.assertTrue(
                    (
                        Path(directory) / filename.replace('.json', '.md')
                    ).exists()
                )

    def test_replay_cli_real_case_runs_safe_direction(self):
        fixtures = json.loads(
            (ROOT / 'tests/fixtures/deep_clipping_real_v7.json').read_text()
        )
        fixtures['cases'] = [
            c
            for c in fixtures['cases']
            if c['provenance']['debug_line'] == 498
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('builtins.print'),
        ):
            fixture_path = Path(directory) / 'fixtures.json'
            replay.write_new(fixture_path, fixtures)
            replay.main(
                [
                    'replay',
                    '--fixtures',
                    str(fixture_path),
                    '--baseline-search',
                    str(
                        EXTERNAL
                        / 'deep_clipping_phase1/move_search.baseline.py'
                    ),
                    '--output',
                    directory,
                ]
            )
            result = json.loads(
                (Path(directory) / 'reference_replay.json').read_text()
            )
            self.assertEqual(
                result['counts']['controlled_historical_raw_replay'], 1
            )
            self.assertEqual(result['counts']['raw_result_preserved'], 1)
            self.assertTrue(result['results'][0]['safe_direction_applied'])

    def test_compact_cli_preserves_real_external_history_and_root(self):
        fixtures = json.loads(
            (ROOT / 'tests/fixtures/deep_clipping_real_v7.json').read_text()
        )
        case = next(
            c
            for c in fixtures['cases']
            if c['provenance']['debug_line'] == 498
        )
        with gzip.open(
            EXTERNAL / 'deep_clipping_phase1/fixture_histories.json.gz', 'rt'
        ) as stream:
            history = json.load(stream)[
                case['history']['external_history_key']
            ]
        case['history'] = history
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('builtins.print'),
        ):
            fixture_path = Path(directory) / 'fixtures.json'
            replay.write_new(fixture_path, {'cases': [case]})
            replay.main(
                [
                    'compact',
                    '--fixtures',
                    str(fixture_path),
                    '--output',
                    directory,
                ]
            )
            compact = json.loads(fixture_path.read_text())['cases'][0]
            self.assertEqual(compact['diagnostic'], case['diagnostic'])
            self.assertEqual(len(compact['history']['signatures']), 80)
            with gzip.open(
                Path(directory) / 'fixture_histories.json.gz', 'rt'
            ) as stream:
                saved = json.load(stream)
            self.assertEqual(
                saved[compact['history']['external_history_key']], history
            )


if __name__ == '__main__':
    unittest.main()
