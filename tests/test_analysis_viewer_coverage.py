import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import analyze_matches as analyzer
from tools import match_viewer as viewer


def turn(board='|   B|\n| A* |\n|    |', **changes):
    data = {'board': board, 'side': 'A', 'remaining_moves': 20,
            'score_1': 0, 'score_2': 0}
    data.update(changes)
    return data


class AnalyzerCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def log(self, turns=(), actions=(), final=None, debug=(), metadata=None):
        records = []
        if metadata is not None:
            records.append(('=', metadata))
        records.extend(('<', {'event': 'your_turn', 'data': data}) for data in turns)
        records.extend(('>', {'action': 'move', 'data': {'direction': direction}})
                       for direction in actions)
        records.extend(('?', record) for record in debug)
        if final is not None:
            records.append(('<', {'event': 'game_over', 'data': final}))
        path = self.root / 'match.log'
        path.write_text('\n'.join(prefix + ' ' + json.dumps(data)
                                  for prefix, data in records), encoding='utf-8')
        return path

    def test_parse_log_filters_events_and_preserves_protocol_payloads(self):
        data = turn(turn_token='one')
        metadata = {'event': 'bot_version', 'version': 'v-test', 'notes': 'fixture'}
        debug = {'event': 'decision_debug', 'turn_token': 'one', 'plan': None}
        final = {'score_1': 101, 'score_2': 2}
        path = self.log([data], ['right'], final, [debug], metadata)
        with path.open('a', encoding='utf-8') as handle:
            handle.write('\nignored diagnostic\n= {"event":"other"}\n'
                         '< {"event":"joined"}\n> {"action":"ping"}\n'
                         '? {"event":"other"}\n')
        self.assertEqual(analyzer.parse_log(path),
                         ([data], [{'direction': 'right'}], final, metadata, [debug]))

    def test_parse_log_defaults_missing_action_and_final_data(self):
        path = self.root / 'minimal.log'
        path.write_text('> {"action":"move"}\n< {"event":"game_over"}\n')
        self.assertEqual(analyzer.parse_log(path), ([], [{}], {}, {}, []))

    def test_parse_log_rejects_malformed_protocol_json(self):
        path = self.root / 'invalid.log'
        path.write_text('< {broken\n')
        with self.assertRaises(json.JSONDecodeError):
            analyzer.parse_log(path)

    def test_empty_match_has_zero_rates_and_unknown_version(self):
        metrics = analyzer.analyze(self.log())
        self.assertEqual(metrics['total_turns'], 0)
        self.assertEqual(metrics['first_food_move'], -1)
        self.assertEqual(metrics['food_per_100_moves'], 0)
        self.assertEqual(metrics['average_food_interval'], 0)
        self.assertEqual(metrics['bot_version'], 'unknown')
        self.assertEqual(metrics['final'], {})

    def test_food_scores_intervals_cycles_and_final_metadata(self):
        turns = [turn(score_1=0), turn(score_1=100, score_2=100),
                 turn(score_1=101, score_2=100), turn(score_1=201, score_2=100)]
        metrics = analyzer.analyze(self.log(
            turns, ['right', 'down', 'right', 'down'],
            {'score_1': '202', 'score_2': '100'},
            metadata={'event': 'bot_version', 'version': 'test', 'notes': 'local'}))
        expected = {'foods_us': 2, 'foods_A': 2, 'foods_B': 1, 'food_difference': 1,
                    'first_food_move': 0, 'average_food_interval': 2.0,
                    'food_per_100_moves': 50.0, 'repeated_states': 3,
                    'longest_cycle': 4, 'adjacent_food_taken': 2,
                    'adjacent_food_rejected': 2, 'final_score_A': 202,
                    'final_score_B': 100, 'bot_version': 'test', 'bot_notes': 'local'}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertEqual(metrics[key], value)

    def test_missing_and_invalid_actions_are_skipped_but_turns_counted(self):
        metrics = analyzer.analyze(self.log([turn(), turn(), turn()], ['teleport', 'down']))
        self.assertEqual(metrics['total_turns'], 3)
        self.assertEqual(metrics['adjacent_food_seen'], 1)
        self.assertEqual(metrics['max_turns_without_food'], 2)
        self.assertEqual(metrics['first_food_move'], -1)

    def test_collision_diagnostics_use_real_board_and_legal_moves(self):
        cases = [('|A  |\n|  B|', 'up', 'out_of_bounds', 0),
                 ('|    |\n| Aa |\n|  B |', 'right', 'own_body_collision', 0),
                 ('|    |\n| AB |\n|    |', 'right', 'rival_collision', 0),
                 ('|AB|', 'right', 'rival_collision', 1),
                 ('| B |', 'left', 'out_of_bounds', 1)]
        for board, direction, metric, trapped in cases:
            with self.subTest(board=board):
                metrics = analyzer.analyze(self.log([turn(board)], [direction]))
                self.assertEqual(metrics[metric], 1)
                self.assertEqual(metrics['no_legal_moves'], trapped)
                self.assertEqual(metrics['self_collisions'], metrics['own_body_collision'])
                if not trapped:
                    self.assertEqual(metrics['illegal_moves_repaired'], 1)

    def test_pickup_wrong_digit_and_wall_moves_are_distinguished(self):
        for board, metric in [('|AX B|', 'multiplier_pickups_us'),
                              ('|A21B|', 'wrong_digits_us'),
                              ('|A# B|', 'wall_hits_us')]:
            with self.subTest(metric=metric):
                metrics = analyzer.analyze(self.log([turn(board)], ['right']))
                self.assertEqual(metrics[metric], 1)
                self.assertEqual(metrics['foods_us'], 0)

    def test_illegal_move_still_reported_when_fallback_keeps_direction(self):
        with patch.object(analyzer, 'final_safe_direction', return_value='up'):
            metrics = analyzer.analyze(self.log([turn('|A B|')], ['up']))
        self.assertEqual(metrics['out_of_bounds'], 1)
        self.assertEqual(metrics['illegal_moves_repaired'], 0)

    def test_adjacent_rejection_classifies_each_debug_pressure(self):
        cases = [({'forced_zero': True}, 'hard'),
                 ({'forced_delayed_zero': True}, 'hard'),
                 ({'zero_escape_replies': 1}, 'hard'),
                 ({'one_escape_replies': 2}, 'soft'),
                 ({'delayed_forced_zero_replies': 1}, 'soft'),
                 ({'forced_zero': True, 'one_escape_replies': 2}, 'hard'),
                 ({}, None)]
        for stats, category in cases:
            with self.subTest(stats=stats):
                debug = {'event': 'decision_debug', 'turn_token': 't',
                         'plan': {'our_distance': 1, 'first_move': 'right'},
                         'plan_move_stats': stats}
                plan = SimpleNamespace(food=(1, 2), safety='SAFE')
                with patch.object(analyzer, 'choose_food_plan', return_value=plan):
                    metrics = analyzer.analyze(self.log(
                        [turn(turn_token='t')], ['down'], debug=[debug]))
                self.assertEqual(metrics['safe_adjacent_food_rejected'], 1)
                self.assertEqual(metrics['self_traps'], 1)
                self.assertEqual(metrics['hard_adjacent_food_rejected'], int(category == 'hard'))
                self.assertEqual(metrics['soft_adjacent_food_rejected'], int(category == 'soft'))
                self.assertEqual(metrics['decision_debug_lines'], 1)

    def test_unsafe_or_absent_candidate_does_not_count_as_safe_rejection(self):
        for plan in (None, SimpleNamespace(food=(1, 2), safety='UNSAFE')):
            with self.subTest(plan=plan), patch.object(
                    analyzer, 'choose_food_plan', return_value=plan):
                metrics = analyzer.analyze(self.log([turn()], ['down'], debug=[
                    {'event': 'decision_debug', 'turn_token': None},
                    {'event': 'decision_debug', 'turn_token': 'unmatched'}]))
                self.assertEqual(metrics['adjacent_food_rejected'], 1)
                self.assertEqual(metrics['safe_adjacent_food_rejected'], 0)
                self.assertEqual(metrics['hard_adjacent_food_rejected'], 0)

    def test_incomplete_or_nonadjacent_debug_does_not_claim_pressure_rejection(self):
        for debug_plan in (None, {'our_distance': 2, 'first_move': 'right'},
                           {'our_distance': 1, 'first_move': 'down'}):
            with self.subTest(plan=debug_plan):
                metrics = analyzer.analyze(self.log(
                    [turn(turn_token='t')], ['down'], debug=[
                        {'event': 'decision_debug', 'turn_token': 't', 'plan': debug_plan,
                         'plan_move_stats': {'forced_zero': True}}]))
                self.assertEqual(metrics['adjacent_food_rejected'], 1)
                self.assertEqual(metrics['hard_adjacent_food_rejected'], 0)
                self.assertEqual(metrics['soft_adjacent_food_rejected'], 0)

    def test_partial_final_score_preserves_missing_score_distinction(self):
        metrics = analyzer.analyze(self.log(final={'score_2': '300'}))
        self.assertNotIn('final_score_A', metrics)
        self.assertEqual(metrics['final_score_B'], 300)
        self.assertEqual(metrics['final'], {'score_2': '300'})

    def test_target_change_counts_only_still_available_objectives(self):
        first = SimpleNamespace(food=(0, 1), safety='SAFE')
        second = SimpleNamespace(food=(0, 3), safety='ACCEPTABLE_RISK')
        for next_board, next_plan, changes, abandoned in [
                ('| * *B|\n|A    |\n|     |', second, 1, 1),
                ('| * *B|\n|A    |\n|     |', None, 1, 0),
                ('| * *B|\n|A    |\n|     |',
                 SimpleNamespace(food=(0, 3), safety='UNSAFE'), 1, 0),
                ('|   *B|\n|A    |\n|     |', second, 0, 0),
                ('| * *B|\n|A    |\n|     |', first, 0, 0)]:
            with self.subTest(board=next_board, plan=next_plan), patch.object(
                    analyzer, 'choose_food_plan', side_effect=[first, next_plan]):
                metrics = analyzer.analyze(self.log(
                    [turn('| * *B|\n|A    |\n|     |'), turn(next_board)], ['down', 'down']))
                self.assertEqual(metrics['target_changes'], changes)
                self.assertEqual(metrics['target_abandoned'], abandoned)

    def test_signature_ignores_scores_but_tracks_objective_and_multiplier_changes(self):
        base = analyzer.parse_state(turn('|A1XB|\n| #  |'))
        same = analyzer.parse_state(turn('|A1XB|\n| #  |', score_1=400))
        changed = analyzer.parse_state(turn('|A1XB|\n| #  |', multiplier_1=2))
        self.assertEqual(analyzer.state_signature(base), analyzer.state_signature(same))
        self.assertNotEqual(analyzer.state_signature(base), analyzer.state_signature(changed))

    def test_score_food_delta_threshold_and_missing_side(self):
        for before, after, expected in [(0, 99, False), (10, 110, True),
                                        (100, 0, False)]:
            with self.subTest(before=before, after=after):
                self.assertEqual(analyzer.score_food_delta(
                    SimpleNamespace(scores={'A': before}),
                    SimpleNamespace(scores={'A': after}), 'A'), expected)
        self.assertFalse(analyzer.score_food_delta(
            SimpleNamespace(scores={}), SimpleNamespace(scores={}), 'B'))

    def test_expand_paths_supports_sorted_globs_games_fallback_and_missing_files(self):
        (self.root / 'games').mkdir()
        for name in ('z.log', 'a.log', 'games/fallback.log'):
            (self.root / name).write_text('')
        original = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(analyzer.expand_log_paths(['*.log']),
                             [Path('a.log'), Path('z.log')])
            self.assertEqual(analyzer.expand_log_paths(['fall*.log', 'fallback.log',
                                                        'a.log', 'missing.log']),
                             [Path('games/fallback.log'), Path('games/fallback.log'),
                              Path('a.log'), Path('missing.log')])
            self.assertEqual(analyzer.expand_log_paths(['none*.log']), [Path('none*.log')])
        finally:
            os.chdir(original)

    def test_main_prints_real_metrics_and_optional_details(self):
        path = self.log([turn()], ['right'], {'score_1': 100},
                        metadata={'event': 'bot_version', 'version': 'cli', 'notes': 'fixture'})
        output = io.StringIO()
        with patch('sys.argv', ['analyze_matches', str(path)]), contextlib.redirect_stdout(output):
            analyzer.main()
        text = output.getvalue()
        self.assertIn(str(path), text)
        self.assertIn('  foods_us: 1\n', text)
        self.assertIn('  bot_version: cli\n', text)
        self.assertIn('  bot_notes: fixture\n', text)
        self.assertIn("  final: {'score_1': 100}\n", text)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analyzer.print_report(path, analyzer.analyze(self.log()))
        self.assertNotIn('bot_notes:', output.getvalue())
        self.assertNotIn('  final:', output.getvalue())


class ViewerSnapshotCoverageTests(unittest.TestCase):
    def test_read_snapshot_returns_unicode_payload_or_none_for_bad_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.json'
            payload = {'game_id': 'local', 'turn_data': {'player_1': 'Jos\u00e9'}}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            self.assertEqual(viewer.read_snapshot(path), payload)
            path.write_text('{invalid', encoding='utf-8')
            self.assertIsNone(viewer.read_snapshot(path))
            self.assertIsNone(viewer.read_snapshot(Path(directory) / 'absent.json'))
        with patch.object(Path, 'read_text', side_effect=PermissionError('denied')):
            self.assertIsNone(viewer.read_snapshot(path))

    def test_food_eaten_prefers_stored_zero_and_infers_legacy_growth(self):
        snapshot = {'food_eaten': {'A': 0}, 'turn_data': {'board': '|Aaaaa Bbbb|'}}
        self.assertEqual(viewer.food_eaten(snapshot, 'A'), 0)
        self.assertEqual(viewer.food_eaten(snapshot, 'B'), 1)
        self.assertEqual(viewer.food_eaten({'turn_data': {'board': 'Aaa'}}, 'A'), 0)
        self.assertEqual(viewer.food_eaten({'turn_data': None}, 'B'), 0)

    def test_game_summary_has_defaults_and_preserves_explicit_fields(self):
        self.assertEqual(viewer.game_summary({}), {
            'game_id': '', 'status': 'playing', 'started_at': 0, 'updated_at': 0,
            'turn_number': 0, 'bot_version': '', 'player_1': 'Jugador A',
            'player_2': 'Jugador B', 'score_1': 0, 'score_2': 0,
            'remaining_moves': 0, 'side': '', 'food_eaten_1': 0, 'food_eaten_2': 0})
        summary = viewer.game_summary({
            'game_id': 'g', 'status': 'finished', 'updated_at': 9, 'turn_number': 4,
            'bot_version': 'v', 'food_eaten': {'A': 2, 'B': 1},
            'turn_data': {'player_1': 'one', 'player_2': 'two', 'score_1': 201,
                          'score_2': 101, 'remaining_moves': 3, 'side': 'B'}})
        self.assertEqual(summary['started_at'], 9)
        self.assertEqual(summary['food_eaten_1'], 2)
        self.assertEqual(summary['food_eaten_2'], 1)
        self.assertEqual(summary['score_1'], 201)
        self.assertEqual(summary['player_2'], 'two')
        self.assertEqual(summary['status'], 'finished')
        self.assertEqual(summary['side'], 'B')

    def test_cli_binds_requested_address_and_closes_on_interrupt_or_error(self):
        for failure in (KeyboardInterrupt(), RuntimeError('server failure'), None):
            with self.subTest(failure=failure):
                server = Mock()
                server.serve_forever.side_effect = failure
                output = io.StringIO()
                with patch('sys.argv', ['match_viewer', '--host', 'localhost', '--port', '1234']), \
                        patch.object(viewer, 'ThreadingHTTPServer', return_value=server) as factory, \
                        contextlib.redirect_stdout(output):
                    if isinstance(failure, RuntimeError):
                        with self.assertRaisesRegex(RuntimeError, 'server failure'):
                            viewer.main()
                    else:
                        viewer.main()
                factory.assert_called_once_with(('localhost', 1234), viewer.ViewerHandler)
                server.serve_forever.assert_called_once_with()
                server.server_close.assert_called_once_with()
                self.assertIn('http://localhost:1234', output.getvalue())

    def test_cli_default_address(self):
        with patch('sys.argv', ['match_viewer']), \
                patch.object(viewer, 'ThreadingHTTPServer') as factory, \
                contextlib.redirect_stdout(io.StringIO()):
            viewer.main()
        factory.assert_called_once_with(('127.0.0.1', 8765), viewer.ViewerHandler)
        factory.return_value.server_close.assert_called_once_with()

    def test_access_log_suppresses_success_but_reports_errors(self):
        handler = viewer.ViewerHandler.__new__(viewer.ViewerHandler)
        with patch.object(viewer.SimpleHTTPRequestHandler, 'log_message') as logger:
            handler.log_message('%s %s', 'GET /api/games', '200')
            logger.assert_not_called()
            handler.log_message('%s %s', 'GET /api/game/missing', '404')
            logger.assert_called_once_with('%s %s', 'GET /api/game/missing', '404')


class ViewerEndpointCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = viewer.ThreadingHTTPServer(('127.0.0.1', 0), viewer.ViewerHandler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        live_patch = patch.object(viewer, 'LIVE_DIR', self.root / 'live')
        live_patch.start()
        self.addCleanup(live_patch.stop)
        logger_patch = patch.object(viewer.SimpleHTTPRequestHandler, 'log_message')
        logger_patch.start()
        self.addCleanup(logger_patch.stop)

    def snapshot(self, game_id, **changes):
        viewer.LIVE_DIR.mkdir(exist_ok=True)
        payload = {'game_id': game_id, 'status': 'playing', 'updated_at': 1}
        payload.update(changes)
        (viewer.LIVE_DIR / ('game_' + game_id + '.json')).write_text(
            json.dumps(payload), encoding='utf-8')
        return payload

    def request(self, path, method='GET', origin=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request(method, path, headers={'Origin': origin} if origin else {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_games_returns_empty_when_live_directory_does_not_exist(self):
        status, headers, body = self.request('/api/games?refresh=1')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'games': []})
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(int(headers['Content-Length']), len(body))
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')

    def test_games_sorts_active_by_start_finished_by_update_and_skips_bad_snapshots(self):
        self.snapshot('active-new', started_at=20, updated_at=2)
        self.snapshot('active-old', started_at=10, updated_at=30)
        self.snapshot('finished-new', status='finished', started_at=0, updated_at=50)
        self.snapshot('finished-old', status='finished', started_at=100, updated_at=40)
        (viewer.LIVE_DIR / 'game_bad.json').write_text('{bad')
        (viewer.LIVE_DIR / 'game_empty.json').write_text('{}')
        (viewer.LIVE_DIR / 'ignored.json').write_text('{"game_id":"ignored"}')
        status, _, body = self.request('/api/games')
        self.assertEqual(status, 200)
        games = json.loads(body)['games']
        self.assertEqual([game['game_id'] for game in games],
                         ['active-old', 'active-new', 'finished-old', 'finished-new'])
        self.assertEqual(games[0]['player_1'], 'Jugador A')

    def test_games_limits_results_to_first_hundred(self):
        for index in range(102):
            self.snapshot('g' + str(index), started_at=index)
        status, _, body = self.request('/api/games')
        games = json.loads(body)['games']
        self.assertEqual(status, 200)
        self.assertEqual(len(games), 100)
        self.assertEqual(games[0]['game_id'], 'g0')
        self.assertEqual(games[-1]['game_id'], 'g99')

    def test_game_decodes_valid_id_and_returns_complete_snapshot(self):
        payload = self.snapshot('match_A-1', turn_data={'player_1': 'Jos\u00e9'},
                                history=[{'direction': 'right'}])
        status, headers, body = self.request('/api/game/match%5FA-1?refresh=1')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode('utf-8')), payload)
        self.assertEqual(int(headers['Content-Length']), len(body))

    def test_game_rejects_empty_encoded_traversal_and_punctuation_ids(self):
        for game_id in ('', '..%2Fsecret', 'a%20b', 'a.json', '%2F', '%00'):
            with self.subTest(game_id=game_id):
                status, _, body = self.request('/api/game/' + game_id)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body), {'error': 'invalid game id'})

    def test_game_returns_not_found_for_missing_or_unreadable_snapshot(self):
        self.snapshot('broken')
        (viewer.LIVE_DIR / 'game_broken.json').write_text('not json')
        for game_id in ('missing', 'broken'):
            with self.subTest(game_id=game_id):
                status, _, body = self.request('/api/game/' + game_id)
                self.assertEqual(status, 404)
                self.assertEqual(json.loads(body), {'error': 'game not found'})

    def test_empty_snapshot_is_returned_by_detail_but_omitted_from_listing(self):
        self.snapshot('empty')
        (viewer.LIVE_DIR / 'game_empty.json').write_text('{}')
        status, _, body = self.request('/api/game/empty')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {})
        status, _, body = self.request('/api/games')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'games': []})

    def test_options_cors_headers_allow_known_origins_only(self):
        for origin in sorted(viewer.ALLOWED_BROWSER_ORIGINS) + ['https://example.invalid', None]:
            with self.subTest(origin=origin):
                status, headers, body = self.request('/api/games', 'OPTIONS', origin)
                self.assertEqual(status, 204)
                self.assertEqual(body, b'')
                self.assertEqual(headers['Cache-Control'], 'no-store')
                if origin in viewer.ALLOWED_BROWSER_ORIGINS:
                    self.assertEqual(headers['Access-Control-Allow-Origin'], origin)
                    self.assertEqual(headers['Access-Control-Allow-Methods'], 'GET, OPTIONS')
                    self.assertEqual(headers['Access-Control-Allow-Headers'], 'Content-Type')
                    self.assertEqual(headers['Access-Control-Allow-Private-Network'], 'true')
                    self.assertEqual(headers['Vary'], 'Origin')
                else:
                    self.assertNotIn('Access-Control-Allow-Origin', headers)

    def test_game_errors_include_cors_for_allowed_origin(self):
        status, headers, body = self.request('/api/game/missing', origin='http://localhost:8765')
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)['error'], 'game not found')
        self.assertEqual(headers['Access-Control-Allow-Origin'], 'http://localhost:8765')


if __name__ == '__main__':
    unittest.main()
