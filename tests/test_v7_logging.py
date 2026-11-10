import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run
import v7_logging
from tests.test_run import FakeWebSocket, InTempDirTestCase, turn


class LoggingTests(InTempDirTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_all_observations_hide_tokens_but_wire_keeps_token(self):
        data = turn('|A* |\n|   |\n|  B|', turn_token='synthetic-private',
                    auth_token='synthetic-auth', unknown={'password': 'synthetic-pass'})
        ws = FakeWebSocket([{'event': 'your_turn', 'data': data},
                            {'event': 'game_over', 'data': {'game_id': 'g_1'}}])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            await run.play(ws)
        observed = output.getvalue() + self.read_log('g_1')
        observed += Path('games/live/game_g_1.json').read_text(encoding='utf-8')
        for secret in ('synthetic-private', 'synthetic-auth', 'synthetic-pass', 'turn_token'):
            self.assertNotIn(secret, observed)
        self.assertEqual(ws.sent[0]['data']['turn_token'], 'synthetic-private')
        self.assertEqual(data['auth_token'], 'synthetic-auth')

    async def test_connection_auth_remains_on_wire_only(self):
        ws = FakeWebSocket()
        uris = []

        class Connection:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *args):
                return False

        output = io.StringIO()

        def connect(uri):
            uris.append(uri)
            return Connection()

        with patch.object(run, 'websockets', SimpleNamespace(connect=connect)):
            with patch.object(run, 'play', side_effect=KeyboardInterrupt):
                with contextlib.redirect_stdout(output):
                    await run.start('synthetic-auth')
        self.assertNotIn('synthetic-auth', output.getvalue())
        self.assertEqual(
            uris, ['wss://server.codechallenge.net.ar/ws?token=synthetic-auth']
        )

    async def test_malformed_and_exception_text_is_not_echoed(self):
        ws = FakeWebSocket(['synthetic-malformed-secret'])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            await run.play(ws)
        self.assertNotIn('synthetic-malformed-secret', output.getvalue())

    async def test_observed_transition_and_final_unknown_are_retained(self):
        ws = FakeWebSocket()
        first = turn('|Aa  |\n|   B|', remaining=20, score_1=100)
        second = turn('| aA |\n|  B |', remaining=18, score_1=101)
        with patch.object(run.BOT, 'choose_move', return_value='right'), \
                patch.object(run.BOT, 'safe_direction', return_value='right'), \
                patch.object(run.BOT, 'commit_move'):
            await run.process_move(ws, {'data': first})
            await run.process_move(ws, {'data': second})
        await run.on_game_over(ws, {'event': 'game_over', 'data': {'game_id': 'g_1'}})
        entries = [json.loads(line[2:]) for line in self.read_log('g_1').splitlines()]
        audits = [entry['verification'] for entry in entries if 'verification' in entry]
        self.assertEqual(audits[1]['before']['board'], first['board'])
        self.assertEqual(audits[1]['after']['board'], second['board'])
        self.assertEqual(audits[1]['score_deltas'], {'A': 1, 'B': 0})
        self.assertEqual(audits[1]['before']['sent_move'], 'right')
        self.assertEqual(audits[1]['after']['crashed'], {'A': None, 'B': None})
        self.assertIn('rival_actions_unobserved', audits[1]['uncertainty'])
        self.assertEqual(audits[-1]['game_over_cause'], 'unknown')
        self.assertIsNone(audits[-1]['after']['board'])

    async def test_diagnostic_failure_does_not_block_send_or_cleanup(self):
        ws = FakeWebSocket()
        with patch.object(run, 'verification_record', side_effect=ValueError('private')):
            await run.process_move(ws, {'data': turn('|A* |\n|   |\n|  B|')})
            await run.on_game_over(ws, {'event': 'game_over', 'data': {'game_id': 'g_1'}})
        self.assertEqual(ws.sent[0]['data']['direction'], 'right')
        self.assertNotIn('g_1', run.BOT.observed_states)

    async def test_stdout_failure_does_not_prevent_wire_send(self):
        ws = FakeWebSocket()
        with patch.object(run, 'print_diagnostic', side_effect=OSError('synthetic-private')):
            await run.send(ws, 'move', {'turn_token': 'synthetic-private', 'direction': 'up'})
        self.assertEqual(ws.sent[0]['data']['turn_token'], 'synthetic-private')

    async def test_full_move_and_cleanup_survive_all_diagnostic_sink_failures(self):
        ws = FakeWebSocket()
        with patch.object(run, 'print_diagnostic', side_effect=OSError('private')), \
                patch.object(run, 'write_live_snapshot', side_effect=TypeError('private')), \
                patch.object(run, 'log_debug', side_effect=ValueError('private')), \
                patch.object(run, 'log_event', side_effect=ValueError('private')), \
                patch.object(run, 'log_action', side_effect=ValueError('private')):
            await run.on_your_turn(ws, {'data': turn('|A* |\n|   |\n|  B|')})
            self.assertIn('g_1', run.BOT.pending_moves)
            await run.on_game_over(ws, {'data': {'game_id': 'g_1'}})
        self.assertEqual(ws.sent[0]['data']['direction'], 'right')
        self.assertNotIn('g_1', run.BOT.pending_moves)

    def test_console_sink_encoding_and_io_failures_are_ignored(self):
        with patch('builtins.print', side_effect=OSError('private')):
            run.print_diagnostic('safe')
        with patch.object(run.sys, 'stdout', SimpleNamespace(encoding='unknown-codec')):
            run.print_diagnostic('safe')

    async def test_play_console_failure_still_dispatches_move(self):
        ws = FakeWebSocket([{'event': 'your_turn', 'data': turn('|A* |\n|   |\n|  B|')}])
        with patch.object(run, 'print_diagnostic', side_effect=OSError('private')):
            await run.play(ws)
        self.assertEqual(ws.sent[0]['data']['direction'], 'right')
        self.assertIn('g_1', run.BOT.pending_moves)

    def test_live_serialization_failure_is_private_and_fail_open(self):
        output = io.StringIO()
        with patch.object(run.json, 'dumps', side_effect=TypeError('synthetic-private')):
            with contextlib.redirect_stdout(output):
                run.write_live_snapshot('g_1', turn('|AB|'))
        self.assertNotIn('synthetic-private', output.getvalue())

    async def test_connection_console_failure_does_not_block_mock_auth(self):
        ws = FakeWebSocket()

        class Connection:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *args):
                return False

        with (
            patch('builtins.print', side_effect=OSError('private')),
            patch.object(
                run,
                'websockets',
                SimpleNamespace(connect=lambda uri: Connection()),
            ),
            patch.object(run, 'play', side_effect=KeyboardInterrupt),
            patch.object(run.time, 'sleep', side_effect=RuntimeError('retry')),
        ):
            await run.start('synthetic-auth')

    async def test_failed_final_event_log_does_not_rewrite_last_action(self):
        ws = FakeWebSocket()
        await run.process_move(ws, {'data': turn('|A* |\n|   |\n|  B|')})
        previous = list(run.HISTORY['g_1'])
        with patch.object(run, 'log_event', side_effect=ValueError('private')):
            await run.on_game_over(ws, {'event': 'game_over', 'data': {'game_id': 'g_1'}})
        self.assertEqual(run.HISTORY['g_1'], previous)

    def test_tail_symbols_are_net_changes_with_unknown_cleanup(self):
        before = v7_logging.verification_record(turn('|A\u24b6\u24b7B|', remaining=20))
        after = v7_logging.verification_record(turn('|A \u24b7B|', remaining=18), previous=before)
        self.assertEqual(after['tail_net_changes']['\u24b6']['removed'], [[0, 1]])
        self.assertEqual(after['tail_origin_by_symbol'], {'\u24b6': 'B', '\u24b7': 'A'})
        self.assertEqual(after['tail_cleanup_or_consumption'], 'unknown')

    def test_absent_fields_and_duplicate_stale_gaps_do_not_invent_evidence(
        self,
    ):
        before = v7_logging.verification_record(
            turn('|AB|', remaining=20, score_1=500)
        )
        for horizon, transition in (
            (20, 'duplicate_or_correction'),
            (21, 'stale'),
        ):
            after = v7_logging.verification_record(
                turn('|AB|', remaining=horizon), previous=before
            )
            self.assertEqual(after['transition'], transition)
            self.assertEqual(after['score_deltas'], {'A': None, 'B': None})
        after = v7_logging.verification_record(
            {'game_id': 'g_1'}, previous=before, finished=True
        )
        self.assertIsNone(after['after']['scores']['A'])
        self.assertIsNone(after['after']['remaining_moves'])
        self.assertEqual(after['game_over_cause'], 'unknown')
        self.assertEqual(after['last_observed_snapshot'], before['after'])

    def test_cumulative_crash_history_uses_observed_not_predicted_geometry(self):
        data = turn('|AaB|', remaining=18)
        state = run.parse_state(data, crash_history={'A': True, 'B': False})
        record = v7_logging.verification_record(data, state)
        self.assertEqual(record['after']['crashed'], {'A': True, 'B': False})
        self.assertEqual(record['crash_cause'], {'A': 'unknown', 'B': 'unknown'})
        mismatch = v7_logging.verification_record(turn('| A B|'), state)
        self.assertIsNone(mismatch['after']['tracked_body'])
        self.assertEqual(mismatch['after']['crashed'], {'A': None, 'B': None})

    def test_final_causes_remain_possible_or_ambiguous(self):
        for extra, cause in (
            ({'remaining_moves': 0}, 'possible_horizon_exhaustion'),
            (
                {'score_1': v7_logging.ELIMINATION_SCORE_THRESHOLD - 1},
                'possible_score_threshold',
            ),
            (
                {
                    'remaining_moves': 0,
                    'score_1': v7_logging.ELIMINATION_SCORE_THRESHOLD - 1,
                },
                'ambiguous_horizon_or_score_threshold',
            ),
        ):
            record = v7_logging.verification_record(
                turn('|AB|', **extra), finished=True
            )
            self.assertEqual(record['game_over_cause'], cause)

    async def test_official_final_cause_preserved_without_replacing_inference(self):
        for cause in ('horizon', 'score_threshold'):
            game_id = 'official_' + cause
            request = {'event': 'game_over', 'data': {'game_id': game_id,
                       'reason': cause, 'cause': cause, 'turn_token': 'synthetic-private'}}
            await run.on_game_over(FakeWebSocket(), request)
            event = json.loads(run.HISTORY[game_id][-1][2:])
            self.assertEqual(event['data']['reason'], cause)
            self.assertEqual(event['verification']['server_reported_cause'],
                             {'reason': cause, 'cause': cause})
            self.assertEqual(event['verification']['game_over_cause'], 'unknown')

    async def test_credential_bearing_reason_and_cause_are_redacted_in_all_sinks(
        self,
    ):
        for marker in (
            'token',
            'auth',
            'secret',
            'password',
            'credential',
            'bearer',
            'api_key',
        ):
            for field in ('reason', 'cause'):
                self.assertEqual(
                    v7_logging.safe_data(
                        {field: marker + '=synthetic-private'}
                    )[field],
                    '[redacted]',
                )
        self.assertEqual(
            v7_logging.safe_message({'reason': 'api_key=synthetic-private'})[
                'reason'
            ],
            '[redacted]',
        )
        for field, text in (
            ('reason', 'invalid turn_token=synthetic-private'),
            ('cause', 'Bearer synthetic-private'),
        ):
            game_id = 'redacted_' + field
            request = {
                'event': 'game_over',
                'data': {'game_id': game_id, field: text},
            }
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                await run.play(FakeWebSocket([request]))
            observed = output.getvalue() + self.read_log(game_id)
            observed += Path(f'games/live/game_{game_id}.json').read_text(
                encoding='utf-8'
            )
            self.assertNotIn('synthetic-private', observed)
            event = json.loads(run.HISTORY[game_id][-1][2:])
            self.assertEqual(event['data'][field], '[redacted]')

    def test_nested_debug_secrets_and_unknown_protocol_fields_are_removed(self):
        debug = {'event': 'decision_debug', 'unknown': 'synthetic-secret',
                 'plan': {'food': [0, 1], 'auth_token': 'synthetic-private',
                          'api_key': 'synthetic-key', 'auth': 'synthetic-auth'}}
        safe = v7_logging.safe_message(debug)
        self.assertEqual(safe, {'event': 'decision_debug', 'plan': {'food': [0, 1]}})
        self.assertEqual(v7_logging.safe_data({'board': '|AB|', 'unknown': 'secret'}),
                         {'board': '|AB|'})

    def test_same_board_with_different_horizon_or_scores_has_no_old_history(self):
        data = turn('|Aa B|', remaining=20, score_1=500)
        state = run.parse_state(data, crash_history={'A': True, 'B': False})
        for changed in (turn(data['board'], remaining=19, score_1=500),
                        turn(data['board'], remaining=20, score_1=0),
                        {'game_id': 'g_1', 'board': data['board']}):
            record = v7_logging.verification_record(changed, state, finished=True)
            self.assertIsNone(record['after']['tracked_body'])
            self.assertEqual(record['after']['crashed'], {'A': None, 'B': None})

    def test_sent_prediction_collision_causes_and_unknown_crash_score(self):
        for board, direction, cause in (
            ('|A  B|', 'left', 'border'),
            ('|A# B|', 'right', 'wall'),
            ('|Aa B|', 'right', 'self'),
            ('|AB  |', 'right', 'rival'),
        ):
            data = turn(board, score_1=500)
            record = v7_logging.verification_record(
                data, run.parse_state(data), direction=direction
            )
            prediction = record['sent_action_prediction']
            self.assertEqual(prediction['collision_cause'], cause)
            self.assertIsNone(prediction['expected_score'])
            self.assertEqual(
                prediction['source'], 'prediction_not_server_confirmation'
            )
            self.assertEqual(
                len(prediction['expected_body']),
                len(run.parse_state(data).body('A')),
            )

    def test_prediction_tail_reward_and_released_tail_growth_nuance(self):
        data = turn('|A\u24b6 B|', score_1=100, multiplier_1=2)
        state = run.parse_state(data)
        record = v7_logging.verification_record(data, state, direction='right')
        prediction = record['sent_action_prediction']
        self.assertEqual(
            prediction['tail_interaction'],
            {
                'symbol': '\u24b6',
                'beneficiary': 'A',
                'originator': 'B',
                'event': 'consume',
            },
        )
        self.assertEqual(
            prediction['expected_score'],
            run.apply_move(state, 'right').scores['A'],
        )
        from dataclasses import replace

        data = turn('|AaB|\n|aa |', score_1=0)
        state = replace(
            run.parse_state(data),
            snakes={'A': ((0, 0), (1, 0), (1, 1), (0, 1)), 'B': ((0, 2),)},
        )
        record = v7_logging.verification_record(data, state, direction='right')
        self.assertIsNone(record['sent_action_prediction']['collision_cause'])
        growing = replace(state, food=frozenset({(0, 1)}))
        record = v7_logging.verification_record(
            data, growing, direction='right'
        )
        self.assertEqual(
            record['sent_action_prediction']['collision_cause'], 'self'
        )

    def test_prediction_reconciliation_requires_single_action_and_explicit_fields(
        self,
    ):
        data = turn('|A  B|', remaining=20, score_1=100)
        state = run.parse_state(data)
        before = v7_logging.verification_record(data, state, direction='right')
        before = json.loads(json.dumps(before))
        after = v7_logging.verification_record(
            turn('| A B|', side='B', remaining=19, score_1=101),
            previous=before,
        )
        self.assertEqual(
            after['prediction_reconciliation'],
            'consistent_body_and_score_only',
        )
        gap = v7_logging.verification_record(
            turn('| A B|', remaining=18, score_1=101), previous=before
        )
        self.assertEqual(gap['prediction_reconciliation'], 'unavailable')

    def test_known_crash_prediction_and_unmatched_single_action(self):
        data = turn('|A  B|', remaining=20, score_1=500)
        state = run.parse_state(data, crash_history={'A': True})
        before = v7_logging.verification_record(data, state, direction='left')
        self.assertEqual(
            before['sent_action_prediction']['expected_score'],
            run.apply_move(state, 'left').scores['A'],
        )
        after = v7_logging.verification_record(
            turn('|A  B|', side='B', remaining=19, score_1=1), previous=before
        )
        self.assertEqual(
            after['prediction_reconciliation'], 'differs_body_or_score'
        )
        self.assertEqual(after['crash_cause']['A'], 'unknown')

    def test_reconciliation_checks_head_marker_and_explicit_next_side(self):
        data = turn('|Aa B|\n|    |', remaining=20, score_1=100)
        state = run.parse_state(data)
        before = json.loads(
            json.dumps(
                v7_logging.verification_record(data, state, direction='down')
            )
        )
        for board, side in (('|A  B|\n|a   |', 'B'), ('|a  B|\n|A   |', 'A')):
            after = v7_logging.verification_record(
                turn(board, side=side, remaining=19, score_1=101),
                previous=before,
            )
            self.assertEqual(
                after['prediction_reconciliation'], 'differs_head_or_side'
            )

    def test_prediction_reports_detached_tail_cells_and_beneficiary(self):
        for side, board, symbol, beneficiary in (
            ('A', '|Aaaa B|', '\u24b7', 'B'),
            ('B', '|Bbbb A|', '\u24b6', 'A'),
        ):
            data = turn(board, side=side, remaining=20)
            state = run.parse_state(data)
            prediction = v7_logging.verification_record(
                data, state, direction='left'
            )['sent_action_prediction']
            self.assertEqual(
                prediction['expected_detached_tail'],
                {
                    'cells': [[0, 3]],
                    'symbol': symbol,
                    'beneficiary': beneficiary,
                    'originator': side,
                },
            )
            self.assertEqual(len(prediction['expected_body']), 3)

    def test_reward_prediction_requires_matching_supplied_multiplier(self):
        for board in ('|A\u24b6 B|', '|A* B|'):
            data = turn(board, remaining=20, score_1=100, multiplier_1=2)
            state = run.parse_state(data)
            changed = dict(data, multiplier_1=3)
            prediction = v7_logging.verification_record(
                changed, state, direction='right'
            )['sent_action_prediction']
            self.assertIsNone(prediction['expected_score'])

    def test_windows_print_fallback_and_utf8_log(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding='cp1252')
        with patch.object(run.sys, 'stdout', stream):
            run.print_diagnostic('\u24b6\u24b7')
        stream.flush()
        self.assertEqual(
            stream.buffer.getvalue(),
            (
                b'\\u24b6\\u24b7\r\n'
                if __import__('os').name == 'nt'
                else b'\\u24b6\\u24b7\n'
            ),
        )
        run.HISTORY['utf8'] = ['\u24b6\u24b7']
        run.write_game_log('utf8')
        self.assertEqual(
            Path('games/game_utf8.log').read_text(encoding='utf-8'),
            '\u24b6\u24b7\n',
        )
