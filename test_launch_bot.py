import tempfile
import contextlib
import io
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import launch_bot


class TestConfiguration(unittest.TestCase):
    def load(self, content, environ=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text(content, encoding='utf-8')
            return launch_bot.load_bots(path, environ or {})

    def test_numbered_bot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('CODECHALLENGE_BOT_1_NAME=arielcohen\n'
                            'CODECHALLENGE_BOT_1_TOKEN=test-secret\n', encoding='utf-8')
            self.assertEqual(launch_bot.load_bots(path, {}), [('arielcohen', 'test-secret')])

    def test_sparse_indices_sorted_and_quotes_comments_literal(self):
        content = ('# Bots\n\nCODECHALLENGE_BOT_8_NAME="Charmander" # name\n'
                   "CODECHALLENGE_BOT_8_TOKEN='literal$()#secret'\n"
                   'CODECHALLENGE_BOT_2_NAME=arielcohen # name\n'
                   'CODECHALLENGE_BOT_2_TOKEN=second-secret # token\n')
        self.assertEqual(self.load(content), [('arielcohen', 'second-secret'),
                                             ('Charmander', 'literal$()#secret')])

    def test_empty_and_missing_file_fallback(self):
        for content in ('', 'CODECHALLENGE_BOT_1_TOKEN=\n'):
            with self.subTest(content=content):
                self.assertEqual(self.load(content, {'CODECHALLENGE_TOKEN': 'fallback'}),
                                 [('Token de entorno', 'fallback')])
        self.assertEqual(launch_bot.load_bots(Path('nonexistent-launcher-env'),
                                             {'CODECHALLENGE_TOKEN': 'fallback'}),
                         [('Token de entorno', 'fallback')])

    def test_configured_bots_take_precedence(self):
        self.assertEqual(self.load('CODECHALLENGE_BOT_3_TOKEN=configured\n',
                                   {'CODECHALLENGE_TOKEN': 'fallback'}), [('Bot 3', 'configured')])

    def test_missing_and_no_tokens_are_clear(self):
        with self.assertRaisesRegex(ValueError, 'No existe.*env'):
            launch_bot.load_bots(Path('nonexistent-launcher-env'), {})
        for content in ('', 'CODECHALLENGE_BOT_1_NAME=arielcohen\nCODECHALLENGE_BOT_1_TOKEN=\n'):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'tokens'):
                self.load(content)

    def test_reports_names_with_missing_tokens(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bots = self.load('CODECHALLENGE_BOT_1_NAME=arielcohen\n'
                             'CODECHALLENGE_BOT_1_TOKEN=\n'
                             'CODECHALLENGE_BOT_2_NAME=Charmander\n'
                             'CODECHALLENGE_BOT_2_TOKEN=good-secret\n')
        self.assertEqual(bots, [('Charmander', 'good-secret')])
        self.assertIn('arielcohen', output.getvalue())
        self.assertNotIn('good-secret', output.getvalue())

    def test_bad_lines_do_not_echo_content(self):
        for content in ('not-an-assignment secret', 'CODECHALLENGE_BOT_1_TOKEN="secret',
                        'CODECHALLENGE_BOT_1_TOKEN="secret" trailing',
                        'CODECHALLENGE_BOT_1_TOKEN=a\nCODECHALLENGE_BOT_1_TOKEN=b'):
            with self.subTest(content=content), self.assertRaises(ValueError) as error:
                self.load(content)
            self.assertNotIn('secret', str(error.exception))


class TestSelection(unittest.TestCase):
    def test_single_bot_still_prompts(self):
        with patch('builtins.input', return_value='1') as prompt, patch('builtins.print') as output:
            self.assertEqual(launch_bot.select_bot([('arielcohen', 'secret')]), 'secret')
        prompt.assert_called_once()
        self.assertIn('arielcohen', str(output.call_args_list))
        self.assertNotIn('secret', str(output.call_args_list))

    def test_invalid_choices_retry_then_second_bot(self):
        with patch('builtins.input', side_effect=['', 'bad', '0', '3', '-1', '2']), patch('builtins.print'):
            self.assertEqual(launch_bot.select_bot([('A', 'first'), ('B', 'second')]), 'second')


class TestLaunch(unittest.TestCase):
    def test_main_with_real_configuration_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('CODECHALLENGE_BOT_1_NAME=arielcohen\n'
                                      'CODECHALLENGE_BOT_1_TOKEN=first-secret\n'
                                      'CODECHALLENGE_BOT_9_NAME=Charmander\n'
                                      'CODECHALLENGE_BOT_9_TOKEN=second-secret\n', encoding='utf-8')
            for choice, expected in (('1', 'first-secret'), ('2', 'second-secret')):
                with self.subTest(choice=choice), patch('launch_bot.ROOT', root), \
                        patch('builtins.input', side_effect=['invalid', choice]), \
                        patch('builtins.print') as output, patch('launch_bot.start_viewer') as viewer, \
                        patch('launch_bot.subprocess.run', return_value=Mock(returncode=0)) as run:
                    self.assertEqual(launch_bot.main(), 0)
                    viewer.assert_called_once()
                    self.assertEqual(run.call_args.args[0][-1], expected)
                    self.assertEqual(run.call_args.kwargs['env']['CODECHALLENGE_TOKEN'], expected)
                    self.assertEqual(run.call_args.kwargs['cwd'], root)
                    self.assertNotIn(expected, str(output.call_args_list))

    def call_main(self, **kwargs):
        with patch('launch_bot.load_bots', return_value=[('A', 'secret')]), \
                patch('launch_bot.select_bot', **kwargs), \
                patch('launch_bot.start_viewer') as viewer, \
                patch('launch_bot.subprocess.run', return_value=Mock(returncode=7)) as run, \
                patch('builtins.print') as output:
            result = launch_bot.main()
        return result, viewer, run, output

    def test_launch_passes_token_env_and_root_without_mutating_environment(self):
        before = os.environ.copy()
        result, viewer, run, output = self.call_main(return_value='secret')
        self.assertEqual(result, 7)
        viewer.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], [launch_bot.sys.executable, str(launch_bot.ROOT / 'run.py'), 'secret'])
        self.assertEqual(kwargs['env']['CODECHALLENGE_TOKEN'], 'secret')
        self.assertEqual(kwargs['cwd'], launch_bot.ROOT)
        self.assertEqual(dict(os.environ), before)
        self.assertNotIn('secret', str(output.call_args_list))

    def test_cancel_does_not_start_viewer_or_bot(self):
        for error in (KeyboardInterrupt, EOFError):
            with self.subTest(error=error):
                result, viewer, run, _ = self.call_main(side_effect=error)
                self.assertEqual(result, 0)
                viewer.assert_not_called()
                run.assert_not_called()

    def test_bad_configuration_does_not_start_anything(self):
        for message in ('No existe .env', 'No hay tokens'):
            with self.subTest(message=message), patch('launch_bot.load_bots', side_effect=ValueError(message)), \
                    patch('launch_bot.start_viewer') as viewer, patch('launch_bot.subprocess.run') as run, \
                    patch('builtins.print') as output:
                self.assertEqual(launch_bot.main(), 1)
                viewer.assert_not_called()
                run.assert_not_called()
                self.assertIn(message, str(output.call_args_list))

    def test_process_errors_do_not_echo_tokens(self):
        for failure in (OSError('secret'), launch_bot.subprocess.SubprocessError('secret')):
            with self.subTest(failure=type(failure)), patch('launch_bot.load_bots', return_value=[('A', 'secret')]), \
                    patch('launch_bot.select_bot', return_value='secret'), patch('launch_bot.start_viewer'), \
                    patch('launch_bot.subprocess.run', side_effect=failure), patch('builtins.print') as output:
                self.assertEqual(launch_bot.main(), 1)
                self.assertNotIn('secret', str(output.call_args_list))

    def test_unreadable_configuration_does_not_echo_bytes_or_details(self):
        failures = (UnicodeDecodeError('utf-8', b'secret', 0, 6, 'invalid'), OSError('secret'))
        for failure in failures:
            with self.subTest(failure=type(failure)), \
                    patch('launch_bot.load_bots', side_effect=failure), \
                    patch('launch_bot.start_viewer') as viewer, \
                    patch('launch_bot.subprocess.run') as run, patch('builtins.print') as output:
                self.assertEqual(launch_bot.main(), 1)
                self.assertNotIn('secret', str(output.call_args_list))
                self.assertNotIn('0x', str(output.call_args_list))
                output.assert_called_once_with(
                    'Error: no se pudo leer la configuracion o iniciar los procesos locales.')
                viewer.assert_not_called()
                run.assert_not_called()

    def test_interrupt_during_bot_exits_without_pause(self):
        with patch('launch_bot.load_bots', return_value=[('A', 'secret')]), \
                patch('launch_bot.select_bot', return_value='secret'), patch('launch_bot.start_viewer'), \
                patch('launch_bot.subprocess.run', side_effect=KeyboardInterrupt), patch('builtins.print'):
            self.assertEqual(launch_bot.main(), 0)


class TestViewer(unittest.TestCase):
    def test_healthy_viewer_is_reused(self):
        with patch('launch_bot.viewer_healthy', return_value=True), \
                patch('launch_bot.subprocess.Popen') as process, patch('launch_bot.webbrowser.open') as browser:
            launch_bot.start_viewer()
        process.assert_not_called()
        browser.assert_called_once_with('http://127.0.0.1:8765/')

    def test_windows_viewer_starts_hidden_then_opens_when_ready(self):
        with patch('launch_bot.subprocess.CREATE_NO_WINDOW', 0x08000000, create=True), \
                patch('launch_bot.viewer_healthy', side_effect=[False, False, True]), \
                patch('launch_bot.os.name', 'nt'), patch('launch_bot.subprocess.Popen') as process, \
                patch('launch_bot.time.sleep'), patch('launch_bot.webbrowser.open') as browser:
            launch_bot.start_viewer()
        process.assert_called_once_with(
            [launch_bot.sys.executable, str(launch_bot.ROOT / 'match_viewer.py')],
            cwd=launch_bot.ROOT, creationflags=0x08000000,
            stdout=launch_bot.subprocess.DEVNULL, stderr=launch_bot.subprocess.DEVNULL)
        browser.assert_called_once()

    def test_viewer_timeout_is_clear_and_does_not_open_browser(self):
        with patch('launch_bot.viewer_healthy', return_value=False), \
                patch('launch_bot.subprocess.Popen'), patch('launch_bot.time.sleep'), \
                patch('launch_bot.webbrowser.open') as browser:
            with self.assertRaisesRegex(ValueError, 'visualizador'):
                launch_bot.start_viewer()
        browser.assert_not_called()

    def test_health_success_and_failure_use_local_url(self):
        with patch('launch_bot.urllib.request.urlopen') as request:
            request.return_value.__enter__.return_value.status = 200
            self.assertTrue(launch_bot.viewer_healthy())
            request.assert_called_once_with('http://127.0.0.1:8765/api/health', timeout=0.5)
        with patch('launch_bot.urllib.request.urlopen', side_effect=OSError('offline')):
            self.assertFalse(launch_bot.viewer_healthy())


class TestRunnerFiles(unittest.TestCase):
    def test_both_batches_share_launcher_and_preserve_error_status(self):
        for filename in ('boot.bat', 'boot1.bat'):
            with self.subTest(filename=filename):
                text = (launch_bot.ROOT / filename).read_text()
                self.assertIn('"venv\\Scripts\\python.exe" launch_bot.py', text)
                self.assertNotIn('run.py', text)
                self.assertNotIn('CODECHALLENGE_TOKEN', text)
                self.assertIn('if not exist "venv\\Scripts\\python.exe"', text)
                self.assertIn('set "launch_status=%errorlevel%"', text)
                self.assertIn('if not "%launch_status%"=="0" pause', text)
                self.assertIn('exit /b %launch_status%', text)

    def test_example_has_empty_tokens_and_expected_names(self):
        text = (launch_bot.ROOT / '.env_example').read_text()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'tokens'):
                launch_bot.load_bots(launch_bot.ROOT / '.env_example', {})
        self.assertIn('CODECHALLENGE_BOT_1_NAME=arielcohen', text)
        self.assertIn('CODECHALLENGE_BOT_2_NAME=Charmander', text)
        for line in text.splitlines():
            if '_TOKEN=' in line and not line.startswith('#'):
                self.assertEqual(line.split('=', 1)[1], '')

    def test_ignore_env_underscore_variants_except_example(self):
        text = (launch_bot.ROOT / '.gitignore').read_text().splitlines()
        self.assertIn('.env', text)
        self.assertIn('.env_*', text)
        self.assertIn('!.env_example', text)
        self.assertGreater(text.index('!.env_example'), text.index('.env_*'))


if __name__ == '__main__':
    unittest.main()
