"""Local launcher; .env values are literal, never evaluated or expanded.

Subset: KEY=value, blank lines, # comments, single/double quoted values.
Unquoted inline comments require whitespace before #. No escapes, multiline
values, export prefixes, interpolation or duplicate keys are supported.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parent
VIEWER_URL = 'http://127.0.0.1:8765/'


def load_bots(path, environ):
    values = {}
    exists = path.exists()
    if exists:
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
            if not match or match[1] in values:
                raise ValueError(f'Format .env invalido en linea {number}.')
            key, value = match.groups()
            if value.startswith(('"', "'")):
                quoted = re.fullmatch(r"(['\"])(.*?)\1\s*(?:#.*)?", value)
                if not quoted:
                    raise ValueError(f'Format .env invalido en linea {number}.')
                value = quoted[2]
            else:
                value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
                if value.startswith('#'):
                    value = ''
            values[key] = value

    indices = sorted({int(match[1]) for key in values
                      if (match := re.fullmatch(r'CODECHALLENGE_BOT_([1-9][0-9]*)_(NAME|TOKEN)', key))})
    bots = []
    for index in indices:
        name = values.get(f'CODECHALLENGE_BOT_{index}_NAME', '').strip() or f'Bot {index}'
        token = values.get(f'CODECHALLENGE_BOT_{index}_TOKEN', '').strip()
        if token:
            bots.append((name, token))
        else:
            print(f'Falta token para: {name}.')
    if not bots:
        token = environ.get('CODECHALLENGE_TOKEN', '').strip()
        if token:
            bots.append(('Token de entorno', token))
        elif not exists:
            raise ValueError('No existe .env en la raiz. Completa .env usando .env_example.')
        else:
            raise ValueError('No hay tokens configurados en .env ni CODECHALLENGE_TOKEN en el entorno.')
    return bots


def select_bot(bots):
    for number, (name, _) in enumerate(bots, 1):
        print(f'{number}. {name}')
    while True:
        choice = input('Selecciona el numero de bot: ').strip()
        if choice.isascii() and choice.isdigit() and len(choice) <= len(str(len(bots))):
            number = int(choice)
            if 1 <= number <= len(bots):
                return bots[number - 1][1]
        print('Seleccion invalida. Ingresa un numero de la lista.')


def viewer_healthy():
    try:
        with urllib.request.urlopen(VIEWER_URL + 'api/health', timeout=0.5) as response:
            return response.status == 200
    except OSError:
        return False


def start_viewer():
    if not viewer_healthy():
        subprocess.Popen(
            [sys.executable, str(ROOT / 'match_viewer.py')], cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(20):
            time.sleep(0.25)
            if viewer_healthy():
                break
        else:
            raise ValueError('El visualizador no responde en http://127.0.0.1:8765/.')
    webbrowser.open(VIEWER_URL)


def main():
    try:
        token = select_bot(load_bots(ROOT / '.env', os.environ))
        start_viewer()
        environ = os.environ.copy()
        environ['CODECHALLENGE_TOKEN'] = token
        return subprocess.run([sys.executable, str(ROOT / 'run.py'), token],
                              cwd=ROOT, env=environ).returncode
    except (KeyboardInterrupt, EOFError):
        print('\nInicio cancelado o ejecucion interrumpida.')
        return 0
    except (OSError, UnicodeError, subprocess.SubprocessError):
        # Process exceptions may contain the command (and hence the token).
        print('Error: no se pudo leer la configuracion o iniciar los procesos locales.')
        return 1
    except ValueError as error:
        print(f'Error: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
