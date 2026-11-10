# Bot Snake CodeChallenge

[![Lint](https://github.com/Arielcohen1827/Codechallenge/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/Arielcohen1827/Codechallenge/actions/workflows/lint.yml)
[![Complexity](https://github.com/Arielcohen1827/Codechallenge/actions/workflows/complexity.yml/badge.svg?branch=main)](https://github.com/Arielcohen1827/Codechallenge/actions/workflows/complexity.yml)
[![Coverage](https://raw.githubusercontent.com/Arielcohen1827/Codechallenge/python-coverage-comment-action-data/badge.svg)](https://github.com/Arielcohen1827/Codechallenge/tree/python-coverage-comment-action-data)

Bot en Python para el Snake por turnos de Computación II 2026.
El cliente mantiene partidas independientes y responde al servidor por WebSocket.
Las fuentes oficiales guardadas están en `docs/reglas/`.

## Instalación

Requiere Python 3.10 o posterior y acceso al servidor de CodeChallenge.

En PowerShell, desde la raíz del repositorio, ejecutar

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-quality.txt
```

En Linux o macOS, ejecutar

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-quality.txt
```

`requirements.txt` declara el cliente WebSocket y coverage.
`requirements-quality.txt` declara Flake8, Xenon y Radon.

## Ejecución

El token personal se obtiene en My Bots de la plataforma. El cliente acepta el token
como argumento y se conecta a `wss://server.codechallenge.net.ar/ws`.

```powershell
python run.py <TOKEN_PERSONAL>
```

El lanzador de Windows también permite seleccionar un bot y abrir el visor local.
Copiar `.env_example` a `.env` y completar localmente los pares
`CODECHALLENGE_BOT_1_NAME` y `CODECHALLENGE_BOT_1_TOKEN`.
Para otro bot se cambia el número. Abrir `boot.bat` o `boot1.bat` después de
preparar el entorno con `install.bat`.

Los tokens deben permanecer en el entorno local. `.env`, los logs, los entornos
virtuales y los pesos activos están ignorados por Git.

## Reglas acumulativas y cubo 3D

El protocolo vigente v11 usa un cubo de 20 × 20 × 20.
Cada celda conserva las coordenadas `(level, row, col)`.
`up/down/left/right` cambian fila o columna, y `ascend/descend` cambian el nivel.
El nivel cero es el superior. Los disparos admiten las seis direcciones.

Las mecánicas acumulativas incluyen números en orden cíclico y sus copias,
multiplicadores X, paredes, choques y comida de cola, bombas de proximidad,
disparos y comida de disparo, estrellas e invencibilidad. En v11 cada dígito
tiene entre 10 y 15 copias, hay seis X, seis bombas grandes, quince pequeñas
y treinta estrellas durante el frenesí. Las paredes son placas cuadradas
que pierden su anillo exterior cada ronda. La estrella normal debe estar a
igual distancia Manhattan 3D de ambas cabezas y ser visible para ambas.

El servidor envía únicamente los tres planos que pasan por la cabeza propia.
`board_x` fija la columna y se indexa por nivel y fila.
`board_y` fija la fila y se indexa por nivel y columna.
`board_z` fija el nivel y se indexa por fila y columna.
`board` combina esos planos en orden x, y, z separados por una línea vacía.

`CubeObservation` deduplica las intersecciones e incluye espacios vacíos
observados. `KnowledgeState` conserva recuerdos con su turno de observación.
Volver a ver una celda vacía elimina el recuerdo anterior. Un objeto fuera
de los planos queda como información posiblemente desactualizada. Un rival
oculto tiene posición actual desconocida. La búsqueda y las verificaciones
de acciones conservan esa incertidumbre.

## Arquitectura

| Componente | Responsabilidad |
|---|---|
| `run.py` | WebSocket, eventos, envío de acciones y logs redactados |
| `snake_brain.py` | Coordinación y memoria independiente por partida |
| `geometry.py` | Coordenadas, vecinos y distancias compartidas |
| `observation.py` | Tres planos y conocimiento parcial |
| `snake_state.py` y `game_rules.py` | Estado y transiciones de las reglas |
| `food_planner.py` | Rutas, economía de objetivos y horizonte X |
| `move_search.py` y `search_engine.py` | Seguridad, búsqueda acotada y representación compacta |
| `v7_logging.py` | Predicciones, reconciliación y diagnóstico |
| `tools/bot_lab.py` | Simulación con estado verdadero y observaciones proyectadas |
| `viewer/` y `tools/match_viewer.py` | Visualización local de partidas y planos |

La simulación puede conocer el cubo completo. El bot recibe las observaciones
parciales. Las transiciones se concentran en el mismo motor de reglas.
Con observación parcial, la búsqueda táctica evalúa sucesores observados de
un movimiento y marca las respuestas rivales como inciertas. La búsqueda
adversarial profunda sigue disponible para estados completos.

## Tests y controles de calidad

Desde la raíz y con el entorno activado, ejecutar

```sh
python -m unittest discover
coverage erase
coverage run -m unittest discover
coverage report -m
coverage report --fail-under=90
flake8 .
```

`.coveragerc` mide todo el código propio, incluye las herramientas, excluye tests
y dependencias y establece `relative_files = true`.
El mínimo exigido es 90 %. Flake8 lee `setup.cfg` y utiliza un solo proceso
para mantener acotado el consumo de recursos.

Para medir todo el código Python versionado de la solución, excluyendo los
tests, ejecutar en PowerShell

```powershell
$sources = git ls-files '*.py' | Where-Object { $_ -notlike 'tests/*' }
xenon --max-absolute A --max-modules A --max-average A $sources
git diff --check
```

En Bash, el comando equivalente es

```sh
mapfile -d '' sources < <(git ls-files -z -- '*.py' ':(exclude)tests/**')
xenon --max-absolute A --max-modules A --max-average A "${sources[@]}"
```

Los tres límites A se aplican a cada bloque, al promedio por archivo y al
promedio del proyecto. Esta selección incluye todas las herramientas propias.

## CI y publicación de cobertura

Los workflows de lint, tests y complejidad se ejecutan en pushes a main y en
pull requests. Las pruebas cubren Python 3.10 y 3.12.
Cada control devuelve un fallo si no cumple su umbral.

El job de Python 3.12 publica los datos de cobertura en main y comenta los
pull requests mediante `python-coverage-comment-action`.
Tiene permisos `contents: write` y `pull-requests: write`.
El badge de cobertura se genera en la rama de datos de la acción.
Los pull requests externos publican el comentario mediante un artefacto y un
workflow posterior que consume ese comentario.
Los badges muestran el estado remoto publicado, que puede diferir del trabajo local.

## Logs, visor y laboratorio

Los logs y snapshots locales se escriben bajo `games/`.
Conservan la observación, la acción y diagnósticos de incertidumbre.
Los datos de autenticación se redactan antes de escribirlos.

```sh
python -m tools.match_viewer
python -m tools.bot_lab --help
python -m tools.bot_lab simulate --help
python -m tools.analyze_matches games/game_ejemplo.log
```

El visor permite inspeccionar los planos de una observación 3D.
Los comandos históricos de comparación y optimización siguen disponibles
bajo `tools/`. Los pesos activos se cargan desde el archivo local
`weights/active_weights.json`, cuando existe.

## Preparación para la defensa

`docs/GUIA_DEFENSA.md` explica el recorrido de una decisión y los controles.
La participación en torneos, la asignación del repositorio y la comprensión
del código se comprueban con evidencia externa y con la evaluación presencial.
