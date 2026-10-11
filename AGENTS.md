# Snake CodeChallenge

Bot competitivo de Snake. Las reglas son acumulativas: cada semana puede
agregar mecanicas, y todas las anteriores deben seguir funcionando.

## Responsabilidades

- `snake_bot/snake_state.py`: reglas, estado del juego y transiciones.
- `snake_bot/snake_brain.py`: coordinacion de la decision.
- `snake_bot/food_planner.py`: planificacion y heuristicas.
- `snake_bot/move_search.py` y `snake_bot/search_engine.py`: busqueda tactica y profunda.
- `tools/`: laboratorio, analisis, benchmark, optimizador y servidor del viewer.
- `tests/`: tests de comportamiento; descubrimiento desde la raiz.

## Reglas de trabajo

- Antes de cambios grandes, analizar el codigo afectado, sus dependencias y
  tests. Mantener los cambios acotados y preservar cambios ajenos.
- Integrar cada regla nueva sin romper las anteriores. Separar reglas del
  juego de heuristicas: una preferencia estrategica no es una regla.
- Toda regla nueva debe tener tests de comportamiento. Si interactua con
  reglas anteriores, testear tambien esas combinaciones y casos limite.
- No agregar heuristicas solo porque parezcan buenas. Definir la hipotesis
  y comparar el cambio contra una linea base con tests, logs reales,
  `tools/bot_lab.py` y, cuando corresponda, `tools/optimize_weights.py`.
- Para comparar estrategias, usar las mismas semillas, escenarios y limites;
  registrar resultados de ambos lados, metricas y posibles regresiones.
  Cambiar una cosa medible por vez y comunicar si la evidencia no alcanza.
- Ejecutar `venv\Scripts\python.exe -m unittest discover -v` antes y despues
  de cambios relevantes. Reportar fallos preexistentes sin arreglarlos fuera
  del alcance de la tarea.
- No modificar pesos activos automaticamente, ni usar `--activate`, salvo
  que la tarea lo requiera explicitamente.
- No borrar ni sobrescribir logs de partidas para limpiar el proyecto.
- No agregar dependencias sin una necesidad concreta y justificada.
- Nunca guardar tokens o credenciales en archivos trackeados, tests o
  documentacion. Usar `.env` local ignorado o variables de entorno; mantener
  `.env_example` sin secretos y no mostrar tokens en salidas o respuestas.
