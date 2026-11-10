# Guía de defensa del bot

El bot toma decisiones independientes para cada partida. Su objetivo es elegir
una acción útil dentro de un presupuesto breve, con las reglas del servidor
y con la información que realmente recibió.

## Recorrido de una acción

1. El WebSocket de `run.py` recibe un evento `your_turn`.
2. `SnakeBrain` identifica la partida y reconcilia la nueva observación.
3. El parser convierte los planos en coordenadas y construye `GameState`.
4. El planificador propone objetivos y rutas. La búsqueda compara acciones
   posibles bajo un límite de tiempo.
5. El cliente envía `move` o `shoot` con dirección y token del turno.
6. Guarda una predicción. La siguiente observación confirma lo verificable.

## Estado, observación y memoria

`GameState` representa las reglas y los datos utilizados por una transición.
Sus cuerpos y objetos usan coordenadas completas. En v11 son
`(level, row, col)`. El nivel cero está arriba.

`CubeObservation` transforma `board_x/y/z` en un mapa deduplicado.
El plano x fija la columna, el y fija la fila y el z fija el nivel.
Los espacios visibles también son información. Una celda fuera de esos tres
planos es desconocida.

`KnowledgeState` conserva lo visto antes con el turno de observación.
Cuando vuelve a verse una celda, la información actual reemplaza la anterior.
Los recuerdos fuera de plano pueden estar desactualizados. Una posición vieja
del rival no permite afirmar que sigue allí ni garantizar un disparo.

El simulador conoce un estado verdadero completo. Proyecta los tres planos
antes de entregárselos al bot. Así se puede comprobar la información parcial.

## Reglas y transiciones

`game_rules.py` contiene las reglas compartidas y `snake_state.py` aplica
acciones. Ambos estados, el normal y el compacto, deben producir el mismo
resultado para una acción determinista.

Los números forman una ventana cíclica de cinco dígitos. Comer el objetivo
retira todas sus copias y avanza la secuencia. Comer otro dígito aplica la
regla de penalización vigente. Bajo observación parcial, la ausencia de un
dígito visible no demuestra su ausencia en el cubo.

X da 50 puntos y aumenta permanentemente el multiplicador. Su valor estratégico
depende de los alimentos que puedan cobrarse después. El horizonte X contempla
esas rutas y la cantidad de turnos disponibles.

Un choque normal deja la cabeza en su lugar, conserva los primeros tres
segmentos y puede generar comida de cola para el rival. Las bombas usan
distancia Chebyshev 3D y su explosión tiene un tratamiento distinto al choque.
Las placas pierden el anillo exterior después de una ronda.

## Planificación y búsqueda

El food planner compara rutas, distancias, espacio disponible, competencia
por comida y recompensas. Manhattan es una cota geométrica. Los obstáculos
pueden hacer que la ruta real sea más larga o imposible.

`CompactState` guarda los datos en una forma adecuada para la búsqueda.
El índice de una celda 3D incluye nivel, fila y columna para que dos niveles
no se confundan. Las claves de caché deben distinguir también el conocimiento
relevante de cada observación.

La búsqueda usa profundidad iterativa. Primero completa una profundidad pequeña
y luego intenta profundizar mientras queda tiempo. Alpha-beta descarta ramas
que ya no pueden mejorar el resultado frente a la mejor respuesta disponible.
La tabla de transposiciones reutiliza posiciones ya evaluadas.

Con rival invisible, las respuestas tácticas inciertas deben degradarse con
prudencia. El presupuesto evita recorrer repetidamente todo el cubo en cada
decisión, especialmente con varias partidas simultáneas.
En producción, la observación parcial limita la búsqueda táctica a un
movimiento observado. El árbol adversarial profundo requiere un estado
completo, como el que puede proporcionar el laboratorio.

## Disparos e invencibilidad

Disparar cuesta 5000, mantiene al tirador quieto y sigue una de seis direcciones.
Una pared detiene el rayo. Un impacto corta desde la primera celda rival
alcanzada y puede producir comida de disparo.

Una estrella suma alimento y da 50 turnos propios de invencibilidad.
El contador baja también al disparar, enviar una acción inválida o agotar el
turno. Comer otra estrella lo reinicia. La cabeza invencible cambia a minúscula
y el cuerpo a mayúscula. El parser usa también los contadores del protocolo.
Un disparo no corta al rival invencible. Embestirlo durante invencibilidad
tiene su propia regla de corte.

## Tests y calidad

Los tests de comportamiento cubren casos normales, límites y combinaciones de
reglas. Los tests 3D comprueban planos, seis direcciones, memoria, bombas entre
niveles, placas, disparos y estrellas. Las regresiones históricas protegen
las reglas anteriores.

Coverage mide líneas ejecutadas. El umbral de 90 % ayuda a detectar zonas sin
pruebas, pero la corrección depende de lo que las aserciones comprueban.
Flake8 revisa errores y estilo. Xenon usa Radon para exigir complejidad A,
entre uno y cinco, en cada función y en los promedios.

Las funciones se separaron por responsabilidades concretas, como interpretar
una observación, aplicar una regla o resumir un resultado. Esto permite
comprobarlas y explicar cada paso sin seguir una gran cadena de condiciones.

Los workflows de GitHub ejecutan tests, cobertura, lint y complejidad.
La publicación de cobertura usa permisos de escritura para su rama de datos
y para los comentarios de pull requests. El resultado remoto se comprueba
después de publicar el commit.

Para preparar la defensa, conviene seguir una jugada desde su JSON hasta el
envío, explicar una transición y mostrar el test que verifica sus límites.
