const state = { games: [], selectedId: null, refreshBusy: false };

const el = (id) => document.getElementById(id);
const directionNames = { up: 'ARRIBA', down: 'ABAJO', left: 'IZQUIERDA', right: 'DERECHA' };

function targetDigit(values) {
  const present = new Set(values.map(Number));
  if (!present.size) return null;
  const previous = (digit) => ((digit + 7) % 9) + 1;
  const candidates = [...present].filter((digit) => !present.has(previous(digit)));
  return Math.min(...(candidates.length ? candidates : present));
}

function relativeTime(timestamp) {
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - Number(timestamp || 0)));
  if (seconds < 5) return 'ahora';
  if (seconds < 60) return `hace ${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `hace ${minutes}m`;
}

function displayName(value, fallback) {
  const text = String(value || fallback);
  return text.includes('@') ? text.split('@')[0] : text;
}

function foodLabel(count) {
  const value = Number(count || 0);
  return `${value} comida${value === 1 ? '' : 's'}`;
}

function boardFoodCount(data, side) {
  const board = String(data.board || '');
  const length = [...board].filter((char) => char === side || char === side.toLowerCase()).length;
  return Math.max(0, length - 3);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function setConnection(online) {
  const connection = document.querySelector('.connection');
  connection.classList.toggle('online', online);
  connection.classList.toggle('offline', !online);
  el('connection-label').textContent = online ? 'Telemetria activa' : 'Sin conexion';
}

function renderGameList() {
  const list = el('game-list');
  const active = state.games.filter((game) => game.status === 'playing').length;
  el('live-count').textContent = `${active} activa${active === 1 ? '' : 's'}`;
  list.replaceChildren();
  for (const game of state.games) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `game-item ${game.status === 'finished' ? 'finished' : ''} ${game.game_id === state.selectedId ? 'selected' : ''}`;
    const player1 = escapeHtml(displayName(game.player_1, 'A'));
    const player2 = escapeHtml(displayName(game.player_2, 'B'));
    const timestamp = game.status === 'playing' ? game.started_at : game.updated_at;
    button.innerHTML = `
      <div class="game-item-top">
        <strong><span class="live-pip"></span>${player1} vs ${player2}</strong>
        <span>${relativeTime(timestamp)}</span>
      </div>
      <div class="game-item-score">
        <span>${game.status === 'playing' ? 'En juego' : 'Finalizada'} · Comidas ${game.food_eaten_1 || 0}-${game.food_eaten_2 || 0}</span>
        <b>${game.score_1} - ${game.score_2}</b>
      </div>`;
    button.addEventListener('click', () => {
      state.selectedId = game.game_id;
      renderGameList();
      refreshSelected();
    });
    list.appendChild(button);
  }
}

function parseBoard(data) {
  const lines = String(data.board || '').split(/\r?\n/).filter((line) => line.length);
  return lines.map((line) => {
    if (line.startsWith('|')) line = line.slice(1);
    if (line.endsWith('|')) line = line.slice(0, -1);
    return line;
  });
}

function cellClass(char, isTarget) {
  if (char === 'A') return 'head-a';
  if (char === 'a') return 'body-a';
  if (char === 'B') return 'head-b';
  if (char === 'b') return 'body-b';
  if (char === 'X') return 'pickup';
  if (char === '#') return 'wall';
  if (/^[1-9*]$/.test(char)) return isTarget ? 'target-food' : 'food';
  return '';
}

function renderBoard(data) {
  const rows = parseBoard(data);
  const rowCount = Number(data.rows) || rows.length || 1;
  const colCount = Number(data.cols) || Math.max(1, ...rows.map((row) => row.length));
  const digits = rows.flatMap((row) => [...row]).filter((char) => /^[1-9]$/.test(char));
  const next = targetDigit(digits);
  const board = el('board');
  board.style.setProperty('--rows', rowCount);
  board.style.setProperty('--cols', colCount);
  board.replaceChildren();
  for (let row = 0; row < rowCount; row += 1) {
    for (let col = 0; col < colCount; col += 1) {
      const char = rows[row]?.[col] || ' ';
      const cell = document.createElement('div');
      const isTarget = /^[1-9]$/.test(char) && Number(char) === next;
      cell.className = `cell ${cellClass(char, isTarget)}`;
      cell.textContent = char === ' ' ? '' : char;
      board.appendChild(cell);
    }
  }
  el('next-digit').textContent = next ?? '-';
}

function renderSnapshot(snapshot) {
  const data = snapshot.turn_data || {};
  const decision = snapshot.decision || {};
  const game = state.games.find((item) => item.game_id === snapshot.game_id);
  el('empty-state').hidden = true;
  el('match-content').hidden = false;
  el('version').textContent = `Charmander v${snapshot.bot_version || '?'}`;
  el('game-id').textContent = snapshot.game_id;
  el('match-title').textContent = `${displayName(data.player_1, 'Jugador A')} vs ${displayName(data.player_2, 'Jugador B')}`;
  el('turn-number').textContent = snapshot.turn_number || 0;
  el('remaining-moves').textContent = data.remaining_moves ?? '-';
  el('player-a').textContent = displayName(data.player_1, 'Jugador A');
  el('player-b').textContent = displayName(data.player_2, 'Jugador B');
  el('score-a').textContent = data.score_1 ?? 0;
  el('score-b').textContent = data.score_2 ?? 0;
  el('multiplier-a').textContent = `x${data.multiplier_1 ?? 1}`;
  el('multiplier-b').textContent = `x${data.multiplier_2 ?? 1}`;
  const foodEaten = snapshot.food_eaten || {};
  el('food-a').textContent = foodLabel(foodEaten.A ?? boardFoodCount(data, 'A'));
  el('food-b').textContent = foodLabel(foodEaten.B ?? boardFoodCount(data, 'B'));
  const status = el('match-status');
  status.textContent = snapshot.status === 'finished' ? 'FINALIZADA' : 'EN JUEGO';
  status.classList.toggle('finished', snapshot.status === 'finished');
  el('direction').textContent = directionNames[snapshot.direction] || '-';
  el('reason').textContent = decision.reason || '-';
  const plan = decision.plan || {};
  const target = plan.food || decision.target;
  el('target').textContent = Array.isArray(target) ? `[${target.join(', ')}]` : '-';
  el('safety').textContent = plan.safety || decision.safety || '-';
  el('legal-moves').textContent = Array.isArray(decision.legal) ? decision.legal.join(', ') : '-';
  renderBoard(data);
  if (game) game.updated_at = snapshot.updated_at;
}

async function refreshSelected() {
  if (!state.selectedId) return;
  const response = await fetch(`/api/game/${encodeURIComponent(state.selectedId)}`, { cache: 'no-store' });
  if (!response.ok) return;
  renderSnapshot(await response.json());
}

async function refresh() {
  if (state.refreshBusy) return;
  state.refreshBusy = true;
  try {
    const response = await fetch('/api/games', { cache: 'no-store' });
    if (!response.ok) throw new Error('telemetry unavailable');
    const payload = await response.json();
    state.games = payload.games || [];
    if (!state.games.some((game) => game.game_id === state.selectedId)) {
      state.selectedId = state.games[0]?.game_id || null;
    }
    renderGameList();
    if (state.selectedId) {
      await refreshSelected();
    } else {
      el('empty-state').hidden = false;
      el('match-content').hidden = true;
    }
    setConnection(true);
  } catch (_error) {
    setConnection(false);
  } finally {
    state.refreshBusy = false;
  }
}

refresh();
setInterval(refresh, 750);
