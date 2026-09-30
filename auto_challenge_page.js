/*
Paste this in the browser console while you are on:
https://codechallenge.net.ar/challenge

It uses your already logged-in browser session and the real challenge form.
Run the Snake bot separately so it can accept/play the games.
*/
(() => {
  const CONFIG = {
    myBot: "Charmander",
    game: "snake",
    roundIntervalMs: 60_000,
    maxSimultaneousGames: 25,
    maxTotalGames: 30,
    assumedGameDurationMs: 8 * 60_000,
    debugMode: false,
    dryRun: false,
  };

  const STORE_KEY = "codechallenge_auto_challenger_charmander_v3";
  const same = (a, b) => String(a || "").trim().toLowerCase() === String(b || "").trim().toLowerCase();
  const norm = (name) => String(name || "").trim().toLowerCase();

  let stopped = false;
  let running = false;

  const loadState = () => {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    } catch {
      return {};
    }
  };

  const saveState = (state) => {
    localStorage.setItem(STORE_KEY, JSON.stringify(state));
  };

  const storedState = loadState();
  const state = {
    trackedGames: storedState.trackedGames || [],
    totalSent: 0,
  };
  saveState(state);

  const panel = document.createElement("div");
  panel.style.cssText = [
    "position:fixed",
    "right:16px",
    "bottom:16px",
    "z-index:2147483647",
    "width:380px",
    "max-height:48vh",
    "overflow:auto",
    "background:#101418",
    "color:#f4f7fb",
    "font:12px/1.35 system-ui,Segoe UI,sans-serif",
    "border:1px solid #2d3742",
    "box-shadow:0 8px 28px rgba(0,0,0,.35)",
    "border-radius:8px",
    "padding:10px",
  ].join(";");
  panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <strong style="font-size:13px">Auto desafiante: ${CONFIG.myBot}</strong>
      <button data-stop style="margin-left:auto">Parar</button>
    </div>
    <div data-status>Iniciando...</div>
    <pre data-log style="white-space:pre-wrap;margin:8px 0 0;max-height:34vh;overflow:auto"></pre>
  `;
  document.body.appendChild(panel);

  const statusEl = panel.querySelector("[data-status]");
  const logEl = panel.querySelector("[data-log]");
  panel.querySelector("[data-stop]").addEventListener("click", () => {
    stopped = true;
    statusEl.textContent = "Parado.";
  });

  const log = (message) => {
    const line = `${new Date().toLocaleTimeString()} ${message}`;
    console.log(`[auto-challenge] ${message}`);
    logEl.textContent = `${line}\n${logEl.textContent}`.slice(0, 5000);
  };

  const shortText = (value) => String(value || "").replace(/\s+/g, " ").trim().slice(0, 220);

  const optionData = (select) => Array.from(select?.options || []).map((option) => ({
    value: option.value,
    name: option.textContent.trim(),
  })).filter((option) => option.name);

  const getChallengePage = async () => {
    const response = await fetch("/challenge", {
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`GET /challenge HTTP ${response.status}`);
    }
    const html = await response.text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    const form = doc.querySelector("form");
    if (!form) {
      throw new Error("No encontre el formulario. Revisa si seguis logueado.");
    }
    return {doc, form};
  };

  const findOption = (options, name) => options.find((option) => same(option.name, name));

  const csrfFrom = (form) => form.querySelector("[name=csrfmiddlewaretoken]")?.value || "";

  const responseProblem = (text) => {
    const doc = new DOMParser().parseFromString(text, "text/html");
    const alertText = Array.from(doc.querySelectorAll(".alert-danger,.alert-error,.errorlist,[role=alert]"))
      .map((node) => shortText(node.textContent))
      .filter(Boolean)
      .join(" | ");
    if (alertText) {
      return alertText;
    }

    const bodyText = shortText(doc.body?.textContent || text);
    if (/csrf|forbidden|prohibido|unauthorized|no autorizado|login required|inicia sesi/i.test(bodyText)) {
      return bodyText;
    }
    return "";
  };

  const postChallenge = async ({form, bot1, bot2, game}) => {
    const body = new URLSearchParams();
    const csrf = csrfFrom(form);
    if (csrf) {
      body.set("csrfmiddlewaretoken", csrf);
    }
    body.set("bot1", bot1.value);
    body.set("bot2", bot2.value);
    body.set("game", game.value);
    if (CONFIG.debugMode) {
      body.set("debug_mode", "on");
    }

    if (CONFIG.dryRun) {
      log(`dry-run: ${bot1.name} vs ${bot2.name}`);
      return true;
    }

    const response = await fetch(form.action || "/challenge", {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        ...(csrf ? {"X-CSRFToken": csrf} : {}),
      },
      body,
      redirect: "follow",
    });
    const text = await response.text();
    if (!response.ok || response.url.includes("/login")) {
      log(`fallo ${bot2.name}: HTTP ${response.status} ${shortText(text)}`);
      return false;
    }
    const problem = responseProblem(text);
    if (problem) {
      log(`fallo ${bot2.name}: ${problem}`);
      return false;
    }
    log(`enviado: ${bot1.name} vs ${bot2.name}`);
    return true;
  };

  const pruneTrackedGames = (now) => {
    state.trackedGames = (state.trackedGames || []).filter((game) => (
      game && now - Number(game.sentAt || 0) < CONFIG.assumedGameDurationMs
    ));
    saveState(state);
  };

  const trackSentGame = (botName, now) => {
    state.trackedGames = state.trackedGames || [];
    state.trackedGames.push({
      opponent: botName,
      sentAt: now,
    });
    state.totalSent += 1;
    saveState(state);
  };

  const activeRowsFromPage = (doc) => {
    const statusPattern = /jugando|en juego|en curso|pendiente|aceptad|running|playing|active|pending|queued|started|in progress/i;
    const rowNodes = Array.from(doc.querySelectorAll("tr,li,.card,.list-group-item,.game,.match"));
    const rows = rowNodes
      .map((node) => shortText(node.textContent))
      .filter((text) => text && text.toLowerCase().includes(CONFIG.myBot.toLowerCase()))
      .filter((text) => text.toLowerCase().includes(CONFIG.game.toLowerCase()))
      .filter((text) => statusPattern.test(text));
    return rows.length;
  };

  const activeGameCount = (doc, now) => {
    pruneTrackedGames(now);
    const tracked = state.trackedGames.length;
    const detected = activeRowsFromPage(doc);
    return Math.max(tracked, detected);
  };

  const challengeTargets = (online) => online
    .filter((bot) => !same(bot.name, CONFIG.myBot))
    .sort((a, b) => norm(a.name).localeCompare(norm(b.name)));

  const tick = async () => {
    if (running || stopped) {
      return;
    }
    running = true;
    try {
      const now = Date.now();
      const {doc, form} = await getChallengePage();
      const mine = optionData(doc.querySelector("[name=bot1]"));
      const online = optionData(doc.querySelector("[name=bot2]"));
      const games = optionData(doc.querySelector("[name=game]"));
      const myBot = findOption(mine, CONFIG.myBot) || mine[0];
      const game = findOption(games, CONFIG.game) || games[0];

      if (!myBot || !game) {
        throw new Error("No encontre mi bot o el juego Snake en el formulario.");
      }

      const active = activeGameCount(doc, now);
      const slots = Math.max(0, CONFIG.maxSimultaneousGames - active);
      const remaining = Math.max(0, CONFIG.maxTotalGames - state.totalSent);
      const targets = challengeTargets(online);
      const selected = targets.slice(0, Math.min(slots, remaining));
      statusEl.textContent = `Enviadas: ${state.totalSent}/${CONFIG.maxTotalGames} | online: ${targets.length} | activas: ${active}/${CONFIG.maxSimultaneousGames}`;

      if (!remaining) {
        stopped = true;
        statusEl.textContent = `Completado: ${state.totalSent}/${CONFIG.maxTotalGames} desafios enviados.`;
        log("limite total alcanzado; el script se detuvo");
        return;
      }

      if (!slots) {
        log(`limite alcanzado: ${active}/${CONFIG.maxSimultaneousGames}; espero la proxima ronda`);
        return;
      }
      if (!selected.length) {
        log("no hay rivales online para desafiar");
        return;
      }

      await Promise.all(selected.map(async (bot) => {
        if (await postChallenge({form, bot1: myBot, bot2: bot, game})) {
          trackSentGame(bot.name, now);
        }
      }));
      if (state.totalSent >= CONFIG.maxTotalGames) {
        stopped = true;
        statusEl.textContent = `Completado: ${state.totalSent}/${CONFIG.maxTotalGames} desafios enviados.`;
        log("30 desafios enviados; el script se detuvo automaticamente");
      }
    } catch (error) {
      log(`error: ${error.message || error}`);
    } finally {
      running = false;
    }
  };

  if (!confirm("Activar auto desafios desde esta sesion web? Va a mandar desafios reales a bots online.")) {
    stopped = true;
    statusEl.textContent = "Cancelado.";
  } else {
    log("activo");
    tick();
    const timer = setInterval(() => {
      if (stopped) {
        clearInterval(timer);
      } else {
        tick();
      }
    }, CONFIG.roundIntervalMs);
  }
})();
