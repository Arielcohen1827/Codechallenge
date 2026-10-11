/*
Paste this whole file in the browser console while you are on:
https://codechallenge.net.ar/challenge

Keep boot.bat running. The script sends one challenge to every eligible online
bot, waits for those games to finish in the local viewer, and then starts the
next round. It has no internal round or challenge limit; server limits and
errors are still respected.
*/
(() => {
  const CONFIG = {
    myBot: "Charmander",
    excludedBots: ["Charmander", "arielcohen"],
    game: "snake",
    viewerGamesUrl: "http://127.0.0.1:8765/api/games",
    pollMs: 5_000,
    betweenRoundsMs: 5_000,
    noRivalsRetryMs: 20_000,
    gameStartTimeoutMs: 2 * 60_000,
    serverLimitRetryMs: 15 * 60_000,
    debugMode: false,
    dryRun: false,
  };

  const same = (a, b) => String(a || "").trim().toLowerCase() === String(b || "").trim().toLowerCase();
  const norm = (name) => String(name || "").trim().toLowerCase();
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const shortText = (value) => String(value || "").replace(/\s+/g, " ").trim().slice(0, 240);

  if (window.__charmanderRoundChallenger) {
    window.__charmanderRoundChallenger.stop();
  }

  const runtime = {
    stopped: false,
    round: 0,
    totalSent: 0,
    viewerErrorShown: false,
  };
  window.__charmanderRoundChallenger = {
    stop: () => {
      runtime.stopped = true;
    },
  };

  const panel = document.createElement("div");
  panel.style.cssText = [
    "position:fixed",
    "right:16px",
    "bottom:16px",
    "z-index:2147483647",
    "width:400px",
    "max-height:52vh",
    "overflow:auto",
    "background:#101418",
    "color:#f4f7fb",
    "font:12px/1.4 system-ui,Segoe UI,sans-serif",
    "border:1px solid #2d3742",
    "box-shadow:0 8px 28px rgba(0,0,0,.35)",
    "border-radius:8px",
    "padding:10px",
  ].join(";");
  panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <strong style="font-size:13px">Rondas automaticas: ${CONFIG.myBot}</strong>
      <button data-stop style="margin-left:auto">Parar</button>
    </div>
    <div data-status>Iniciando...</div>
    <pre data-log style="white-space:pre-wrap;margin:8px 0 0;max-height:36vh;overflow:auto"></pre>
  `;
  document.body.appendChild(panel);

  const statusEl = panel.querySelector("[data-status]");
  const logEl = panel.querySelector("[data-log]");
  const setStatus = (message) => {
    statusEl.textContent = message;
  };
  const log = (message) => {
    const line = `${new Date().toLocaleTimeString()} ${message}`;
    console.log(`[auto-challenge] ${message}`);
    logEl.textContent = `${line}\n${logEl.textContent}`.slice(0, 7000);
  };

  panel.querySelector("[data-stop]").addEventListener("click", () => {
    runtime.stopped = true;
    setStatus("Parado por el usuario.");
    log("detenido");
  });

  const optionData = (select) => Array.from(select?.options || [])
    .map((option) => ({value: option.value, name: option.textContent.trim()}))
    .filter((option) => option.name);

  const findOption = (options, name) => options.find((option) => same(option.name, name));
  const csrfFrom = (form) => form.querySelector("[name=csrfmiddlewaretoken]")?.value || "";

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
    if (/csrf|forbidden|unauthorized|login required|inicia sesi/i.test(bodyText)) {
      return bodyText;
    }
    return "";
  };

  const isServerLimit = (message) => (
    /daily limit|limit of \d+ challenges|used all|limite diario|cupo diario|resets tomorrow/i
      .test(String(message || ""))
  );

  const postChallenge = async ({form, bot1, bot2, game}) => {
    if (CONFIG.dryRun) {
      log(`dry-run: ${bot1.name} vs ${bot2.name}`);
      return {accepted: true, limited: false};
    }

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
    const problem = responseProblem(text);
    const limited = isServerLimit(problem || text);
    if (!response.ok || response.url.includes("/login") || problem) {
      log(`fallo ${bot2.name}: ${problem || `HTTP ${response.status}`}`);
      return {accepted: false, limited};
    }
    log(`enviado: ${bot1.name} vs ${bot2.name}`);
    return {accepted: true, limited: false};
  };

  const challengeTargets = (online) => {
    const seen = new Set();
    return online
      .filter((bot) => !CONFIG.excludedBots.some((name) => same(bot.name, name)))
      .filter((bot) => {
        const key = norm(bot.name);
        if (!key || seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      })
      .sort((a, b) => norm(a.name).localeCompare(norm(b.name)));
  };

  const viewerGames = async () => {
    const response = await fetch(`${CONFIG.viewerGamesUrl}?t=${Date.now()}`, {
      cache: "no-store",
      credentials: "omit",
    });
    if (!response.ok) {
      throw new Error(`visualizador HTTP ${response.status}`);
    }
    const payload = await response.json();
    return payload.games || [];
  };

  const opponentInGame = (game) => {
    if (same(game.player_1, CONFIG.myBot)) {
      return game.player_2;
    }
    if (same(game.player_2, CONFIG.myBot)) {
      return game.player_1;
    }
    return null;
  };

  const gameStartedAtMs = (game) => Number(game.started_at || 0) * 1000;

  const roundGames = (games, round) => {
    const expected = new Set(round.opponents.map(norm));
    const byOpponent = new Map();
    for (const game of games) {
      const opponent = opponentInGame(game);
      const key = norm(opponent);
      if (!expected.has(key) || gameStartedAtMs(game) < round.startedAt - 2_000) {
        continue;
      }
      const previous = byOpponent.get(key);
      if (!previous || gameStartedAtMs(game) > gameStartedAtMs(previous)) {
        byOpponent.set(key, game);
      }
    }
    return byOpponent;
  };

  const waitForRound = async (round) => {
    while (!runtime.stopped) {
      try {
        const games = await viewerGames();
        runtime.viewerErrorShown = false;
        const observed = roundGames(games, round);
        const active = Array.from(observed.values()).filter((game) => game.status !== "finished");
        const finished = Array.from(observed.values()).filter((game) => game.status === "finished");
        const missing = round.opponents.filter((name) => !observed.has(norm(name)));
        const startWindowExpired = Date.now() - round.startedAt >= CONFIG.gameStartTimeoutMs;

        setStatus(
          `Ronda ${round.number} | jugando ${active.length} | finalizadas ${finished.length}`
          + ` | por iniciar ${missing.length} | enviadas totales ${runtime.totalSent}`
        );

        if (!active.length && !missing.length) {
          log(`ronda ${round.number} terminada: ${finished.length}/${round.opponents.length} partidas`);
          return;
        }
        if (!active.length && startWindowExpired) {
          if (missing.length) {
            log(`ronda ${round.number}: no iniciaron ${missing.join(", ")}; no bloquean la siguiente ronda`);
          }
          return;
        }
      } catch (error) {
        setStatus("Esperando al visualizador local...");
        if (!runtime.viewerErrorShown) {
          runtime.viewerErrorShown = true;
          log(`no puedo consultar ${CONFIG.viewerGamesUrl}: ${error.message || error}`);
          log("deja boot.bat abierto para detectar cuando terminan las partidas");
        }
      }
      await sleep(CONFIG.pollMs);
    }
  };

  const startRound = async () => {
    const {doc, form} = await getChallengePage();
    const mine = optionData(doc.querySelector("[name=bot1]"));
    const online = optionData(doc.querySelector("[name=bot2]"));
    const games = optionData(doc.querySelector("[name=game]"));
    const myBot = findOption(mine, CONFIG.myBot) || mine[0];
    const game = findOption(games, CONFIG.game) || games[0];
    if (!myBot || !game) {
      throw new Error("No encontre Charmander o el juego Snake en el formulario.");
    }

    const targets = challengeTargets(online);
    if (!targets.length) {
      setStatus(`Sin rivales elegibles online | enviadas totales ${runtime.totalSent}`);
      log("no hay rivales elegibles online; vuelvo a consultar mas tarde");
      return {opponents: [], limited: false};
    }

    runtime.round += 1;
    const round = {
      number: runtime.round,
      startedAt: Date.now(),
      opponents: [],
    };
    setStatus(`Ronda ${round.number}: enviando a ${targets.length} bots online...`);
    log(`ronda ${round.number}: ${targets.map((bot) => bot.name).join(", ")}`);

    const results = await Promise.all(targets.map(async (bot) => ({
      bot,
      result: await postChallenge({form, bot1: myBot, bot2: bot, game}),
    })));
    round.opponents = results
      .filter(({result}) => result.accepted)
      .map(({bot}) => bot.name);
    round.limited = results.some(({result}) => result.limited);
    runtime.totalSent += round.opponents.length;
    return round;
  };

  const runForever = async () => {
    log("activo: una ronda por vez, sin limite interno");
    while (!runtime.stopped) {
      try {
        const round = await startRound();
        if (round.limited) {
          setStatus("El servidor rechazo por limite; reintento mas tarde.");
          log("limite del servidor detectado; pausa de 15 minutos");
          await sleep(CONFIG.serverLimitRetryMs);
          continue;
        }
        if (!round.opponents.length) {
          await sleep(CONFIG.noRivalsRetryMs);
          continue;
        }
        await waitForRound(round);
        if (!runtime.stopped) {
          setStatus(`Ronda ${round.number} completa. Preparando la siguiente...`);
          await sleep(CONFIG.betweenRoundsMs);
        }
      } catch (error) {
        log(`error: ${error.message || error}`);
        setStatus("Error temporal; reintento en 20 segundos.");
        await sleep(CONFIG.noRivalsRetryMs);
      }
    }
  };

  if (!confirm("Activar rondas automaticas? Se enviaran desafios reales a todos los bots online elegibles.")) {
    runtime.stopped = true;
    setStatus("Cancelado.");
  } else {
    runForever();
  }
})();
