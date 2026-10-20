/*
Paste this whole file in the browser console while you are on:
https://codechallenge.net.ar/challenge

Keep the bot and local viewer running. Confirm explicitly, connect the local
monitor, then maintain at most 11 active/pending games, up to 22 total.
Only confirmed completions release slots. No retries of ambiguous POSTs.
*/
(() => {
  const CONFIG = {
    myBot: "Charmander",
    excludedBots: ["Charmander", "arielcohen"],
    game: "snake",
    maxGames: 11,
    totalGames: 22,
    monitorOrigin: "http://127.0.0.1:8765",
    telemetryTimeout: 15000,
    pendingTimeout: 120000,
    singleBotTelemetry: false,
    debugMode: false,
    dryRun: false,
  };

  const same = (a, b) => String(a || "").trim().toLowerCase() === String(b || "").trim().toLowerCase();
  const norm = (name) => String(name || "").trim().toLowerCase();
  const shortText = (value) => String(value || "").replace(/\s+/g, " ").trim().slice(0, 240);

  if (location.origin !== "https://codechallenge.net.ar") {
    throw new Error("Usar solamente en https://codechallenge.net.ar/challenge.");
  }
  if (!same(CONFIG.myBot, "Charmander")) {
    throw new Error("La confirmacion de visor exclusivo de este helper corresponde solo a Charmander.");
  }
  const previous = window.__charmanderRoundChallenger;
  const ledger = previous?.dispose?.() || {
    records: new Map(), reservations: [], seen: new Set(), spent: 0,
    baselineAt: 0, cursor: 0, halt: "", completions: 0, scopeName: CONFIG.myBot,
  };
  ledger.scopeName ||= "Charmander";
  if (!same(ledger.scopeName, CONFIG.myBot)) ledger.halt = "La sesion conservada pertenece a otro bot";
  previous?.stop?.();
  const runtime = { stopped: false, disposed: false, confirmed: false,
    ready: false, lastTelemetry: 0, busy: false, endpoint: null, frame: null,
    popup: null, sequence: 0, monitorBlock: "", singleBotTelemetry: false,
    message: "Confirmar antes de conectar el monitor." };
  const nonce = crypto.randomUUID();

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
      <strong style="font-size:13px">Pool ${CONFIG.myBot}: ${CONFIG.maxGames} / ${CONFIG.totalGames}</strong>
      <button data-stop style="margin-left:auto">Parar</button>
    </div>
    <div data-counts></div>
    <div data-status></div>
    <label style="display:block;margin-top:8px">
      <input data-single-bot type="checkbox"> El visor registra solo Charmander
    </label>
    <div style="display:flex;gap:8px;margin-top:8px">
      <button data-start>Confirmar y conectar</button>
      <button data-popup disabled>Abrir monitor</button>
    </div>
    <div style="margin-top:8px">Las partidas propias activas iniciales cuentan dentro de 22.
      Desafios entrantes u otros scripts pueden superar el limite global.</div>
    <pre data-log style="white-space:pre-wrap;margin:8px 0 0;max-height:36vh;overflow:auto"></pre>
  `;
  document.body.appendChild(panel);

  const statusEl = panel.querySelector("[data-status]");
  const logEl = panel.querySelector("[data-log]");
  const setStatus = (message) => {
    runtime.message = message;
    statusEl.textContent = message;
  };
  const countsEl = panel.querySelector("[data-counts]");
  const activeCount = () => [...ledger.records.values()].filter((g) => g.status !== "finished").length;
  const pendingCount = () => ledger.reservations.filter((r) => !r.gameId && r.state !== "rejected").length;
  const render = () => {
    countsEl.textContent = `Activas ${activeCount()} | pendientes ${pendingCount()} | cupo usado/reservado ${ledger.spent}/${CONFIG.totalGames} | partidas observadas ${ledger.records.size} | finalizadas ${[...ledger.records.values()].filter((g) => g.status === "finished").length} | max ${CONFIG.maxGames} | errores ${ledger.halt || runtime.monitorBlock || "ninguno"}`;
  };
  const log = (message) => {
    const line = `${new Date().toLocaleTimeString()} ${message}`;
    logEl.textContent = `${line}\n${logEl.textContent}`.slice(0, 7000);
  };
  const stop = () => {
    runtime.stopped = true;
    setStatus("Sin nuevos desafios. Se siguen observando los ya enviados; no se cancelan.");
    render();
  };
  panel.querySelector("[data-stop]").addEventListener("click", stop);

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
    /daily limit|limit of \d+ challenges|used all|limite diario|cupo diario|resets tomorrow|too many|concurrent|simultaneous|partidas activas|limite.*partidas/i
      .test(String(message || ""))
  );

  const postChallenge = async ({form, bot1, bot2, game}) => {
    if (CONFIG.dryRun) {
      log(`dry-run: ${bot1.name} vs ${bot2.name}`);
      return {accepted: false, limited: false};
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

    const action = new URL(form.getAttribute("action") || "/challenge", location.href);
    if (action.origin !== location.origin) {
      throw new Error("Destino del formulario fuera del origen permitido.");
    }
    const response = await fetch(action.href, {
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
      log(`fallo ${bot2.name}: ${limited ? "limite del servidor" : `HTTP ${response.status} o alerta del formulario`}`);
      // A server error can follow an accepted write. Keep its reservation.
      return {accepted: false, limited, ambiguous: !limited && (response.status >= 500 || response.ok)};
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

  const pause = (reason) => {
    ledger.halt = reason;
    setStatus("Envios pausados: " + reason + ". Las reservas se conservan.");
    render();
  };
  const blockMonitor = (reason) => {
    runtime.monitorBlock = reason;
    setStatus("Esperando telemetria segura: " + reason + ". Las reservas se conservan.");
    render();
  };
  const send = (type, extra = {}) => {
    if (runtime.endpoint && !runtime.disposed) {
      runtime.endpoint.postMessage({ protocol: "charmander-pool-v1", nonce, type, ...extra }, CONFIG.monitorOrigin);
    }
  };
  const trackedIds = () => [...ledger.records.values()]
    .filter((g) => g.status !== "finished").map((g) => g.game_id);
  const fresh = () => runtime.ready && Date.now() - runtime.lastTelemetry < CONFIG.telemetryTimeout;
  const playerName = (name) => String(name || "").split("@")[0].trim();
  const ownership = (g) => {
    const names = [g.player_1, g.player_2].map(playerName);
    const unknown = runtime.singleBotTelemetry ? "own" : "unknown";
    if (names.some((name) => !name || /^Jugador [AB]$/i.test(name))) return unknown;
    if (names.some((name) => same(name, CONFIG.myBot))) return "own";
    // Account-only names do not identify which bot of that account is playing.
    if (names.some((name) => same(name, "arielcohen"))) return unknown;
    return "foreign";
  };
  const rivalName = (g) => {
    const names = [g.player_1, g.player_2].map(playerName)
      .filter((name) => name && !/^Jugador [AB]$/i.test(name));
    const rivals = names.filter((name) => !same(name, CONFIG.myBot) && !same(name, "arielcohen"));
    return rivals.length === 1 ? rivals[0] : "";
  };
  const validGame = (g) => g && typeof g.game_id === "string"
    && /^[a-zA-Z0-9_-]{1,160}$/.test(g.game_id)
    && ["playing", "finished"].includes(g.status)
    && typeof g.player_1 === "string" && typeof g.player_2 === "string"
    && Number.isFinite(g.started_at) && Number.isFinite(g.updated_at);
  const consumeGame = (g, bootstrap) => {
    const old = ledger.records.get(g.game_id);
    if (old) {
      if (old.status !== "finished" && g.status === "finished") {
        ledger.completions += 1;
        log("Finalizacion confirmada: " + g.game_id);
      }
      // Terminal states never regress, even when a delayed snapshot arrives.
      if (old.status !== "finished") ledger.records.set(g.game_id, g);
      return;
    }
    if (ledger.seen.has(g.game_id)) return;
    if (g.status === "finished" && (bootstrap || g.started_at <= ledger.baselineAt)) {
      ledger.seen.add(g.game_id);
      return;
    }
    const owner = ownership(g);
    if (owner === "unknown") {
      blockMonitor("No se puede identificar el bot de la partida " + g.game_id
        + "; volver a pegar y marcar 'El visor registra solo Charmander' solo si es cierto");
      return;
    }
    ledger.seen.add(g.game_id);
    if (owner === "foreign") return;
    const rival = rivalName(g);
    // FIFO without a rival name is allowed only after the user's exclusive-scope attestation.
    const reservation = ledger.reservations.find((r) => !r.gameId && r.state !== "rejected"
      && (rival ? same(r.rival, rival) : runtime.singleBotTelemetry)
      && g.started_at >= r.at / 1000 - 2);
    if (reservation) {
      reservation.gameId = g.game_id;
    } else {
      // Incoming games also consume capacity and this batch's total budget.
      ledger.spent += 1;
    }
    ledger.records.set(g.game_id, g);
    if (g.status === "finished") ledger.completions += 1;
  };
  const canCreate = () => !runtime.stopped && !runtime.disposed && runtime.confirmed
    && fresh() && !runtime.monitorBlock && !ledger.halt && ledger.spent < CONFIG.totalGames;
  const fillPool = async () => {
    if (runtime.busy || !canCreate()) return;
    runtime.busy = true;
    try {
      const {doc, form} = await getChallengePage();
      if (!canCreate()) return;
      const myBot = findOption(optionData(doc.querySelector("[name=bot1]")), CONFIG.myBot);
      const game = findOption(optionData(doc.querySelector("[name=game]")), CONFIG.game);
      const rivals = challengeTargets(optionData(doc.querySelector("[name=bot2]")));
      if (!myBot || !game || !rivals.length) {
        pause("No hay formulario, bot, juego o rivales elegibles");
        return;
      }
      const capacity = Math.max(0, Math.min(CONFIG.maxGames - activeCount() - pendingCount(),
        CONFIG.totalGames - ledger.spent));
      if (CONFIG.dryRun) {
        for (let index = 0; index < capacity; index += 1) {
          log("dry-run: " + myBot.name + " vs " + rivals[(ledger.cursor + index) % rivals.length].name);
        }
        stop();
        return;
      }
      runtime.initialFill = true;
      runtime.lastFillCompletions = ledger.completions;
      // Reserve the entire parallel batch synchronously before any POST or message.
      const batch = Array.from({length: capacity}, () => {
        const bot = rivals[ledger.cursor++ % rivals.length];
        const reservation = {rival: bot.name, at: Date.now(), state: "inflight", gameId: null};
        ledger.reservations.push(reservation);
        ledger.spent += 1;
        return {bot, reservation};
      });
      render();
      const settled = await Promise.allSettled(batch.map(async ({bot, reservation}) => {
        try {
          const result = await postChallenge({form, bot1: myBot, bot2: bot, game});
          reservation.state = result.accepted ? "accepted" : result.ambiguous ? "ambiguous" : "rejected";
          if (!result.accepted) {
            // Attempts stay in the total budget, even definitive rejections.
            ledger.halt = result.limited ? "Cupo o limite del servidor" :
              result.ambiguous ? "POST ambiguo; no se reintenta" : "Desafio rechazado; no se reintenta";
          }
        } catch {
          reservation.state = "ambiguous";
          ledger.halt = "POST ambiguo o destino inseguro; no se reintenta";
        }
      }));
      if (settled.some((r) => r.status === "rejected")) ledger.halt = "Envio sin confirmar";
    } catch {
      ledger.halt = "No se pudo leer el formulario; sin reintentos automaticos";
    } finally {
      runtime.busy = false;
      if (!runtime.disposed) {
        render();
        if (ledger.halt) pause(ledger.halt);
        else if (runtime.monitorBlock) blockMonitor(runtime.monitorBlock);
        else if (ledger.spent >= CONFIG.totalGames) setStatus("Presupuesto agotado. Sin nuevos desafios; esperando el drenaje.");
        else if (!runtime.stopped) setStatus("Pool activo. Reposicion solo despues de finalizaciones confirmadas.");
        // A completion can arrive while POST responses are still in flight.
        if (canCreate() && ledger.completions > runtime.lastFillCompletions) {
          void fillPool();
        }
      }
    }
  };

  const onMessage = (event) => {
    if (runtime.disposed || event.origin !== CONFIG.monitorOrigin) return;
    const data = event.data;
    if (!data || data.protocol !== "charmander-pool-v1" || data.nonce !== nonce) return;
    const isFrame = runtime.frame && event.source === runtime.frame.contentWindow;
    const isPopup = runtime.popup && event.source === runtime.popup;
    if (!isFrame && !isPopup) return;
    if (data.type === "hello") {
      if (runtime.endpoint && runtime.endpoint !== event.source) return;
      runtime.endpoint = event.source;
      send("init", { ids: trackedIds(), myBot: CONFIG.myBot, singleBotTelemetry: runtime.singleBotTelemetry });
      return;
    }
    if (event.source !== runtime.endpoint) return;
    if (data.type === "error") {
      runtime.ready = false;
      blockMonitor("Monitor sin datos actuales; revisar el monitor local");
      return;
    }
    if (data.type !== "snapshot" || !Number.isSafeInteger(data.sequence)
      || data.sequence <= runtime.sequence || !Array.isArray(data.games)
      || !Array.isArray(data.missing) || typeof data.complete !== "boolean"
      || !Number.isFinite(data.at) || Math.abs(Date.now() - data.at) > CONFIG.telemetryTimeout
      || !data.games.every(validGame)) return;
    runtime.sequence = data.sequence;
    runtime.lastTelemetry = Date.now();
    runtime.monitorBlock = "";
    const bootstrap = !ledger.baselineAt;
    if (bootstrap) ledger.baselineAt = data.at / 1000;
    if (!data.complete) blockMonitor("Lista saturada de partidas activas; no se puede confirmar toda la capacidad");
    const relevantMissing = data.missing.filter((id) => ledger.records.has(id)
      || ownership(data.games.find((g) => g.game_id === id) || {}) !== "foreign");
    if (relevantMissing.length) blockMonitor("Partidas seguidas sin snapshot; no se liberan sus plazas");
    for (const g of data.games) {
      consumeGame(g, bootstrap);
      if (g.status === "playing" && ownership(g) !== "foreign"
        && data.at / 1000 - g.updated_at > 60) {
        blockMonitor("Snapshot activo antiguo; confirmar telemetria del bot antes de continuar");
      }
    }
    runtime.ready = data.complete && !relevantMissing.length;
    send("track", {ids: trackedIds()});
    render();
    if (activeCount() + pendingCount() > CONFIG.maxGames) pause("La actividad existente supera 11");
    if (runtime.stopped || ledger.halt || runtime.monitorBlock) return;
    if (ledger.spent >= CONFIG.totalGames) {
      setStatus(activeCount() + pendingCount() ? "Sin nuevos desafios. Drenando partidas restantes." : "Drenaje completo. Finalizado.");
      return;
    }
    if (!runtime.initialFill) {
      void fillPool();
    } else if (ledger.completions > runtime.lastFillCompletions && canCreate()) {
      void fillPool();
    }
  };
  window.addEventListener("message", onMessage);
  const monitorUrl = CONFIG.monitorOrigin + "/challenge_monitor.html#nonce=" + encodeURIComponent(nonce);
  const startButton = panel.querySelector("[data-start]");
  const popupButton = panel.querySelector("[data-popup]");
  const singleBotCheckbox = panel.querySelector("[data-single-bot]");
  singleBotCheckbox.checked = CONFIG.singleBotTelemetry === true;
  startButton.addEventListener("click", () => {
    if (runtime.stopped || runtime.confirmed) return;
    const scopeConfirmation = singleBotCheckbox.checked
      ? " Confirmas que este visor registra exclusivamente Charmander? Las partidas sin identidad se contaran como propias de Charmander."
      : " Las partidas sin identidad pausaran los envios.";
    if (!confirm("Mantener hasta 11 activas/pendientes y como maximo 22 de cupo usado/reservado (incluye propias ya activas)? No se cancelan partidas. Repetir rivales esta permitido si el servidor lo acepta." + scopeConfirmation)) return;
    runtime.singleBotTelemetry = singleBotCheckbox.checked;
    singleBotCheckbox.disabled = true;
    runtime.confirmed = true;
    startButton.disabled = true;
    popupButton.disabled = false;
    runtime.frame = document.createElement("iframe");
    runtime.frame.title = "Monitor local de partidas";
    runtime.frame.style.cssText = "width:100%;height:100px;border:0;margin-top:8px";
    runtime.frame.src = monitorUrl;
    panel.appendChild(runtime.frame);
    setStatus("Esperando handshake y snapshot seguro del monitor. Abrir monitor si el iframe esta bloqueado.");
  });
  popupButton.addEventListener("click", () => {
    if (!runtime.confirmed || runtime.stopped || fresh()) return;
    send("stop");
    runtime.endpoint = null;
    runtime.sequence = 0;
    runtime.ready = false;
    runtime.frame?.remove();
    runtime.frame = null;
    runtime.popup?.close();
    // Called directly from a user gesture. No noopener: handshake needs its opener.
    runtime.popup = window.open(monitorUrl, "charmanderMonitor-" + nonce, "popup,width=520,height=320");
    if (!runtime.popup) setStatus("Popup bloqueado. Permitirlo manualmente o usar el iframe; no se enviaron desafios.");
  });
  const watchdog = setInterval(() => {
    if (runtime.disposed) return;
    render();
    if (runtime.confirmed && !fresh()) {
      setStatus("Monitor desconectado, bloqueado o sin telemetria actual. Sin envios; reservas conservadas.");
    }
    if (ledger.reservations.some((r) => !r.gameId && r.state !== "rejected" && Date.now() - r.at > CONFIG.pendingTimeout)) {
      pause("Desafio pendiente sin ID durante 120s; comprobar manualmente, no se reenvia");
    }
    send("track", {ids: trackedIds()});
  }, 2000);
  window.__charmanderRoundChallenger = {
    stop,
    dispose: () => {
      runtime.stopped = true;
      clearInterval(watchdog);
      send("stop");
      runtime.disposed = true;
      window.removeEventListener("message", onMessage);
      runtime.frame?.remove();
      runtime.popup?.close();
      panel.remove();
      return ledger;
    },
  };
  render();
  setStatus(runtime.message);
})();
