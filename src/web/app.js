/* DarkWatch page logic. It plays back a replay file and explains it in plain words.
   All the thinking (belief, routes, choices) was done in Python; this file only shows it. */
(function () {
  "use strict";
  const R = window.DarkRender;
  const $ = (id) => document.getElementById(id);
  const BASE_FPS = 2;                       // frames per second at 1x (one frame = one 2 hour step)
  const STRATEGY = {
    "Random": "Picks a random nearby spot to visit. It has no memory and no plan.",
    "Fixed sweep": "Follows the same lawnmower route over the whole sea, whatever it sees.",
    "Last known position": "Heads to the place where a boat was last seen, nearest first. It does not guess where the boat went.",
    "Greedy belief": "Keeps a belief about each hidden boat and goes to the cell with the most belief, however far it is.",
    "Belief + A*": "Weighs the belief against the fuel: chance of a find divided by the cost of the route there.",
    "MCTS": "Imagines many possible futures, with boats placed where the belief says they might be, and picks the target that did best on average."
  };
  const WHY = {
    utility: "Score = chance of detection × belief in the sensor square ÷ (route fuel + 1). The longest bar wins.",
    visits: "Bars show how many of the imagined futures went for each place. The most visited place wins.",
    mass: "This strategy goes for the cell with the most belief, whatever it costs to get there."
  };
  const state = {
    replays: [], replay: null, S: { cache: null }, t: 0, playing: false, speed: 1, server: false, hover: null,
    events: [], frame: -1, lastDraw: 0, last: 0, agents: [],
    opts: { belief: true, density: true, trails: true, truth: false, why: false }
  };

  /* ---------- small helpers ---------- */
  const fix = (v, d) => Number(v).toFixed(d);
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  function banner(msg) { const b = $("banner"); b.textContent = msg || ""; b.hidden = !msg; }
  function validateReplay(r) {
    if (!r || typeof r !== "object") return "That file is not a replay.";
    if (r.format !== 1) return "That replay has an unsupported format (expected format 1).";
    if (!r.map || !Array.isArray(r.map.cost) || !r.map.cost.length) return "That replay has no map.";
    if (!Array.isArray(r.frames) || !r.frames.length) return "That replay has no frames.";
    return null;
  }
  function hours(r, i) { return Math.round(i * ((r.config && r.config.step_minutes) || 120) / 60); }

  /* ---------- replay store ---------- */
  function addReplay(r) {
    const bad = validateReplay(r);
    if (bad) { banner(bad); return false; }
    state.replays = state.replays.filter((q) => !(q.agent === r.agent && q.seed === r.seed));
    state.replays.push(r);
    return true;
  }
  function buildEvents(r) {
    const out = [];
    for (let i = 1; i < r.frames.length; i++) {
      r.frames[i].boats.forEach((b) => {
        const before = r.frames[i - 1].boats.find((q) => q.id === b.id);
        if (before.state === "visible" && b.state === "dark") out.push({ frame: i, kind: "dark", text: `Boat ${b.id + 1} switched off its AIS` });
        if (before.state !== "found" && b.state === "found") out.push({ frame: i, kind: "found", text: `Boat ${b.id + 1} found` });
      });
    }
    const s = r.summary;
    out.push({ frame: r.frames.length - 1, kind: "end", text: `Done: ${s.found} of ${s.went_dark} dark boats found` });
    return out;
  }
  function setReplay(r, keepPlaying) {
    state.replay = r; state.S = { cache: null }; state.t = 0; state.frame = -1;
    state.events = buildEvents(r); state.playing = !!keepPlaying;
    $("empty").hidden = true; $("done").hidden = true; $("hud").hidden = false;
    const n = r.frames.length;
    $("scrub").max = String(n - 1); $("scrub").value = "0";
    $("seed").value = r.seed;
    const badge = $("mapbadge"); badge.hidden = false;
    badge.textContent = r.map.source === "real" ? "Real map" : "Simulated map";
    badge.className = "badge" + (r.map.source === "real" ? "" : " sim");
    const hasBelief = r.frames.some((f) => f.belief);
    const chip = document.querySelector('.chip[data-opt="belief"]');
    chip.disabled = !hasBelief; chip.title = hasBelief ? "B" : "This strategy keeps no belief about hidden boats";
    buildTicks(r); renderEvents(); renderCompare(); refreshSelect(); fit(); updatePanels(0); syncPlay();
  }
  function buildTicks(r) {
    const n = Math.max(1, r.frames.length - 1), t = $("ticks");
    t.innerHTML = "";
    state.events.filter((e) => e.kind !== "end").forEach((e) => {
      const i = document.createElement("i"); i.className = e.kind; i.style.left = (100 * e.frame / n) + "%";
      i.title = `Step ${e.frame}: ${e.text}`; t.appendChild(i);
    });
  }
  function refreshSelect() {
    const sel = $("agent"), current = state.replay && state.replay.agent;
    if (state.server) { sel.value = current || sel.value; return; }
    const seed = state.replay && state.replay.seed;
    const names = state.replays.filter((q) => q.seed === seed).map((q) => q.agent);
    sel.innerHTML = "";
    names.forEach((n) => { const o = document.createElement("option"); o.value = n; o.textContent = n; sel.appendChild(o); });
    if (current) sel.value = current;
  }

  /* ---------- the plain-words explanation ---------- */
  function counts(r, i) {
    const boats = r.frames[i].boats, ui = R.prepare(r);
    return { dark: boats.filter((b) => b.state === "dark").length, found: boats.filter((b) => b.state === "found").length,
             visible: boats.filter((b) => b.state === "visible").length,
             everDark: ui.boatIds.filter((id) => ui.everDark[id][i]).length };
  }
  function story(r, i) {
    const c = counts(r, i), last = i === r.frames.length - 1, hasBelief = !!r.frames[i].belief;
    if (last) return `The episode is over. The patrol found ${c.found} of the ${r.summary.went_dark} boats that went dark and used ${fix(r.summary.fuel_used, 0)} units of fuel.`;
    if (c.everDark === 0) return `All ${c.visible} boats are still broadcasting AIS, so the patrol knows where they are. Nothing to search for yet.`;
    const parts = [];
    if (c.dark > 0) {
      parts.push(`${plural(c.dark, "boat has", "boats have")} gone dark.` + (hasBelief
        ? " The glow shows where they might be, worked out from their last known speed and heading."
        : " This strategy does not try to work out where they are."));
    } else parts.push("No boat is hidden right now.");
    if (c.found > 0) parts.push(`${plural(c.found, "boat is", "boats are")} already found.`);
    return parts.join(" ");
  }

  /* ---------- panels ---------- */
  function updatePanels(i) {
    const r = state.replay; if (!r) return;
    state.frame = i;
    const f = r.frames[i], c = counts(r, i), total = (r.config && r.config.fuel) || 100;
    $("hudStep").textContent = `Step ${i}`; $("hudTime").textContent = `${hours(r, i)} h`;
    $("sFound").textContent = f.found_total; $("sFoundOf").textContent = c.everDark ? `of ${c.everDark} that went dark` : "none gone dark yet";
    $("sDark").textContent = c.dark; $("sTime").textContent = `${hours(r, i)} h`; $("sStep").textContent = `step ${i} of ${r.frames.length - 1}`;
    $("sRate").textContent = f.fuel_used > 0 ? fix(100 * f.found_total / f.fuel_used, 1) : "0.0";
    const left = Math.max(0, f.fuel / total);
    $("fuelBar").style.width = (100 * left) + "%"; $("fuelBar").classList.toggle("low", left < 0.25);
    $("fuelText").textContent = `${fix(f.fuel, 1)} / ${fix(total, 0)}`;
    $("story").textContent = story(r, i);
    $("how").textContent = STRATEGY[r.agent] ? `${r.agent}: ${STRATEGY[r.agent]}` : r.agent;
    renderWhy(r, f);
    document.querySelectorAll("#events li[data-frame]").forEach((li) => li.classList.toggle("past", Number(li.dataset.frame) <= i));
  }
  function renderWhy(r, f) {
    const list = $("whyList"), text = $("whyText");
    list.innerHTML = "";
    const d = f.decision;
    if (!d || !d.candidates.length) {
      text.textContent = r.frames.length - 1 === f.step ? "The episode is over."
        : (r.frames.some((q) => q.decision) ? "No choice was needed at this step." : "This strategy does not weigh different places, so there is nothing to compare.");
      return;
    }
    const key = d.candidates[0].utility !== undefined ? "utility" : d.candidates[0].visits !== undefined ? "visits" : "mass";
    text.textContent = WHY[key];
    const cs = d.candidates.slice().sort((a, b) => (b[key] || 0) - (a[key] || 0)).slice(0, 5), top = (cs[0][key] || 0) || 1;
    cs.forEach((c, k) => {
      const chosen = f.target && c.x === f.target[0] && c.y === f.target[1];
      const li = document.createElement("li"); if (chosen) li.className = "chosen";
      li.innerHTML = `<span class="rank">${k + 1}</span><span class="meter"><i style="width:${Math.max(4, 100 * (c[key] || 0) / top)}%"></i></span><span class="cell">${c.x},${c.y}</span>`;
      list.appendChild(li);
    });
  }
  function renderEvents() {
    const ul = $("events"); ul.innerHTML = "";
    if (state.events.length <= 1 && state.replay && state.replay.summary.went_dark === 0) {
      ul.innerHTML = '<li class="empty">Nothing happened in this episode.</li>';
    }
    state.events.forEach((e) => {
      const li = document.createElement("li"); li.className = e.kind; li.dataset.frame = e.frame;
      li.innerHTML = `<span class="when">${hours(state.replay, e.frame)} h</span><span></span>`;
      li.lastChild.textContent = e.text;
      li.addEventListener("click", () => { state.playing = false; seek(e.frame); });
      ul.appendChild(li);
    });
  }
  function renderCompare() {
    const r = state.replay, same = r ? state.replays.filter((q) => q.seed === r.seed) : [];
    $("compareCard").hidden = same.length < 2;
    const ul = $("compareList"); ul.innerHTML = "";
    const top = Math.max(1, ...same.map((q) => q.summary.went_dark));
    same.forEach((q) => {
      const li = document.createElement("li"); if (q === r) li.className = "current";
      li.innerHTML = `<span class="name"></span><span class="num">${q.summary.found}/${q.summary.went_dark} found · ${fix(q.summary.fuel_used, 0)} fuel</span><span class="meter"><i style="width:${100 * q.summary.found / top}%"></i></span>`;
      li.firstChild.textContent = q.agent;
      li.addEventListener("click", () => { if (q !== state.replay) { setReplay(q, false); } });
      ul.appendChild(li);
    });
  }

  /* ---------- playback ---------- */
  function seek(t) {
    const r = state.replay; if (!r) return;
    state.t = Math.min(r.frames.length - 1, Math.max(0, t)); $("done").hidden = true; syncScrub();
    updatePanels(Math.floor(state.t));
  }
  function syncScrub() { $("scrub").value = String(state.t); }
  function syncPlay() {
    $("playIcon").innerHTML = state.playing ? '<path d="M7 5h3.6v14H7zM13.4 5H17v14h-3.6z" fill="currentColor"/>' : '<path d="M8 5 L19 12 L8 19 Z" fill="currentColor"/>';
  }
  function play() {
    const r = state.replay; if (!r) return;
    if (state.t >= r.frames.length - 1) seek(0);
    state.playing = true; $("done").hidden = true; syncPlay();
  }
  function pause() { state.playing = false; syncPlay(); }
  function step(d) { pause(); const r = state.replay; if (r) seek(Math.round(state.t) + d); }
  function finish() {
    pause(); const s = state.replay.summary;
    $("doneText").textContent = `${state.replay.agent} found ${s.found} of the ${s.went_dark} boats that went dark and used ${fix(s.fuel_used, 0)} fuel.`;
    $("done").hidden = false;
  }

  function fit() {
    const wrap = $("wrap"), c = $("map"), dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, wrap.clientWidth), h = Math.max(200, wrap.clientHeight);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); state.size = { w, h, dpr }; state.lastDraw = 0;
  }
  function frameLoop(now) {
    const dt = Math.min(0.1, (now - (state.last || now)) / 1000); state.last = now;
    const r = state.replay;
    if (r) {
      if (state.playing) {
        state.t += dt * BASE_FPS * state.speed;
        const end = r.frames.length - 1;
        if (state.t >= end) { state.t = end; syncScrub(); finish(); } else syncScrub();
      }
      const i = Math.floor(state.t);
      if (i !== state.frame) updatePanels(i);
      if (state.playing || now - state.lastDraw > 50 || state.dirty) {
        const ctx = $("map").getContext("2d"), z = state.size;
        Object.assign(state.S, { replay: r, t: state.t, time: reduceMotion ? 0 : now, opts: state.opts, hover: state.hover });
        R.draw(ctx, z.w, z.h, z.dpr, state.S); state.lastDraw = now; state.dirty = false;
      }
    }
    window.requestAnimationFrame(frameLoop);
  }
  const reduceMotion = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  /* ---------- hover tooltip ---------- */
  function onMove(e) {
    const r = state.replay; if (!r || !state.size) return;
    const rect = $("map").getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
    const cell = R.cellAt(R.layout(state.size.w, state.size.h, r), x, y), tip = $("tip");
    state.hover = cell; state.dirty = true;
    if (!cell) { tip.hidden = true; return; }
    const info = r.map.grid_info, land = r.map.cost[cell.y][cell.x] === null, f = r.frames[Math.floor(state.t)];
    const rows = [];
    if (info) rows.push(["Position", `${fix(info.lat_max - cell.y * info.step_deg, 2)}°N ${fix(info.lon_min + cell.x * info.step_deg, 2)}°E`]);
    if (land) rows.push(["Terrain", "Land"]);
    else {
      rows.push(["Fishing activity", `${Math.round(100 * r.map.density[cell.y][cell.x])}%`]);
      if (f.belief) rows.push(["Chance a hidden boat is here", `${fix(100 * f.belief[cell.y][cell.x], 1)}%`]);
    }
    tip.innerHTML = `<b>Cell ${cell.x}, ${cell.y}</b>` + rows.map((q) => `<div class="row"><span>${q[0]}</span><span>${q[1]}</span></div>`).join("");
    tip.hidden = false;
    const wrap = $("wrap").getBoundingClientRect();
    tip.style.left = Math.min(wrap.width - 230, x + 16) + "px"; tip.style.top = Math.min(wrap.height - 110, y + 16) + "px";
  }

  /* ---------- talking to the Python server (optional) ---------- */
  function loading(on, text) { $("loading").hidden = !on; if (text) $("loadingText").textContent = text; }
  async function getReplay(agent, seed) {
    const res = await fetch(`/api/replay?agent=${encodeURIComponent(agent)}&seed=${encodeURIComponent(seed)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }
  async function run(agent, seed) {
    if (!state.server) { banner("Running new episodes needs the server. Start it with: python src/server.py"); return; }
    banner(""); loading(true, `Running ${agent} in Python…`);
    try {
      const data = await getReplay(agent, seed);
      if (addReplay(data)) setReplay(data, true);
    } catch (err) { banner(err instanceof TypeError ? "Could not reach the DarkWatch server. Start it with: python src/server.py" : err.message); }
    finally { loading(false); }
  }
  async function compareAll() {
    if (!state.server) { banner("Comparing needs the server. Start it with: python src/server.py"); return; }
    const seed = Number($("seed").value) || 0; banner("");
    try {
      let first = null;
      for (let k = 0; k < state.agents.length; k++) {
        loading(true, `Running ${state.agents[k].name} (${k + 1} of ${state.agents.length})…`);
        const data = await getReplay(state.agents[k].name, seed);
        addReplay(data); if (!first || data.agent === $("agent").value) first = data;
      }
      setReplay(state.replays.find((q) => q.seed === seed && q.agent === $("agent").value) || first, false);
    } catch (err) { banner(err instanceof TypeError ? "Could not reach the DarkWatch server. Start it with: python src/server.py" : err.message); }
    finally { loading(false); }
  }

  /* ---------- files ---------- */
  function readFiles(files) {
    let opened = null, pending = files.length;
    Array.from(files).forEach((file) => {
      const fr = new FileReader();
      fr.onload = () => {
        try { const obj = JSON.parse(fr.result); if (addReplay(obj) && !opened) opened = obj; }
        catch (err) { banner(`${file.name} is not a valid replay file.`); }
        if (--pending === 0 && opened) { banner(""); setReplay(opened, false); }
      };
      fr.onerror = () => { banner(`Could not read ${file.name}.`); pending--; };
      fr.readAsText(file);
    });
  }

  /* ---------- wiring ---------- */
  function toggle(name) {
    const chip = document.querySelector(`.chip[data-opt="${name}"]`);
    if (!chip || chip.disabled) return;
    state.opts[name] = !state.opts[name]; chip.setAttribute("aria-pressed", String(state.opts[name])); state.dirty = true;
  }
  function showHelp(show) { $("legend").hidden = !show; if (!show) { try { localStorage.setItem("darkwatch_seen_help", "1"); } catch (e) { /* storage may be blocked */ } } }
  function bind() {
    $("play").addEventListener("click", () => (state.playing ? pause() : play()));
    $("prev").addEventListener("click", () => step(-1));
    $("next").addEventListener("click", () => step(1));
    $("again").addEventListener("click", () => { seek(0); play(); });
    $("scrub").addEventListener("input", (e) => seek(Number(e.target.value)));
    $("speed").addEventListener("change", (e) => { state.speed = Number(e.target.value); });
    document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => toggle(chip.dataset.opt)));
    $("run").addEventListener("click", () => run($("agent").value, $("seed").value));
    $("compare").addEventListener("click", compareAll);
    $("seed").addEventListener("keydown", (e) => { if (e.key === "Enter") run($("agent").value, $("seed").value); });
    $("dice").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 10000); if (state.server) run($("agent").value, $("seed").value); });
    $("agent").addEventListener("change", (e) => {
      if (state.server) { run(e.target.value, $("seed").value); return; }
      const next = state.replays.find((q) => q.agent === e.target.value && q.seed === state.replay.seed); if (next) setReplay(next, false);
    });
    $("file").addEventListener("change", (e) => { if (e.target.files.length) readFiles(e.target.files); e.target.value = ""; });
    $("help").addEventListener("click", () => showHelp(true));
    $("closeLegend").addEventListener("click", () => showHelp(false));
    $("legend").addEventListener("click", (e) => { if (e.target === $("legend")) showHelp(false); });
    const wrap = $("wrap");
    wrap.addEventListener("mousemove", onMove);
    wrap.addEventListener("mouseleave", () => { state.hover = null; $("tip").hidden = true; state.dirty = true; });
    ["dragover", "dragenter"].forEach((n) => document.addEventListener(n, (e) => { e.preventDefault(); $("drop").hidden = false; }));
    ["dragleave", "drop"].forEach((n) => document.addEventListener(n, (e) => { e.preventDefault(); if (n === "drop" || e.target === document.documentElement) $("drop").hidden = true; }));
    document.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) readFiles(e.dataTransfer.files); });
    window.addEventListener("resize", fit);
    if (window.ResizeObserver) new ResizeObserver(fit).observe(wrap);
    document.addEventListener("keydown", (e) => {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === " " && e.target.tagName !== "BUTTON") { e.preventDefault(); state.playing ? pause() : play(); }
      else if (k === "arrowright") step(1); else if (k === "arrowleft") step(-1);
      else if (k === "home") { pause(); seek(0); } else if (k === "end") { pause(); seek(Infinity); }
      else if (k === "b") toggle("belief"); else if (k === "g") toggle("truth"); else if (k === "w") toggle("why"); else if (k === "t") toggle("trails");
      else if (k === "h" || k === "?") showHelp($("legend").hidden); else if (k === "escape") showHelp(false);
    });
  }

  async function init() {
    bind(); fit(); window.requestAnimationFrame(frameLoop);
    let agents = null;
    try { const res = await fetch("/api/agents"); if (res.ok) agents = await res.json(); } catch (err) { /* no server: offline mode */ }
    if (agents) {
      state.server = true; state.agents = agents;
      const sel = $("agent"); sel.innerHTML = "";
      agents.forEach((a) => { const o = document.createElement("option"); o.value = a.name; o.textContent = a.slow ? `${a.name} (slower)` : a.name; sel.appendChild(o); });
      sel.value = agents.some((a) => a.name === "Belief + A*") ? "Belief + A*" : agents[0].name;
      run(sel.value, $("seed").value);
    } else {
      $("run").disabled = true; $("compare").disabled = true; $("seed").disabled = true; $("dice").disabled = true;
      $("run").title = $("compare").title = "Needs the server: python src/server.py";
      (window.DARKWATCH_REPLAYS || []).forEach(addReplay);
      const first = state.replays.find((q) => q.agent === "Belief + A*") || state.replays[0];
      if (first) setReplay(first, false);
    }
    if (!state.replay) { $("empty").hidden = false; renderCompare(); }
    let seen = false; try { seen = localStorage.getItem("darkwatch_seen_help") === "1"; } catch (e) { /* ignore */ }
    if (!seen) showHelp(true);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
  window.DarkApp = { state, validateReplay, buildEvents, setReplay, addReplay, seek, toggle };
})();