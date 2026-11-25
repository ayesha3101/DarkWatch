/* DarkWatch drawing engine. It only DRAWS what the replay file says; it never decides anything.
   Works in the browser (window.DarkRender) and in Node (module.exports) so it can be tested offline.

   Coordinates: a cell is (x, y), x to the right (east), y downward (south). A cell's centre is at integer (x, y). */
(function (root) {
  "use strict";

  const COLORS = {
    bg: "#060a08", water: "#0a1410", land: "#121a16", landEdge: "rgba(95,211,154,0.38)",
    green: "#2e8b57", bright: "#5fd39a", glow: "#a8ffd4", text: "#e7f0ea", muted: "#7f988b",
    warn: "#f2b84b", white: "#f4faf6"
  };

  let bounds = null;                       // the map area, so labels near the edge stay inside it
  let makeCanvas = function (w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    return c;
  };
  function setCanvasFactory(fn) { makeCanvas = fn; }

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const ease = (a) => (a < 0.5 ? 2 * a * a : 1 - Math.pow(-2 * a + 2, 2) / 2);
  function hex(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function mix(c1, c2, t) { return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)]; }
  const rgba = (c, a) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;

  /* ---------- preparing a replay (done once) ---------- */
  function prepare(replay) {
    if (replay._ui) return replay._ui;
    const frames = replay.frames, n = frames.length;
    const boatIds = frames[0].boats.map((b) => b.id);
    const lastSeen = {}, everDark = {};
    boatIds.forEach((id) => { lastSeen[id] = new Array(n).fill(null); everDark[id] = new Array(n).fill(false); });
    boatIds.forEach((id) => {
      let seen = null, dark = false;
      for (let i = 0; i < n; i++) {
        const b = frames[i].boats.find((q) => q.id === id);
        if (b.state === "visible") { seen = [b.x, b.y]; dark = false; }
        else if (b.state === "dark") dark = true;
        else if (b.state === "found") dark = true;
        lastSeen[id][i] = seen;
        everDark[id][i] = dark;
      }
    });
    const heading = new Array(n).fill(-Math.PI / 2);
    for (let i = 0; i < n; i++) {
      const p = frames[i].patrol, q = frames[Math.min(n - 1, i + 1)].patrol, o = frames[Math.max(0, i - 1)].patrol;
      let dx = q.x - p.x, dy = q.y - p.y;
      if (dx === 0 && dy === 0) { dx = p.x - o.x; dy = p.y - o.y; }
      heading[i] = dx === 0 && dy === 0 ? (i > 0 ? heading[i - 1] : -Math.PI / 2) : Math.atan2(dy, dx);
    }
    replay._ui = { lastSeen, everDark, heading, boatIds, n };
    return replay._ui;
  }

  function frameAt(replay, t) {
    const n = replay.frames.length;
    const i = clamp(Math.floor(t), 0, n - 1), j = Math.min(n - 1, i + 1);
    return { i, j, a: i === j ? 0 : clamp(t - i, 0, 1), f0: replay.frames[i], f1: replay.frames[j] };
  }

  /* ---------- layout ---------- */
  function layout(W, H, replay) {
    const cols = replay.map.width, rows = replay.map.height;
    const ml = 52, mr = 14, mt = 14, mb = 30;
    const cs = Math.max(4, Math.min((W - ml - mr) / cols, (H - mt - mb) / rows));
    const ox = ml + ((W - ml - mr) - cs * cols) / 2, oy = mt + ((H - mt - mb) - cs * rows) / 2;
    return { cols, rows, cs, ox, oy, W, H, mw: cs * cols, mh: cs * rows };
  }
  const cx = (L, x) => L.ox + (x + 0.5) * L.cs;
  const cy = (L, y) => L.oy + (y + 0.5) * L.cs;
  function cellAt(L, px, py) {
    const x = Math.floor((px - L.ox) / L.cs), y = Math.floor((py - L.oy) / L.cs);
    return x >= 0 && y >= 0 && x < L.cols && y < L.rows ? { x, y } : null;
  }

  /* ---------- the layer that never changes between frames: tiles, coast, axis labels ---------- */
  function staticLayer(S, L, dpr) {
    const key = [L.W, L.H, dpr, S.opts.density, S.replay.seed, S.replay.agent].join("|");
    if (S.cache && S.cache.key === key) return S.cache.canvas;
    const canvas = makeCanvas(Math.round(L.W * dpr), Math.round(L.H * dpr));
    const g = canvas.getContext("2d");
    g.scale(dpr, dpr);
    const { cost, density } = S.replay.map;
    const gap = L.cs > 12 ? 1 : 0.5;
    const deep = hex("#0b1712"), hot = hex(COLORS.bright);

    // map plate
    const plate = g.createLinearGradient(L.ox, L.oy, L.ox, L.oy + L.mh);
    plate.addColorStop(0, "#08100c"); plate.addColorStop(1, "#050a07");
    g.fillStyle = plate;
    roundRect(g, L.ox - 6, L.oy - 6, L.mw + 12, L.mh + 12, 12); g.fill();
    g.strokeStyle = "rgba(255,255,255,0.07)"; g.lineWidth = 1; g.stroke();

    for (let y = 0; y < L.rows; y++) {
      for (let x = 0; x < L.cols; x++) {
        const px = L.ox + x * L.cs, py = L.oy + y * L.cs;
        if (cost[y][x] === null) {                           // land
          g.fillStyle = COLORS.land;
          g.fillRect(px + gap / 2, py + gap / 2, L.cs - gap, L.cs - gap);
        } else {                                              // water, shaded by fishing activity
          const d = S.opts.density ? Math.pow(density[y][x], 0.75) : 0;
          g.fillStyle = rgba(mix(deep, hot, d * 0.34), 0.30 + 0.36 * d);
          g.fillRect(px + gap / 2, py + gap / 2, L.cs - gap, L.cs - gap);
        }
      }
    }
    // coastline: a thin bright edge on the sides of land that face the water
    g.strokeStyle = COLORS.landEdge; g.lineWidth = 1.4; g.lineCap = "round";
    g.beginPath();
    for (let y = 0; y < L.rows; y++) {
      for (let x = 0; x < L.cols; x++) {
        if (cost[y][x] !== null) continue;
        const px = L.ox + x * L.cs, py = L.oy + y * L.cs;
        const water = (xx, yy) => xx >= 0 && yy >= 0 && xx < L.cols && yy < L.rows && cost[yy][xx] !== null;
        if (water(x, y + 1)) { g.moveTo(px, py + L.cs); g.lineTo(px + L.cs, py + L.cs); }
        if (water(x, y - 1)) { g.moveTo(px, py); g.lineTo(px + L.cs, py); }
        if (water(x - 1, y)) { g.moveTo(px, py); g.lineTo(px, py + L.cs); }
        if (water(x + 1, y)) { g.moveTo(px + L.cs, py); g.lineTo(px + L.cs, py + L.cs); }
      }
    }
    g.stroke();
    axisLabels(g, S.replay, L);
    S.cache = { key, canvas };
    return canvas;
  }

  function axisLabels(g, replay, L) {
    const info = replay.map.grid_info;
    g.fillStyle = COLORS.muted; g.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    g.textAlign = "center"; g.textBaseline = "top";
    for (let x = 0; x < L.cols; x += 5) {
      const label = info ? (info.lon_min + x * info.step_deg).toFixed(1) + "°E" : String(x);
      g.fillText(label, cx(L, x), L.oy + L.mh + 9);
    }
    g.textAlign = "right"; g.textBaseline = "middle";
    for (let y = 0; y < L.rows; y += 5) {
      const label = info ? (info.lat_max - y * info.step_deg).toFixed(1) + "°N" : String(y);
      g.fillText(label, L.ox - 11, cy(L, y));
    }
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  /* ---------- the belief glow ---------- */
  function drawBelief(ctx, S, L, fr) {
    const { f0, f1, a } = fr;
    if (!f0.belief && !f1.belief) return;
    if (!S.beliefCanvas || S.beliefCanvas.width !== L.cols) S.beliefCanvas = makeCanvas(L.cols, L.rows);
    const bc = S.beliefCanvas, bg = bc.getContext("2d");
    const img = bg.createImageData(L.cols, L.rows), data = img.data;
    const A = f0.belief, B = f1.belief, w = ease(a);
    const vals = new Float32Array(L.cols * L.rows);
    let top = 0;
    for (let y = 0; y < L.rows; y++) for (let x = 0; x < L.cols; x++) {
      const v = lerp(A ? A[y][x] : 0, B ? B[y][x] : 0, w);
      vals[y * L.cols + x] = v; if (v > top) top = v;
    }
    const ref = Math.max(top, 0.1), mint = hex(COLORS.glow);
    for (let k = 0; k < vals.length; k++) {
      const v = Math.pow(clamp(vals[k] / ref, 0, 1), 0.55);
      data[k * 4] = mint[0]; data[k * 4 + 1] = mint[1]; data[k * 4 + 2] = mint[2];
      data[k * 4 + 3] = Math.round(255 * clamp(v * 0.9, 0, 0.9));
    }
    bg.putImageData(img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.beginPath(); ctx.rect(L.ox, L.oy, L.mw, L.mh); ctx.clip();
    if ("filter" in ctx) ctx.filter = `blur(${Math.max(2, L.cs * 0.35)}px)`;
    ctx.globalAlpha = 0.95;
    ctx.drawImage(bc, L.ox, L.oy, L.mw, L.mh);
    ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.45;
    ctx.drawImage(bc, L.ox, L.oy, L.mw, L.mh);
    ctx.restore();
  }

  /* ---------- small shapes ---------- */
  function glowDot(ctx, x, y, r, color, glow) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3.2);
    g.addColorStop(0, rgba(hex(glow), 0.55)); g.addColorStop(1, rgba(hex(glow), 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3.2, 0, 6.2832); ctx.fill();
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
  }
  function ship(ctx, x, y, angle, size) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(size * 1.15, 0); ctx.lineTo(-size * 0.8, size * 0.72); ctx.lineTo(-size * 0.4, 0); ctx.lineTo(-size * 0.8, -size * 0.72); ctx.closePath();
    ctx.fillStyle = COLORS.bright; ctx.shadowColor = COLORS.glow; ctx.shadowBlur = 14; ctx.fill();
    ctx.shadowBlur = 0; ctx.lineWidth = 1.2; ctx.strokeStyle = "#03150c"; ctx.stroke();
    ctx.restore();
  }
  function ring(ctx, x, y, r, color, width, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.stroke(); ctx.restore();
  }
  function badge(ctx, text, x, y, chosen) {
    ctx.save(); ctx.fillStyle = chosen ? COLORS.bright : "rgba(244,250,246,0.9)";
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.2832); ctx.fill();
    ctx.fillStyle = "#04120a"; ctx.font = "700 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y + 0.5); ctx.restore();
  }
  function tag(ctx, text, x, y, color) {
    ctx.save(); ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const w = ctx.measureText(text).width + 10;
    if (bounds) x = Math.max(bounds.left, Math.min(x, bounds.right - w));
    ctx.fillStyle = "rgba(5,10,7,0.78)"; roundRect(ctx, x, y - 8, w, 16, 5); ctx.fill();
    ctx.fillStyle = color || COLORS.text; ctx.fillText(text, x + 5, y + 0.5); ctx.restore();
  }

  /* ---------- everything else, every frame ---------- */
  function draw(ctx, W, H, dpr, S) {
    const replay = S.replay, ui = prepare(replay), L = layout(W, H, replay);
    bounds = { left: L.ox + 3, right: L.ox + L.mw - 3 };
    const fr = frameAt(replay, S.t), { i, j, a, f0, f1 } = fr, w = ease(a), pulse = 0.5 + 0.5 * Math.sin((S.time || 0) / 420);
    ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(staticLayer(S, L, dpr), 0, 0, W, H);

    if (S.opts.belief) drawBelief(ctx, S, L, fr);

    // port
    const port = replay.map.port;
    ctx.strokeStyle = "rgba(244,250,246,0.7)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(cx(L, port[0]) - 5, cy(L, port[1]) - 5, 10, 10);
    tag(ctx, "PORT", cx(L, port[0]) + 10, cy(L, port[1]), COLORS.muted);

    // patrol: wake, sensor square, planned direction
    const p0 = f0.patrol, p1 = f1.patrol, px = cx(L, lerp(p0.x, p1.x, w)), py = cy(L, lerp(p0.y, p1.y, w));
    const r = replay.config.radius || 1;
    if (S.opts.trails) {
      ctx.save(); ctx.lineCap = "round";
      const from = Math.max(0, i - 10);
      for (let k = from; k < i; k++) {
        const A = replay.frames[k].patrol, B = replay.frames[k + 1].patrol, fade = (k - from + 1) / (i - from + 1);
        ctx.strokeStyle = rgba(hex(COLORS.bright), 0.08 + 0.4 * fade); ctx.lineWidth = 1 + 2.2 * fade;
        ctx.beginPath(); ctx.moveTo(cx(L, A.x), cy(L, A.y)); ctx.lineTo(cx(L, B.x), cy(L, B.y)); ctx.stroke();
      }
      ctx.restore();
    }
    if (f0.target) {
      const tx = cx(L, f0.target[0]), ty = cy(L, f0.target[1]);
      ctx.save(); ctx.setLineDash([2, 7]); ctx.lineCap = "round"; ctx.strokeStyle = "rgba(168,255,212,0.55)"; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tx, ty); ctx.stroke(); ctx.restore();
      ring(ctx, tx, ty, 6 + 2 * pulse, "rgba(168,255,212,0.9)", 1.4);
      ctx.fillStyle = "rgba(168,255,212,0.9)"; ctx.beginPath(); ctx.arc(tx, ty, 1.8, 0, 6.2832); ctx.fill();
    }
    ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = "rgba(168,255,212,0.5)"; ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(168,255,212,0.05)";
    const sq = (2 * r + 1) * L.cs; roundRect(ctx, px - sq / 2, py - sq / 2, sq, sq, 4); ctx.fill(); ctx.stroke(); ctx.restore();

    // boats
    ui.boatIds.forEach((id) => {
      const b0 = f0.boats.find((q) => q.id === id), b1 = f1.boats.find((q) => q.id === id);
      const x = cx(L, lerp(b0.x, b1.x, w)), y = cy(L, lerp(b0.y, b1.y, w));
      const state = a < 0.5 ? b0.state : b1.state;
      if (state === "visible") {
        if (S.opts.trails) {
          const from = Math.max(0, i - 8);
          ctx.save(); ctx.lineCap = "round";
          for (let k = from; k < i; k++) {
            const A = replay.frames[k].boats.find((q) => q.id === id), B = replay.frames[k + 1].boats.find((q) => q.id === id);
            if (A.state !== "visible" || B.state !== "visible") continue;
            const fade = (k - from + 1) / (i - from + 1);
            ctx.strokeStyle = `rgba(244,250,246,${0.06 + 0.3 * fade})`; ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.moveTo(cx(L, A.x), cy(L, A.y)); ctx.lineTo(cx(L, B.x), cy(L, B.y)); ctx.stroke();
          }
          ctx.restore();
        }
        glowDot(ctx, x, y, 3.6, COLORS.white, "#ffffff");
        tag(ctx, String(id + 1), x + 8, y - 8, COLORS.text);
      } else if (state === "dark") {
        const seen = ui.lastSeen[id][i];
        if (seen) {
          const sx = cx(L, seen[0]), sy = cy(L, seen[1]);
          ring(ctx, sx, sy, 7 + 2.5 * pulse, "rgba(244,250,246,0.85)", 1.4, [3, 3]);
          ring(ctx, sx, sy, 2, "rgba(244,250,246,0.85)", 1.4);
          tag(ctx, `${id + 1} last seen`, sx + 11, sy - 9, COLORS.muted);
        }
        if (S.opts.truth) {
          ctx.save(); ctx.translate(x, y); ctx.rotate(0.7854);
          ctx.strokeStyle = COLORS.warn; ctx.lineWidth = 1.6; ctx.strokeRect(-4.5, -4.5, 9, 9); ctx.restore();
          tag(ctx, `${id + 1} (hidden)`, x + 9, y + 9, COLORS.warn);
        }
      } else {                                                      // found
        ring(ctx, x, y, 9, rgba(hex(COLORS.bright), 0.95), 2);
        ctx.strokeStyle = COLORS.bright; ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.beginPath(); ctx.moveTo(x - 3.5, y); ctx.lineTo(x - 0.8, y + 3); ctx.lineTo(x + 4, y - 3); ctx.stroke();
        tag(ctx, `${id + 1} found`, x + 12, y - 9, COLORS.bright);
      }
    });

    // "why this route": the best few places the agent weighed, ranked by its own score
    if (S.opts.why && f0.decision && f0.decision.candidates.length) {
      const all = f0.decision.candidates, key = all[0].utility !== undefined ? "utility" : all[0].visits !== undefined ? "visits" : "mass";
      const cs = all.slice().sort((p, q) => (q[key] || 0) - (p[key] || 0)).slice(0, 5), top = (cs[0][key] || 0) || 1;
      cs.forEach((c, k) => {
        const chosen = f0.target && c.x === f0.target[0] && c.y === f0.target[1], rad = 8 + 13 * Math.sqrt((c[key] || 0) / top);
        const x = cx(L, c.x), y = cy(L, c.y);
        ctx.fillStyle = chosen ? "rgba(95,211,154,0.16)" : "rgba(244,250,246,0.05)";
        ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.2832); ctx.fill();
        ring(ctx, x, y, rad, chosen ? COLORS.bright : "rgba(244,250,246,0.45)", chosen ? 2 : 1, chosen ? null : [3, 3]);
        badge(ctx, String(k + 1), x + rad * 0.72, y - rad * 0.72, chosen);
      });
    }

    // the patrol on top, with a fuel ring
    const heading = lerp(ui.heading[i], ui.heading[j], 0);
    const fuel = clamp(lerp(f0.fuel, f1.fuel, w) / (replay.config.fuel || 100), 0, 1);
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.beginPath(); ctx.arc(px, py, 20, 0, 6.2832); ctx.stroke();
    ctx.strokeStyle = fuel < 0.25 ? COLORS.warn : COLORS.bright;
    ctx.beginPath(); ctx.arc(px, py, 20, -Math.PI / 2, -Math.PI / 2 + 6.2832 * fuel); ctx.stroke(); ctx.restore();
    ship(ctx, px, py, heading, 10);

    // hover
    if (S.hover) {
      ctx.strokeStyle = "rgba(244,250,246,0.75)"; ctx.lineWidth = 1.2;
      ctx.strokeRect(L.ox + S.hover.x * L.cs + 0.5, L.oy + S.hover.y * L.cs + 0.5, L.cs - 1, L.cs - 1);
    }
    ctx.restore();
    return L;
  }

  const api = { COLORS, draw, prepare, frameAt, layout, cellAt, setCanvasFactory, ease, lerp };
  root.DarkRender = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);