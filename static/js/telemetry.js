// Live SOC telemetry: KPI strip, top attack vectors and the security event feed.

import { api } from "./api.js";
import { badge, byId, h, humanize } from "./dom.js";

const POLL_INTERVAL_MS = 3000;

let timer = null;
let lastFeedSignature = "";

function renderKpis(snapshot) {
  byId("kpi-total").textContent = snapshot.total_requests;
  byId("kpi-blocked").textContent = snapshot.blocked_count;
  byId("kpi-flagged").textContent = snapshot.flagged_count;
  byId("kpi-redactions").textContent = snapshot.redactions_total;
  byId("kpi-latency").textContent = snapshot.avg_firewall_ms.toFixed(1);
  const missed = byId("kpi-missed");
  missed.textContent = snapshot.shadow_detections;
  missed.closest(".kpi").classList.toggle("is-hot", snapshot.shadow_detections > 0);
}

function renderVectors(vectors) {
  const max = Math.max(1, ...vectors.map((entry) => entry.count));
  const rows = vectors.map((entry) => {
    const fill = h("i");
    fill.style.width = `${(entry.count / max) * 100}%`;
    return h(
      "li",
      { class: "vector" },
      h("span", {}, humanize(entry.vector)),
      h("span", { class: "vector-count" }, entry.count),
      h("span", { class: "vector-bar" }, fill),
    );
  });
  byId("vector-list").replaceChildren(...rows);
}

function eventRow(event) {
  const time = new Date(event.timestamp).toLocaleTimeString([], { hour12: false });
  return h(
    "li",
    { class: `event event-${event.decision.toLowerCase()}` },
    h(
      "div",
      { class: "event-head" },
      h("time", { datetime: event.timestamp }, time),
      badge(event.decision),
      h("span", { class: "event-risk" }, `risk ${event.risk_score}`),
      event.enforced ? null : h("span", { class: "badge badge-off" }, "firewall off"),
      event.blocked_at ? h("span", { class: "tag" }, `stopped at ${event.blocked_at}`) : null,
      event.redaction_count ? h("span", { class: "tag" }, `${event.redaction_count} masked`) : null,
      event.attack_vectors
        .slice(0, 3)
        .map((vector) => h("span", { class: "tag" }, humanize(vector, { capitalize: false }))),
    ),
    h("div", { class: "event-preview", title: event.prompt_preview }, event.prompt_preview),
  );
}

function renderFeed(events) {
  // Skip the rebuild when nothing changed, so the list does not jump while being read.
  const signature = events.map((event) => event.event_id).join(",");
  if (signature === lastFeedSignature) return;
  lastFeedSignature = signature;
  byId("feed-list").replaceChildren(...events.map(eventRow));
}

function setLive(isLive) {
  const indicator = byId("live-state");
  indicator.textContent = isLive ? "Live" : "Disconnected";
  indicator.classList.toggle("is-stale", !isLive);
}

export async function refreshTelemetry() {
  try {
    const snapshot = await api.telemetry();
    renderKpis(snapshot);
    renderVectors(snapshot.top_attack_vectors);
    renderFeed(snapshot.recent_events);
    setLive(true);
  } catch {
    setLive(false);
  }
}

export function startTelemetry() {
  refreshTelemetry();
  timer = setInterval(() => {
    if (!document.hidden) refreshTelemetry();
  }, POLL_INTERVAL_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshTelemetry();
  });
  return () => clearInterval(timer);
}
