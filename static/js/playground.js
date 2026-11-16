// Interactive playground: chat transcript, firewall switch and one-click scenarios.

import { api } from "./api.js";
import { badge, byId, h, plural } from "./dom.js";

// Zero-width and bidi controls, plus the Unicode tag block used for ASCII smuggling.
const INVISIBLE = /[­​-‏‪-‮⁠-⁤⁦-⁩﻿]|[\u{E0000}-\u{E007F}]/gu;

const MODE_NOTES = {
  on: "Every prompt is inspected before it reaches the model, and every answer before it reaches you.",
  off: "Unprotected. Prompts reach the model as typed; the firewall only records what it would have done.",
};

export function createPlayground({ onResult }) {
  const panel = document.querySelector(".playground");
  const form = byId("chat-form");
  const input = byId("message");
  const sendButton = byId("send");
  const transcript = byId("transcript");
  const runAllButton = byId("run-all");
  const modeNote = byId("mode-note");
  const radios = [...document.querySelectorAll('input[name="firewall"]')];

  let busy = false;
  let presets = [];

  const firewallOn = () => radios.find((radio) => radio.checked)?.value !== "off";

  function syncMode() {
    const on = firewallOn();
    panel.classList.toggle("is-unprotected", !on);
    modeNote.textContent = on ? MODE_NOTES.on : MODE_NOTES.off;
  }

  function setBusy(state) {
    busy = state;
    sendButton.disabled = state;
    runAllButton.disabled = state;
    for (const button of document.querySelectorAll(".preset")) button.disabled = state;
    sendButton.textContent = state ? "Sending" : "Send";
  }

  function append(node) {
    transcript.append(node);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function userMessage(text) {
    const hidden = (text.match(INVISIBLE) ?? []).length;
    return h(
      "li",
      { class: "msg msg-user" },
      h("div", { class: "msg-body" }, text),
      hidden
        ? h("div", { class: "msg-meta" }, `contains ${plural(hidden, "invisible character")}`)
        : null,
    );
  }

  function verdictLine(security) {
    const risk = `risk ${security.risk_score}`;
    if (!security.enforced) {
      if (security.decision === "ALLOW") {
        return [h("span", { class: "badge badge-off" }, "firewall off"), risk];
      }
      const action = security.decision === "BLOCK" ? "blocked" : "flagged";
      return [
        h("span", { class: "badge badge-off" }, "firewall off"),
        h(
          "span",
          { class: "leak-note" },
          `Delivered unfiltered. The firewall would have ${action} this (${risk}).`,
        ),
      ];
    }
    const masked = Object.values(security.redactions).reduce((sum, count) => sum + count, 0);
    let detail = risk;
    if (security.blocked_at === "input") detail = `${risk}, stopped before the model`;
    else if (security.blocked_at === "output") detail = `${risk}, response withheld by the output guard`;
    else if (masked) detail = `${risk}, ${plural(masked, "value")} masked in the answer`;
    else if (security.sanitized) detail = `${risk}, sent to the model inside a defensive wrapper`;
    return [badge(security.decision), detail];
  }

  function botMessage(result) {
    const security = result.security;
    const classes = ["msg", "msg-bot"];
    if (security.enforced && security.blocked_at) classes.push("is-blocked");
    if (!security.enforced && security.decision !== "ALLOW") classes.push("is-leak");
    return h(
      "li",
      { class: classes.join(" ") },
      h("div", { class: "msg-body" }, result.response),
      h("div", { class: "msg-meta" }, verdictLine(security)),
    );
  }

  function errorMessage(error) {
    const reference = error.requestId ? ` (ref: ${error.requestId})` : "";
    return h(
      "li",
      { class: "msg msg-error" },
      h("div", { class: "msg-body" }, `${error.message}${reference}`),
    );
  }

  async function send(text) {
    if (busy || !text.trim()) return;
    setBusy(true);
    append(userMessage(text));
    try {
      const result = await api.chat(text, firewallOn());
      append(botMessage(result));
      onResult(result);
    } catch (error) {
      append(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function runAll() {
    for (const preset of presets) {
      await send(preset.prompt);
    }
  }

  function renderPresets(list) {
    presets = list;
    for (const kind of ["benign", "attack"]) {
      const buttons = list
        .filter((preset) => preset.kind === kind)
        .map((preset) =>
          h(
            "button",
            {
              type: "button",
              class: `preset preset-${kind}`,
              title: `${preset.technique}. ${preset.description} Expected with the firewall on: ${preset.expected}.`,
              onclick: () => send(preset.prompt),
            },
            preset.title,
          ),
        );
      byId(`presets-${kind}`).replaceChildren(...buttons);
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value;
    input.value = "";
    send(text);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  for (const radio of radios) radio.addEventListener("change", syncMode);
  runAllButton.addEventListener("click", runAll);
  syncMode();

  return {
    renderPresets,
    clear: () => transcript.replaceChildren(),
    /** Outside demo mode the server ignores the switch, so the UI removes the choice. */
    lockFirewallOn() {
      for (const radio of radios) {
        radio.checked = radio.value === "on";
        radio.disabled = radio.value === "off";
      }
      syncMode();
      modeNote.textContent = `${MODE_NOTES.on} This server is not in demo mode, so the firewall cannot be switched off.`;
    },
    setMaxLength: (limit) => input.setAttribute("maxlength", String(limit)),
  };
}
