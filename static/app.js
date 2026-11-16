// Entry point of the SOC console. Plain ES modules: no bundler, no framework,
// nothing fetched from a third party.

import { api } from "./js/api.js";
import { byId, h } from "./js/dom.js";
import { createPlayground } from "./js/playground.js";
import { clearReport, renderReport, setThresholds } from "./js/report.js";
import { refreshTelemetry, startTelemetry } from "./js/telemetry.js";

function renderStatus(health) {
  const chips = [
    h("li", {}, "Provider ", h("b", {}, health.provider)),
    h("li", {}, h("b", {}, health.rules_loaded), " signature rules"),
  ];
  if (health.ml_model) {
    chips.push(h("li", {}, "Classifier F1 ", h("b", {}, health.ml_model.cv_f1.toFixed(2))));
  }
  if (health.fallback_active) {
    chips.push(h("li", { class: "status-warn" }, "Ollama unreachable, using the simulator"));
  }
  if (health.demo_mode) {
    chips.push(h("li", {}, "Demo mode"));
  }
  byId("status").replaceChildren(...chips);
}

async function init() {
  const playground = createPlayground({
    onResult(result) {
      renderReport(result.security);
      refreshTelemetry();
    },
  });

  const resetButton = byId("reset-telemetry");
  resetButton.addEventListener("click", async () => {
    try {
      await api.resetTelemetry();
      playground.clear();
      clearReport();
      await refreshTelemetry();
    } catch (error) {
      resetButton.textContent = error.message;
    }
  });

  try {
    const [health, presets] = await Promise.all([api.health(), api.presets()]);
    renderStatus(health);
    setThresholds(health.flag_threshold, health.block_threshold);
    playground.setMaxLength(health.max_input_chars);
    playground.renderPresets(presets);
    if (!health.demo_mode) {
      playground.lockFirewallOn();
      resetButton.hidden = true;
    }
  } catch (error) {
    byId("status").replaceChildren(h("li", { class: "status-warn" }, error.message));
  }

  startTelemetry();
}

init();
