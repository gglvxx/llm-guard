// "Latest request" panel: verdict, risk bar, pipeline trace and per-layer findings.

import { badge, byId, fill, formatMs, h, plural } from "./dom.js";

const LAYER_TITLES = {
  normalizer: "Normaliser",
  heuristic: "Signatures and entropy",
  ml_classifier: "ML classifier",
  output_dlp: "Output guard",
};

let thresholds = { flag: 40, block: 70 };

/** Align the risk bar zones with the thresholds configured on the server. */
export function setThresholds(flag, block) {
  thresholds = { flag, block };
  byId("zone-allow").style.width = `${flag}%`;
  byId("zone-flag").style.width = `${block - flag}%`;
  const flagLabel = byId("scale-flag");
  const blockLabel = byId("scale-block");
  flagLabel.style.left = `${flag}%`;
  blockLabel.style.left = `${block}%`;
  flagLabel.textContent = `${flag} flag`;
  blockLabel.textContent = `${block} block`;
}

function levelFor(score) {
  if (score >= thresholds.block) return "is-alert";
  if (score >= thresholds.flag) return "is-warn";
  return "is-pass";
}

function decisionFor(score) {
  if (score >= thresholds.block) return "BLOCK";
  if (score >= thresholds.flag) return "FLAG";
  return "ALLOW";
}

function traceNode({ name, value, note, state }) {
  return h(
    "li",
    { class: `trace-node ${state}` },
    h("span", { class: "trace-name" }, name),
    h("span", { class: "trace-value" }, value),
    h("span", { class: "trace-note" }, note),
  );
}

function layerNode(name, layer, note) {
  if (!layer || layer.skipped) {
    return { name, value: "–", note: layer?.error ?? "not run", state: "is-unreached" };
  }
  return { name, value: Math.round(layer.score), note, state: levelFor(layer.score) };
}

function buildTrace(security) {
  const layers = Object.fromEntries(security.layers.map((layer) => [layer.name, layer]));
  const enforced = security.enforced;
  const stoppedAtInput = security.blocked_at === "input";
  const inputDecision = decisionFor(security.input_risk_score);

  const normalizer = layers.normalizer;
  const heuristic = layers.heuristic;
  const classifier = layers.ml_classifier;
  const output = layers.output_dlp;

  const policyNote = !enforced
    ? `${inputDecision.toLowerCase()}, not enforced`
    : { ALLOW: "allow", FLAG: "flag: prompt fenced", BLOCK: "block" }[inputDecision];

  let modelNote = "never called";
  if (security.llm_called) {
    modelNote = security.sanitized
      ? "got the fenced prompt"
      : enforced
        ? "got the clean prompt"
        : "got the raw prompt";
  }

  const outputAction = output?.details?.action ?? "pass";
  const masked = Number(output?.details?.redactions ?? 0);
  const outputNote = {
    pass: "nothing to mask",
    redact: enforced ? `${plural(masked, "value")} masked` : `${plural(masked, "value")} leaked`,
    withhold: enforced ? "response withheld" : "leak, not stopped",
  }[outputAction];

  const nodes = [
    layerNode(
      "Normalise",
      normalizer,
      normalizer?.findings.length
        ? plural(normalizer.findings.length, "obfuscation signal")
        : "nothing hidden",
    ),
    layerNode(
      "Signatures",
      heuristic,
      heuristic?.findings.length
        ? `${plural(heuristic.findings.length, "rule")} matched`
        : "no rule matched",
    ),
    layerNode(
      "Classifier",
      classifier,
      classifier?.details?.probability !== undefined
        ? `p = ${Number(classifier.details.probability).toFixed(2)}`
        : "",
    ),
    {
      name: "Policy",
      value: security.input_risk_score,
      note: policyNote,
      state: `${levelFor(security.input_risk_score)}${stoppedAtInput ? " is-severed" : ""}`,
    },
    {
      name: "Model",
      value: security.llm_called ? formatMs(security.llm_ms) : "–",
      note: modelNote,
      state: security.llm_called ? "is-pass" : "is-unreached",
    },
    stoppedAtInput
      ? { name: "Output guard", value: "–", note: "not reached", state: "is-unreached" }
      : layerNode("Output guard", output, outputNote),
  ];
  return nodes.map(traceNode);
}

function findingRow(finding) {
  const evidence =
    finding.evidence || !["canonical", "raw", "output"].includes(finding.view)
      ? h(
          "span",
          { class: "finding-evidence" },
          ["canonical", "raw", "output"].includes(finding.view) ? "" : `[${finding.view}] `,
          finding.evidence ?? "",
        )
      : null;
  return h(
    "li",
    { class: "finding" },
    h("span", { class: `finding-rule sev-${finding.severity}` }, finding.rule_id),
    h("span", { class: "finding-description" }, finding.description),
    h("span", { class: "finding-score" }, `+${Math.round(finding.score)}`),
    evidence,
  );
}

function buildFindings(security) {
  return security.layers
    .filter((layer) => layer.findings.length > 0)
    .map((layer) =>
      h(
        "section",
        { class: "finding-group" },
        h(
          "h4",
          {},
          LAYER_TITLES[layer.name] ?? layer.name,
          h(
            "span",
            {},
            `score ${Math.round(layer.score)}, weight ${layer.weight.toFixed(1)}, ${formatMs(layer.duration_ms)}`,
          ),
        ),
        h("ul", {}, layer.findings.map(findingRow)),
      ),
    );
}

export function renderReport(security) {
  byId("last-empty").hidden = true;
  byId("last-body").hidden = false;

  fill(
    byId("verdict"),
    badge(security.decision),
    h("span", {}, "risk ", h("strong", {}, security.risk_score), " / 100"),
    security.enforced ? null : h("span", { class: "badge badge-off" }, "not enforced"),
  );

  const bar = byId("riskbar");
  bar.className = `riskbar is-${security.decision.toLowerCase()}`;
  bar.setAttribute("aria-label", `Risk score ${security.risk_score} of 100: ${security.decision}`);
  byId("risk-fill").style.width = `${security.risk_score}%`;
  byId("risk-needle").style.left = `calc(${security.risk_score}% - 1px)`;

  byId("trace").replaceChildren(...buildTrace(security));

  fill(
    byId("timing"),
    "Firewall ",
    h("b", {}, formatMs(security.firewall_ms)),
    security.llm_called ? ", model " : ", model never called",
    security.llm_called ? h("b", {}, formatMs(security.llm_ms)) : null,
    ", total ",
    h("b", {}, formatMs(security.total_ms)),
  );

  byId("findings").replaceChildren(...buildFindings(security));
}

export function clearReport() {
  byId("last-empty").hidden = false;
  byId("last-body").hidden = true;
  byId("verdict").replaceChildren();
}
