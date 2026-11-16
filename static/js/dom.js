// Safe DOM helpers.
//
// The dashboard renders attacker-controlled text all day (prompts, model output,
// rule evidence). Nothing here ever parses a string as HTML: text only reaches
// the page through text nodes, so a prompt cannot become markup (stored XSS in
// the SOC tool would be an embarrassing way to lose).

/**
 * Create an element. `attrs` supports `class`, `dataset`, `on<event>` listeners
 * and plain attributes; children may be nodes, strings, numbers or nested arrays.
 */
export function h(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") {
      element.className = value;
    } else if (key === "dataset") {
      Object.assign(element.dataset, value);
    } else if (key.startsWith("on") && typeof value === "function") {
      element.addEventListener(key.slice(2), value);
    } else {
      element.setAttribute(key, value === true ? "" : String(value));
    }
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

export const byId = (id) => document.getElementById(id);

/** Replace an element's children, skipping null/false entries (replaceChildren would print "null"). */
export function fill(element, ...children) {
  const nodes = children
    .flat(Infinity)
    .filter((child) => child !== null && child !== undefined && child !== false)
    .map((child) => (child instanceof Node ? child : document.createTextNode(String(child))));
  element.replaceChildren(...nodes);
}

export function badge(decision) {
  const key = decision.toLowerCase();
  return h("span", { class: `badge badge-${key}` }, decision);
}

export function plural(count, singular, pluralForm = `${singular}s`) {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

const ACRONYMS = { pii: "PII", ml: "ML" };

/** "system_prompt_leak" -> "System prompt leak" (or all lower-case with `capitalize: false`). */
export function humanize(identifier, { capitalize = true } = {}) {
  const text = identifier
    .split("_")
    .map((word) => ACRONYMS[word] ?? word)
    .join(" ");
  return capitalize ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

export function formatMs(value) {
  return value >= 100 ? `${Math.round(value)} ms` : `${value.toFixed(1)} ms`;
}
