// Thin client for the LLM-Guard REST API. Every failure becomes an ApiError
// carrying the server's error envelope ({ error: { code, message, request_id } }).

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(status, code, message, requestId = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...options,
      headers: { Accept: "application/json", ...(options.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "network", "Cannot reach the server. Check that it is still running.");
  }

  if (response.status === 204) return null;

  let body = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON body: handled below through the status code.
  }

  if (!response.ok) {
    const error = body?.error ?? {};
    throw new ApiError(
      response.status,
      error.code ?? "error",
      error.message ?? `The server answered with status ${response.status}.`,
      error.request_id ?? null,
    );
  }
  return body;
}

export const api = {
  health: () => request("/health"),
  presets: () => request("/presets"),
  telemetry: () => request("/telemetry"),
  resetTelemetry: () => request("/telemetry", { method: "DELETE" }),
  chat: (message, firewallEnabled) =>
    request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, firewall_enabled: firewallEnabled }),
    }),
};
