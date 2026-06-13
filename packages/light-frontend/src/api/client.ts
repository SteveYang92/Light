const BASE = "/api";

export async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("/api") ? path : `${BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    try {
      const body = JSON.parse(text);
      throw new Error(body.detail || text);
    } catch (e) {
      if (e instanceof Error && e.message !== text) throw e;
      throw new Error(text);
    }
  }
  return res.json();
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  return fetchJSON<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function parseSSE(raw: string): Record<string, unknown> | null {
  if (!raw || raw.startsWith(":")) return null;
  const parts = raw.replace(/^data: /, "").trim().split("|");
  if (parts.length < 3) return null;
  const event: Record<string, unknown> = {
    stage: parts[0],
    progress: parseFloat(parts[1]),
    message: parts[2],
  };
  if (parts.length >= 5) {
    event.chunk = parseInt(parts[3]);
    event.total_chunks = parseInt(parts[4]);
  }
  return event;
}

export function listenSSE(
  videoId: string,
  onEvent: (e: Record<string, unknown>) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): () => void {
  const url = `${BASE}/videos/${videoId}/pipeline/events`;
  const es = new EventSource(url);

  es.onmessage = (msg) => {
    const event = parseSSE(msg.data as string);
    if (!event) return;
    onEvent(event);
    if (event.stage === "done") {
      es.close();
      onDone();
    }
  };

  es.onerror = () => {
    es.close();
    onError(new Error("SSE connection lost"));
  };

  return () => es.close();
}
