import type { Run, RunsEvent } from "./types.js";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function listRuns(): Promise<Run[]> {
  return parseJson<Run[]>(await fetch("/api/runs"));
}

export async function createRun(file: File): Promise<Run> {
  const form = new FormData();
  form.append("file", file);
  return parseJson<Run>(
    await fetch("/api/runs", {
      method: "POST",
      body: form,
    }),
  );
}

export async function waitForRunsEvent(since: number, signal?: AbortSignal): Promise<RunsEvent> {
  const params = new URLSearchParams({
    since: `${since}`,
    timeout: "25",
  });
  return parseJson<RunsEvent>(
    await fetch(`/api/events?${params.toString()}`, {
      cache: "no-store",
      signal,
    }),
  );
}
