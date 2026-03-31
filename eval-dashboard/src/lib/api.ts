import type {
  ModelsConfig,
  Job,
  EvalResults,
  ResultSummary,
} from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export async function getModels(): Promise<ModelsConfig> {
  return fetchJson("/api/models");
}

export async function getJobs(): Promise<Job[]> {
  return fetchJson("/api/jobs");
}

export async function getJob(id: string): Promise<Job> {
  return fetchJson(`/api/jobs/${id}`);
}

export async function getJobLog(id: string): Promise<string> {
  const res = await fetch(`/api/jobs/${id}/log`);
  if (!res.ok) return "";
  return res.text();
}

export async function triggerEval(body: {
  models: string[];
  evals: string[];
  gpu: string;
  hf_ckpt_repo?: string | null;
  ckpt_model_tag?: string | null;
  ckpt_source?: string;
  max_per_task?: number;
  max_problems?: number;
}): Promise<{ id: string }> {
  return fetchJson("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function markJobFailed(id: string): Promise<void> {
  await fetch(`/api/jobs/${id}`, { method: "DELETE" });
}

export async function getResults(): Promise<ResultSummary[]> {
  return fetchJson("/api/results");
}

export async function getResult(id: string): Promise<EvalResults> {
  return fetchJson(`/api/results/${id}`);
}
