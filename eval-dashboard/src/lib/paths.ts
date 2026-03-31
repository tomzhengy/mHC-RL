import path from "path";

// process.cwd() returns the next.js project root (eval-dashboard/)
const EVAL_DASHBOARD_DIR = process.cwd();

export const NANOCHAT_DIR = path.resolve(EVAL_DASHBOARD_DIR, "../nanochat-mHC");
export const MODAL_EVAL_PATH = path.join(NANOCHAT_DIR, "modal_eval.py");
export const MODELS_CONFIG_PATH = path.join(EVAL_DASHBOARD_DIR, "models.json");
export const DATA_DIR = path.join(EVAL_DASHBOARD_DIR, "data");
export const JOBS_DIR = path.join(DATA_DIR, "jobs");
export const RESULTS_DIR = path.join(DATA_DIR, "results");
export const LOCK_DIR = path.join(JOBS_DIR, ".run.lock");

export function jobDir(id: string): string {
  return path.join(JOBS_DIR, id);
}

export function jobMetaPath(id: string): string {
  return path.join(JOBS_DIR, id, "meta.json");
}

export function jobStatusPath(id: string): string {
  return path.join(JOBS_DIR, id, "status.json");
}

export function jobLogPath(id: string): string {
  return path.join(JOBS_DIR, id, "output.log");
}

export function jobScriptPath(id: string): string {
  return path.join(JOBS_DIR, id, "run.sh");
}

export function resultPath(id: string): string {
  return path.join(RESULTS_DIR, `${id}.json`);
}
