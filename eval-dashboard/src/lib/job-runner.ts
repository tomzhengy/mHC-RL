import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import type { JobMeta, JobStatus, ModelEntry } from "./types";
import {
  JOBS_DIR,
  RESULTS_DIR,
  LOCK_DIR,
  MODAL_EVAL_PATH,
  jobDir,
  jobMetaPath,
  jobStatusPath,
  jobScriptPath,
} from "./paths";

const STALE_LOCK_MS = 2 * 60 * 60 * 1000; // 2 hours

// -- locking --

function cleanStaleLock(): void {
  try {
    const stat = fs.statSync(LOCK_DIR);
    if (Date.now() - stat.mtimeMs > STALE_LOCK_MS) {
      fs.rmdirSync(LOCK_DIR);
    }
  } catch {
    // lock dir doesn't exist, nothing to clean
  }
}

function acquireLock(): boolean {
  cleanStaleLock();
  try {
    fs.mkdirSync(LOCK_DIR);
    return true;
  } catch {
    return false;
  }
}

// -- job scanning --

export function listJobs(): { meta: JobMeta; status: JobStatus }[] {
  fs.mkdirSync(JOBS_DIR, { recursive: true });
  const entries = fs.readdirSync(JOBS_DIR, { withFileTypes: true });
  const jobs: { meta: JobMeta; status: JobStatus }[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
    try {
      const meta: JobMeta = JSON.parse(
        fs.readFileSync(path.join(JOBS_DIR, entry.name, "meta.json"), "utf-8")
      );
      const status: JobStatus = JSON.parse(
        fs.readFileSync(
          path.join(JOBS_DIR, entry.name, "status.json"),
          "utf-8"
        )
      );
      jobs.push({ meta, status });
    } catch {
      // skip malformed job dirs
    }
  }

  // sort newest first
  jobs.sort(
    (a, b) =>
      new Date(b.meta.created_at).getTime() -
      new Date(a.meta.created_at).getTime()
  );
  return jobs;
}

export function isAnyJobRunning(): boolean {
  return listJobs().some((j) => j.status.status === "running");
}

// -- validation --

export function validateModelSelection(
  models: ModelEntry[]
): { valid: boolean; error?: string } {
  const ckptModels = models.filter((m) => !m.spec.startsWith("hf:"));

  // checkpoint models need hf_ckpt_repo to run on Modal
  const missingRepo = ckptModels.filter((m) => !m.hf_ckpt_repo);
  if (missingRepo.length > 0) {
    return {
      valid: false,
      error: `checkpoint models missing hf_ckpt_repo: ${missingRepo.map((m) => m.name).join(", ")}`,
    };
  }

  // all checkpoint models must share the same hf_ckpt_repo
  if (ckptModels.length > 1) {
    const repos = new Set(ckptModels.map((m) => m.hf_ckpt_repo));
    if (repos.size > 1) {
      return {
        valid: false,
        error:
          "all checkpoint models must share the same hf_ckpt_repo (Modal downloads one repo per run)",
      };
    }
    const sources = new Set(ckptModels.map((m) => m.ckpt_source));
    if (sources.size > 1) {
      return {
        valid: false,
        error:
          "all checkpoint models must share the same ckpt_source",
      };
    }
  }

  return { valid: true };
}

// -- script generation --

function generateRunScript(meta: JobMeta): string {
  const dir = jobDir(meta.id);
  const startedAt = new Date().toISOString();
  const resultId = `${meta.created_at.replace(/[-:T]/g, "").slice(0, 15)}_${meta.id}`;

  const modelsCsv = meta.models.join(",");
  const evalsCsv = meta.evals.join(",");

  let additionalFlags = "";
  if (meta.hf_ckpt_repo) {
    additionalFlags += ` \\\n  --hf-ckpt-repo "${meta.hf_ckpt_repo}"`;
  }
  if (meta.ckpt_model_tag) {
    additionalFlags += ` \\\n  --ckpt-model-tag "${meta.ckpt_model_tag}"`;
  }
  if (meta.ckpt_source && meta.ckpt_source !== "base") {
    additionalFlags += ` \\\n  --ckpt-source "${meta.ckpt_source}"`;
  }
  if (meta.max_per_task !== -1) {
    additionalFlags += ` \\\n  --max-per-task ${meta.max_per_task}`;
  }
  if (meta.max_problems !== -1) {
    additionalFlags += ` \\\n  --max-problems ${meta.max_problems}`;
  }

  return `#!/bin/bash
JOB_DIR="${dir}"
RESULTS_DIR="${RESULTS_DIR}"
MODAL_EVAL="${MODAL_EVAL_PATH}"
LOCK_DIR="${LOCK_DIR}"
RESULT_ID="${resultId}"
STARTED_AT="${startedAt}"

# redirect all output to log (no parent pipes)
exec > "$JOB_DIR/output.log" 2>&1

# update status to running
cat > "$JOB_DIR/status.json" << 'STATUSEOF'
{"status":"running","started_at":"${STARTED_AT}"}
STATUSEOF
# fix the template
sed -i '' "s/\\\${STARTED_AT}/$STARTED_AT/" "$JOB_DIR/status.json" 2>/dev/null || true

# run modal from job dir (output files land here)
cd "$JOB_DIR"
modal run "$MODAL_EVAL" \\
  --models "${modelsCsv}" \\
  --evals "${evalsCsv}" \\
  --gpu "${meta.gpu}"${additionalFlags}

EXIT_CODE=$?

COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [ $EXIT_CODE -eq 0 ] && [ -f "$JOB_DIR/eval_results.json" ]; then
  mkdir -p "$RESULTS_DIR"
  cp "$JOB_DIR/eval_results.json" "$RESULTS_DIR/$RESULT_ID.json"
  cat > "$JOB_DIR/status.json" << STATUSEOF
{"status":"completed","started_at":"$STARTED_AT","completed_at":"$COMPLETED_AT","result_id":"$RESULT_ID","exit_code":0}
STATUSEOF
else
  cat > "$JOB_DIR/status.json" << STATUSEOF
{"status":"failed","started_at":"$STARTED_AT","completed_at":"$COMPLETED_AT","exit_code":$EXIT_CODE,"error":"modal run exited with code $EXIT_CODE"}
STATUSEOF
fi

# release lock
rmdir "$LOCK_DIR" 2>/dev/null
`;
}

// -- launch --

export function launchJob(meta: JobMeta): { success: boolean; error?: string } {
  // ensure dirs exist
  const dir = jobDir(meta.id);
  fs.mkdirSync(dir, { recursive: true });
  fs.mkdirSync(RESULTS_DIR, { recursive: true });

  // try to acquire lock
  if (!acquireLock()) {
    return { success: false, error: "another job is already running" };
  }

  // write meta
  fs.writeFileSync(jobMetaPath(meta.id), JSON.stringify(meta, null, 2));

  // write initial status
  const status: JobStatus = { status: "pending" };
  fs.writeFileSync(jobStatusPath(meta.id), JSON.stringify(status));

  // write run script
  const script = generateRunScript(meta);
  const scriptPath = jobScriptPath(meta.id);
  fs.writeFileSync(scriptPath, script, { mode: 0o755 });

  // spawn detached
  const child = spawn("bash", [scriptPath], {
    detached: true,
    stdio: "ignore",
  });
  child.unref();

  return { success: true };
}

export function markJobAsFailed(id: string): void {
  const statusPath = jobStatusPath(id);
  try {
    const current: JobStatus = JSON.parse(fs.readFileSync(statusPath, "utf-8"));
    if (current.status !== "running") return;
    const updated: JobStatus = {
      ...current,
      status: "failed",
      completed_at: new Date().toISOString(),
      error: "manually marked as failed",
    };
    fs.writeFileSync(statusPath, JSON.stringify(updated, null, 2));
    // clean up lock if it exists
    try {
      fs.rmdirSync(LOCK_DIR);
    } catch {
      // lock may not exist
    }
  } catch {
    // job doesn't exist
  }
}
