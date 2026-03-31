import { NextResponse } from "next/server";
import fs from "fs";
import crypto from "crypto";
import { MODELS_CONFIG_PATH } from "@/lib/paths";
import {
  listJobs,
  isAnyJobRunning,
  validateModelSelection,
  launchJob,
} from "@/lib/job-runner";
import type { JobMeta, ModelsConfig, ModelEntry } from "@/lib/types";

export async function GET() {
  const jobs = listJobs();
  return NextResponse.json(jobs);
}

export async function POST(req: Request) {
  const body = await req.json();
  const {
    models: modelSpecs,
    evals,
    gpu,
    hf_ckpt_repo,
    ckpt_model_tag,
    ckpt_source,
    max_per_task,
    max_problems,
  } = body;

  // basic validation
  if (!modelSpecs?.length) {
    return NextResponse.json(
      { error: "at least one model is required" },
      { status: 400 }
    );
  }
  if (!evals?.length) {
    return NextResponse.json(
      { error: "at least one eval type is required" },
      { status: 400 }
    );
  }

  // load model config for validation
  let config: ModelsConfig;
  try {
    config = JSON.parse(fs.readFileSync(MODELS_CONFIG_PATH, "utf-8"));
  } catch {
    return NextResponse.json(
      { error: "failed to read models.json" },
      { status: 500 }
    );
  }

  // resolve model entries from specs
  const selectedModels: ModelEntry[] = [];
  for (const spec of modelSpecs) {
    const entry = config.models.find((m) => m.spec === spec);
    if (entry) selectedModels.push(entry);
  }

  // validate model selection
  const validation = validateModelSelection(selectedModels);
  if (!validation.valid) {
    return NextResponse.json({ error: validation.error }, { status: 400 });
  }

  // check for running jobs
  if (isAnyJobRunning()) {
    return NextResponse.json(
      { error: "another eval job is already running" },
      { status: 409 }
    );
  }

  // determine checkpoint info from selected models
  const ckptModels = selectedModels.filter((m) => !m.spec.startsWith("hf:"));
  const resolvedRepo =
    hf_ckpt_repo || (ckptModels.length > 0 ? ckptModels[0].hf_ckpt_repo : null);
  const resolvedTag =
    ckpt_model_tag ||
    (ckptModels.length > 0 ? ckptModels[0].ckpt_model_tag : null);
  const resolvedSource =
    ckpt_source ||
    (ckptModels.length > 0 ? ckptModels[0].ckpt_source : "base");

  const id = crypto.randomBytes(6).toString("hex");
  const meta: JobMeta = {
    id,
    models: modelSpecs,
    evals,
    gpu: gpu || config.defaults.gpu,
    hf_ckpt_repo: resolvedRepo,
    ckpt_model_tag: resolvedTag,
    ckpt_source: resolvedSource,
    max_per_task: max_per_task ?? config.defaults.max_per_task,
    max_problems: max_problems ?? config.defaults.max_problems,
    created_at: new Date().toISOString(),
  };

  const result = launchJob(meta);
  if (!result.success) {
    return NextResponse.json({ error: result.error }, { status: 409 });
  }

  return NextResponse.json({ id }, { status: 201 });
}
