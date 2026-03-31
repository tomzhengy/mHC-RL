import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { RESULTS_DIR } from "@/lib/paths";
import type { EvalResults, ResultSummary } from "@/lib/types";

export async function GET() {
  try {
    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith(".json"));

    const summaries: ResultSummary[] = [];
    for (const file of files) {
      try {
        const raw = fs.readFileSync(path.join(RESULTS_DIR, file), "utf-8");
        const data: EvalResults = JSON.parse(raw);
        const id = file.replace(".json", "");

        const gpuName = data.gpu_info?.gpu_name || "unknown";
        const gpuVram = data.gpu_info?.gpu_vram_gb;
        const gpu = gpuVram ? `${gpuName} (${gpuVram}GB)` : gpuName;

        summaries.push({
          id,
          timestamp: data.timestamp,
          gpu,
          evals: data.eval_config?.evals || [],
          models: data.models.map((m) => ({
            name: m.model_name,
            core_metric: m.core?.core_metric ?? null,
            chatcore_metric: m.chat?.chatcore_metric ?? null,
          })),
        });
      } catch {
        // skip malformed files
      }
    }

    // sort newest first
    summaries.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
    return NextResponse.json(summaries);
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
