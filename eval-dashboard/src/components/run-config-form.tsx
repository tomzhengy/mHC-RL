"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { ModelsConfig } from "@/lib/types";
import { triggerEval } from "@/lib/api";
import { ModelSelector } from "./model-selector";
import { EvalTypeSelector } from "./eval-type-selector";
import { GpuSelector } from "./gpu-selector";

interface Props {
  config: ModelsConfig;
}

export function RunConfigForm({ config }: Props) {
  const router = useRouter();
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedEvals, setSelectedEvals] = useState<string[]>(["core"]);
  const [gpu, setGpu] = useState(config.defaults.gpu);
  const [maxPerTask, setMaxPerTask] = useState(config.defaults.max_per_task);
  const [maxProblems, setMaxProblems] = useState(config.defaults.max_problems);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // build command preview
  const modelsCsv = selectedModels.join(",");
  const evalsCsv = selectedEvals.join(",");
  const preview = selectedModels.length
    ? `modal run modal_eval.py --models "${modelsCsv}" --evals "${evalsCsv}" --gpu ${gpu}`
    : "select models to see command preview";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const result = await triggerEval({
        models: selectedModels,
        evals: selectedEvals,
        gpu,
        max_per_task: maxPerTask,
        max_problems: maxProblems,
      });
      router.push(`/jobs`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <ModelSelector
        models={config.models}
        selected={selectedModels}
        onChange={setSelectedModels}
      />

      <EvalTypeSelector selected={selectedEvals} onChange={setSelectedEvals} />

      <div className="grid grid-cols-2 gap-4">
        <GpuSelector value={gpu} onChange={setGpu} />
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-gray-500 hover:text-gray-700 mt-8"
          >
            {showAdvanced ? "hide advanced" : "show advanced"}
          </button>
        </div>
      </div>

      {showAdvanced && (
        <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
          <div>
            <label className="text-xs font-medium text-gray-600 block mb-1">
              max per task (CORE)
            </label>
            <input
              type="number"
              value={maxPerTask}
              onChange={(e) => setMaxPerTask(parseInt(e.target.value) || -1)}
              className="w-full p-2 rounded border border-gray-200 text-sm"
              min={-1}
            />
            <span className="text-xs text-gray-400">-1 = all</span>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-600 block mb-1">
              max problems (ChatCORE)
            </label>
            <input
              type="number"
              value={maxProblems}
              onChange={(e) => setMaxProblems(parseInt(e.target.value) || -1)}
              className="w-full p-2 rounded border border-gray-200 text-sm"
              min={-1}
            />
            <span className="text-xs text-gray-400">-1 = all</span>
          </div>
        </div>
      )}

      {/* command preview */}
      <div className="p-3 bg-gray-900 text-gray-300 rounded-lg text-xs font-mono overflow-x-auto">
        $ {preview}
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading || selectedModels.length === 0}
        className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium text-sm hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "starting..." : "run evaluation"}
      </button>
    </form>
  );
}
