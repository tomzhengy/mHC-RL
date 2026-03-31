"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { EvalResults } from "@/lib/types";
import { getResult } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";
import { ResultsSummaryCards } from "@/components/results-summary-cards";
import { CoreTable } from "@/components/core-table";
import { ChatTable } from "@/components/chat-table";
import { ThroughputTable } from "@/components/throughput-table";
import { EvalBarChart } from "@/components/bar-chart";

export default function ResultDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const [data, setData] = useState<EvalResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getResult(id)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [id]);

  if (error) {
    return (
      <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
        {error}
      </div>
    );
  }

  if (!data) {
    return <p className="text-gray-500 text-sm">loading...</p>;
  }

  const gpu = data.gpu_info.gpu_vram_gb
    ? `${data.gpu_info.gpu_name} (${data.gpu_info.gpu_vram_gb}GB)`
    : data.gpu_info.gpu_name;

  return (
    <div>
      <h1 className="text-xl font-bold mb-1">eval comparison</h1>
      <div className="text-xs text-gray-500 mb-4">
        {formatTimestamp(data.timestamp)} | {gpu} | evals:{" "}
        {data.eval_config.evals.join(", ")}
      </div>

      <ResultsSummaryCards models={data.models} />
      <CoreTable models={data.models} />
      <ChatTable models={data.models} />
      <ThroughputTable models={data.models} />
      <EvalBarChart models={data.models} type="core" />
      <EvalBarChart models={data.models} type="chat" />
    </div>
  );
}
