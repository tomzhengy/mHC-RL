"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { Job } from "@/lib/types";
import { markJobFailed } from "@/lib/api";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

export function JobStatusCard({
  job,
  onRefresh,
}: {
  job: Job;
  onRefresh?: () => void;
}) {
  const [elapsed, setElapsed] = useState("");
  const isRunning =
    job.status.status === "running" || job.status.status === "pending";

  useEffect(() => {
    if (!isRunning || !job.status.started_at) return;
    const start = new Date(job.status.started_at).getTime();
    const tick = () => {
      const s = Math.floor((Date.now() - start) / 1000);
      const m = Math.floor(s / 60);
      setElapsed(`${m}m ${s % 60}s`);
    };
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [isRunning, job.status.started_at]);

  async function handleMarkFailed() {
    await markJobFailed(job.meta.id);
    onRefresh?.();
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-start justify-between mb-2">
        <div>
          <span
            className={`text-xs px-2 py-0.5 rounded font-medium ${STATUS_STYLES[job.status.status] || ""}`}
          >
            {job.status.status}
          </span>
          {isRunning && elapsed && (
            <span className="text-xs text-gray-500 ml-2">{elapsed}</span>
          )}
        </div>
        <span className="text-xs text-gray-400">{job.meta.id}</span>
      </div>

      <div className="text-sm mb-1">
        <span className="font-medium">models:</span>{" "}
        {job.meta.models.join(", ")}
      </div>
      <div className="text-sm mb-1">
        <span className="font-medium">evals:</span>{" "}
        {job.meta.evals.join(", ")}
      </div>
      <div className="text-sm text-gray-500 mb-2">
        gpu: {job.meta.gpu} | created:{" "}
        {new Date(job.meta.created_at).toLocaleString()}
      </div>

      <div className="flex gap-2 text-xs">
        {job.status.status === "completed" && job.status.result_id && (
          <Link
            href={`/results/${job.status.result_id}`}
            className="text-blue-600 hover:underline"
          >
            view results
          </Link>
        )}
        {job.status.status === "failed" && job.status.error && (
          <span className="text-red-600">{job.status.error}</span>
        )}
        <Link
          href={`/api/jobs/${job.meta.id}/log`}
          target="_blank"
          className="text-gray-500 hover:underline"
        >
          view log
        </Link>
        {isRunning && (
          <button
            onClick={handleMarkFailed}
            className="text-red-500 hover:underline ml-auto"
          >
            mark as failed
          </button>
        )}
      </div>
    </div>
  );
}
