"use client";

import { useEffect, useState, useCallback } from "react";
import type { Job } from "@/lib/types";
import { getJobs } from "@/lib/api";
import { JobStatusCard } from "@/components/job-status-card";

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);

  const refresh = useCallback(() => {
    getJobs().then(setJobs).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // auto-refresh every 3s if any job is running
  const hasRunning = jobs.some(
    (j) => j.status.status === "running" || j.status.status === "pending"
  );
  useEffect(() => {
    if (!hasRunning) return;
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [hasRunning, refresh]);

  return (
    <div>
      <h1 className="text-xl font-bold mb-4">jobs</h1>
      {jobs.length === 0 ? (
        <p className="text-gray-500 text-sm">no jobs yet</p>
      ) : (
        <div className="space-y-3">
          {jobs.map((job) => (
            <JobStatusCard key={job.meta.id} job={job} onRefresh={refresh} />
          ))}
        </div>
      )}
    </div>
  );
}
