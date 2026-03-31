import { NextResponse } from "next/server";
import fs from "fs";
import { jobMetaPath, jobStatusPath } from "@/lib/paths";
import { markJobAsFailed } from "@/lib/job-runner";
import type { JobMeta, JobStatus } from "@/lib/types";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    const meta: JobMeta = JSON.parse(
      fs.readFileSync(jobMetaPath(id), "utf-8")
    );
    const status: JobStatus = JSON.parse(
      fs.readFileSync(jobStatusPath(id), "utf-8")
    );
    return NextResponse.json({ meta, status });
  } catch {
    return NextResponse.json({ error: "job not found" }, { status: 404 });
  }
}

// DELETE = mark as failed
export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  markJobAsFailed(id);
  return NextResponse.json({ ok: true });
}
