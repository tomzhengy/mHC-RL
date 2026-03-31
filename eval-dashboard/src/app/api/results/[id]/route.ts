import { NextResponse } from "next/server";
import fs from "fs";
import { resultPath } from "@/lib/paths";
import type { EvalResults } from "@/lib/types";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    const raw = fs.readFileSync(resultPath(id), "utf-8");
    const data: EvalResults = JSON.parse(raw);
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: "result not found" }, { status: 404 });
  }
}
