import { NextResponse } from "next/server";
import fs from "fs";
import { jobLogPath } from "@/lib/paths";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const logPath = jobLogPath(id);

  try {
    const content = fs.readFileSync(logPath, "utf-8");
    return new NextResponse(content, {
      headers: { "Content-Type": "text/plain" },
    });
  } catch {
    return new NextResponse("", {
      headers: { "Content-Type": "text/plain" },
    });
  }
}
