import { NextResponse } from "next/server";
import fs from "fs";
import { MODELS_CONFIG_PATH } from "@/lib/paths";
import type { ModelsConfig } from "@/lib/types";

export async function GET() {
  try {
    const raw = fs.readFileSync(MODELS_CONFIG_PATH, "utf-8");
    const config: ModelsConfig = JSON.parse(raw);
    return NextResponse.json(config);
  } catch (e) {
    return NextResponse.json(
      { error: "failed to read models.json" },
      { status: 500 }
    );
  }
}
