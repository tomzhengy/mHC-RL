"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { ModelResult } from "@/lib/types";

const MODEL_COLORS = [
  "#1565c0",
  "#e65100",
  "#2e7d32",
  "#6a1b9a",
  "#c62828",
  "#00838f",
];

interface Props {
  models: ModelResult[];
  type: "core" | "chat";
}

export function EvalBarChart({ models, type }: Props) {
  const filtered =
    type === "core"
      ? models.filter((m) => m.core)
      : models.filter((m) => m.chat);

  if (filtered.length === 0) return null;

  // build data: one entry per task
  const tasks =
    type === "core"
      ? Object.keys(filtered[0].core!.centered_results)
      : Object.keys(filtered[0].chat!.results);

  const data = tasks.map((task) => {
    const entry: Record<string, string | number> = { task };
    for (const m of filtered) {
      const val =
        type === "core"
          ? m.core!.centered_results[task]
          : m.chat!.results[task];
      entry[m.model_name] = val != null ? +(val * 100).toFixed(1) : 0;
    }
    return entry;
  });

  const title = type === "core" ? "CORE task comparison" : "ChatCORE task comparison";

  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold mb-2 pb-1 border-b-2 border-gray-200">
        {title}
      </h2>
      <div className="bg-white rounded-lg shadow-sm p-4">
        <ResponsiveContainer width="100%" height={Math.max(300, tasks.length * 40)}>
          <BarChart data={data} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} unit="%" />
            <YAxis dataKey="task" type="category" width={110} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(val: number) => `${val}%`} />
            <Legend />
            {filtered.map((m, i) => (
              <Bar
                key={m.model_spec}
                dataKey={m.model_name}
                fill={MODEL_COLORS[i % MODEL_COLORS.length]}
                barSize={16}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
