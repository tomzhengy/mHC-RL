import type { ModelResult } from "@/lib/types";
import { pct, formatParams, formatDuration } from "@/lib/format";

const MODEL_COLORS = [
  "#1565c0",
  "#e65100",
  "#2e7d32",
  "#6a1b9a",
  "#c62828",
  "#00838f",
];

export function ResultsSummaryCards({ models }: { models: ModelResult[] }) {
  return (
    <div className="flex gap-4 flex-wrap mb-6">
      {models.map((m, i) => {
        const color = MODEL_COLORS[i % MODEL_COLORS.length];
        let totalTime = 0;
        if (m.core) totalTime += m.core.wall_time;
        if (m.chat) totalTime += m.chat.wall_time;

        return (
          <div
            key={m.model_spec}
            className="bg-white rounded-lg p-4 shadow-sm flex-1 min-w-[200px]"
            style={{ borderTop: `4px solid ${color}` }}
          >
            <div className="font-bold text-sm">{m.model_name}</div>
            <div className="text-xs text-gray-500 mb-2">
              {formatParams(m.param_count)} params
            </div>
            {m.core && (
              <div className="text-lg font-semibold">
                CORE: {pct(m.core.core_metric)}
              </div>
            )}
            {m.chat && (
              <div className="text-lg font-semibold">
                ChatCORE:{" "}
                {m.chat.chatcore_metric != null
                  ? pct(m.chat.chatcore_metric)
                  : "n/a"}
              </div>
            )}
            <div className="text-xs text-gray-500 mt-1">
              eval time: {formatDuration(totalTime)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
