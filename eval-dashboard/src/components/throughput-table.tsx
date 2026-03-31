import type { ModelResult } from "@/lib/types";
import { formatDuration, formatParams } from "@/lib/format";

export function ThroughputTable({ models }: { models: ModelResult[] }) {
  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold mb-2 pb-1 border-b-2 border-gray-200">
        throughput / timing
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm bg-white rounded-lg shadow-sm">
          <thead>
            <tr className="bg-gray-50 border-b-2 border-gray-200">
              <th className="text-left px-3 py-2 font-semibold">model</th>
              <th className="text-left px-3 py-2 font-semibold">params</th>
              <th className="text-left px-3 py-2 font-semibold">CORE time</th>
              <th className="text-left px-3 py-2 font-semibold">ChatCORE time</th>
              <th className="text-left px-3 py-2 font-semibold">total</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              let total = 0;
              if (m.core) total += m.core.wall_time;
              if (m.chat) total += m.chat.wall_time;

              return (
                <tr key={m.model_spec} className="border-b border-gray-100">
                  <td className="px-3 py-1.5">{m.model_name}</td>
                  <td className="px-3 py-1.5">
                    {formatParams(m.param_count)}
                  </td>
                  <td className="px-3 py-1.5">
                    {m.core ? formatDuration(m.core.wall_time) : "-"}
                  </td>
                  <td className="px-3 py-1.5">
                    {m.chat ? formatDuration(m.chat.wall_time) : "-"}
                  </td>
                  <td className="px-3 py-1.5 font-semibold">
                    {formatDuration(total)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
