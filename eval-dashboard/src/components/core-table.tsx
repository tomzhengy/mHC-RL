import type { ModelResult } from "@/lib/types";
import { pct, diffPp } from "@/lib/format";

export function CoreTable({ models }: { models: ModelResult[] }) {
  const withCore = models.filter((m) => m.core);
  if (withCore.length === 0) return null;

  const tasks = Object.keys(withCore[0].core!.results);
  const showDiff = withCore.length >= 2;

  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold mb-2 pb-1 border-b-2 border-gray-200">
        CORE benchmark
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm bg-white rounded-lg shadow-sm">
          <thead>
            <tr className="bg-gray-50 border-b-2 border-gray-200">
              <th className="text-left px-3 py-2 font-semibold">task</th>
              {withCore.map((m) => (
                <th key={m.model_spec} className="text-left px-3 py-2 font-semibold">
                  {m.model_name}
                  <br />
                  <span className="text-xs font-normal text-gray-500">
                    centered
                  </span>
                </th>
              ))}
              {showDiff && (
                <th className="text-center px-3 py-2 font-semibold">diff</th>
              )}
            </tr>
          </thead>
          <tbody>
            {tasks.map((task) => {
              const centeredVals = withCore.map(
                (m) => m.core!.centered_results[task] ?? null
              );
              return (
                <tr key={task} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-3 py-1.5">{task}</td>
                  {withCore.map((m) => (
                    <td key={m.model_spec} className="px-3 py-1.5">
                      {pct(m.core!.centered_results[task])}
                    </td>
                  ))}
                  {showDiff && (
                    <td className="px-3 py-1.5 text-center">
                      <DiffCell
                        a={centeredVals[0]}
                        b={centeredVals[centeredVals.length - 1]}
                      />
                    </td>
                  )}
                </tr>
              );
            })}
            <tr className="bg-gray-50 border-t-2 border-gray-200 font-semibold">
              <td className="px-3 py-2">CORE metric</td>
              {withCore.map((m) => (
                <td key={m.model_spec} className="px-3 py-2">
                  {pct(m.core!.core_metric)}
                </td>
              ))}
              {showDiff && (
                <td className="px-3 py-2 text-center">
                  <DiffCell
                    a={withCore[0].core!.core_metric}
                    b={withCore[withCore.length - 1].core!.core_metric}
                  />
                </td>
              )}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DiffCell({ a, b }: { a: number | null; b: number | null }) {
  const { text, color } = diffPp(a, b);
  return (
    <span className="font-semibold" style={{ color }}>
      {text}
    </span>
  );
}
