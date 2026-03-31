"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { ResultSummary } from "@/lib/types";
import { getResults } from "@/lib/api";
import { formatTimestamp, pct } from "@/lib/format";

export default function ResultsPage() {
  const [results, setResults] = useState<ResultSummary[]>([]);

  useEffect(() => {
    getResults().then(setResults).catch(() => {});
  }, []);

  return (
    <div>
      <h1 className="text-xl font-bold mb-4">results</h1>
      {results.length === 0 ? (
        <p className="text-gray-500 text-sm">no results yet</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm bg-white rounded-lg shadow-sm">
            <thead>
              <tr className="bg-gray-50 border-b-2 border-gray-200">
                <th className="text-left px-3 py-2 font-semibold">
                  timestamp
                </th>
                <th className="text-left px-3 py-2 font-semibold">models</th>
                <th className="text-left px-3 py-2 font-semibold">evals</th>
                <th className="text-left px-3 py-2 font-semibold">metrics</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr
                  key={r.id}
                  className="border-b border-gray-100 hover:bg-gray-50"
                >
                  <td className="px-3 py-2">
                    <Link
                      href={`/results/${r.id}`}
                      className="text-blue-600 hover:underline"
                    >
                      {formatTimestamp(r.timestamp)}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    {r.models.map((m) => m.name).join(" vs ")}
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex gap-1">
                      {r.evals.map((e) => (
                        <span
                          key={e}
                          className="text-xs px-1.5 py-0.5 bg-gray-100 rounded"
                        >
                          {e}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {r.models.map((m, i) => (
                      <div key={i}>
                        <span className="font-medium">{m.name}:</span>{" "}
                        {m.core_metric != null && (
                          <span>CORE {pct(m.core_metric)}</span>
                        )}
                        {m.core_metric != null &&
                          m.chatcore_metric != null &&
                          " | "}
                        {m.chatcore_metric != null && (
                          <span>ChatCORE {pct(m.chatcore_metric)}</span>
                        )}
                      </div>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
