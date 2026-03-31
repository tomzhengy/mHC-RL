import type { ModelResult } from "@/lib/types";
import { pct, diffPp } from "@/lib/format";

const CHAT_TASKS = [
  "ARC-Easy",
  "ARC-Challenge",
  "MMLU",
  "GSM8K",
  "HumanEval",
  "SpellingBee",
];

export function ChatTable({ models }: { models: ModelResult[] }) {
  const withChat = models.filter((m) => m.chat);
  if (withChat.length === 0) return null;

  const showDiff = withChat.length >= 2;

  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold mb-2 pb-1 border-b-2 border-gray-200">
        ChatCORE benchmark
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm bg-white rounded-lg shadow-sm">
          <thead>
            <tr className="bg-gray-50 border-b-2 border-gray-200">
              <th className="text-left px-3 py-2 font-semibold">task</th>
              {withChat.map((m) => (
                <th key={m.model_spec} className="text-left px-3 py-2 font-semibold">
                  {m.model_name}
                </th>
              ))}
              {showDiff && (
                <th className="text-center px-3 py-2 font-semibold">diff</th>
              )}
            </tr>
          </thead>
          <tbody>
            {CHAT_TASKS.map((task) => {
              const vals = withChat.map(
                (m) => m.chat!.results[task] ?? null
              );
              return (
                <tr key={task} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-3 py-1.5">{task}</td>
                  {withChat.map((m) => (
                    <td key={m.model_spec} className="px-3 py-1.5">
                      {pct(m.chat!.results[task])}
                    </td>
                  ))}
                  {showDiff && (
                    <td className="px-3 py-1.5 text-center">
                      <DiffCell a={vals[0]} b={vals[vals.length - 1]} />
                    </td>
                  )}
                </tr>
              );
            })}
            <tr className="bg-gray-50 border-t-2 border-gray-200 font-semibold">
              <td className="px-3 py-2">ChatCORE metric</td>
              {withChat.map((m) => (
                <td key={m.model_spec} className="px-3 py-2">
                  {m.chat!.chatcore_metric != null
                    ? pct(m.chat!.chatcore_metric)
                    : "n/a"}
                </td>
              ))}
              {showDiff && (
                <td className="px-3 py-2 text-center">
                  <DiffCell
                    a={withChat[0].chat!.chatcore_metric}
                    b={withChat[withChat.length - 1].chat!.chatcore_metric}
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
