"use client";

interface Props {
  selected: string[];
  onChange: (evals: string[]) => void;
}

const EVAL_TYPES = [
  {
    id: "core",
    label: "CORE",
    description: "~22 few-shot ICL tasks (DCLM paper), ~2-5 min",
  },
  {
    id: "chat",
    label: "ChatCORE",
    description:
      "6 tasks: ARC, MMLU, GSM8K, HumanEval, SpellingBee, ~20-45 min",
  },
];

export function EvalTypeSelector({ selected, onChange }: Props) {
  function toggle(id: string) {
    if (selected.includes(id)) {
      if (selected.length > 1) {
        onChange(selected.filter((s) => s !== id));
      }
    } else {
      onChange([...selected, id]);
    }
  }

  return (
    <div>
      <label className="text-sm font-semibold mb-2 block">eval types</label>
      <div className="flex gap-3">
        {EVAL_TYPES.map((t) => {
          const active = selected.includes(t.id);
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => toggle(t.id)}
              className={`flex-1 p-3 rounded-lg border text-left transition-colors ${
                active
                  ? "bg-blue-50 border-blue-300"
                  : "bg-white border-gray-200 hover:border-gray-300"
              }`}
            >
              <div className="text-sm font-medium">{t.label}</div>
              <div className="text-xs text-gray-500 mt-0.5">
                {t.description}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
