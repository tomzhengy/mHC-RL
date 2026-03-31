"use client";

import type { ModelEntry } from "@/lib/types";

interface Props {
  models: ModelEntry[];
  selected: string[];
  onChange: (specs: string[]) => void;
}

export function ModelSelector({ models, selected, onChange }: Props) {
  function toggle(spec: string) {
    if (selected.includes(spec)) {
      onChange(selected.filter((s) => s !== spec));
    } else {
      onChange([...selected, spec]);
    }
  }

  return (
    <div>
      <label className="text-sm font-semibold mb-2 block">models</label>
      <div className="space-y-2">
        {models.map((m) => {
          const checked = selected.includes(m.spec);
          const isHf = m.spec.startsWith("hf:");
          return (
            <label
              key={m.id}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                checked
                  ? "bg-blue-50 border-blue-300"
                  : "bg-white border-gray-200 hover:border-gray-300"
              }`}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggle(m.spec)}
                className="mt-0.5"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{m.name}</span>
                  {!isHf && m.hf_ckpt_repo && (
                    <span className="text-[10px] px-1.5 py-0.5 bg-amber-100 text-amber-800 rounded font-medium">
                      checkpoint
                    </span>
                  )}
                  {isHf && (
                    <span className="text-[10px] px-1.5 py-0.5 bg-purple-100 text-purple-800 rounded font-medium">
                      HF
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500 truncate">
                  {m.spec}
                </div>
                {m.description && (
                  <div className="text-xs text-gray-400 mt-0.5">
                    {m.description}
                  </div>
                )}
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}
