"use client";

const GPU_OPTIONS = ["A10G", "A100", "H100"];

interface Props {
  value: string;
  onChange: (gpu: string) => void;
}

export function GpuSelector({ value, onChange }: Props) {
  return (
    <div>
      <label className="text-sm font-semibold mb-2 block">GPU</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-2 rounded-lg border border-gray-200 bg-white text-sm"
      >
        {GPU_OPTIONS.map((g) => (
          <option key={g} value={g}>
            {g}
          </option>
        ))}
      </select>
    </div>
  );
}
