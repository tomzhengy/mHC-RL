"use client";

import { useEffect, useState } from "react";
import type { ModelsConfig } from "@/lib/types";
import { getModels } from "@/lib/api";
import { RunConfigForm } from "@/components/run-config-form";

export default function Home() {
  const [config, setConfig] = useState<ModelsConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModels()
      .then(setConfig)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <h1 className="text-xl font-bold mb-4">run evaluation</h1>
      {error && (
        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm mb-4">
          {error}
        </div>
      )}
      {config ? (
        <RunConfigForm config={config} />
      ) : (
        !error && <p className="text-gray-500">loading models...</p>
      )}
    </div>
  );
}
