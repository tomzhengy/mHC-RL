export function pct(value: number | null | undefined, decimals = 2): string {
  if (value == null) return "n/a";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function diffPp(
  a: number | null | undefined,
  b: number | null | undefined
): { text: string; color: string } {
  if (a == null || b == null) return { text: "-", color: "#888" };
  const delta = (b - a) * 100;
  if (Math.abs(delta) < 0.01)
    return { text: "0.00pp", color: "#888" };
  const sign = delta > 0 ? "+" : "";
  const color = delta > 0 ? "#2e7d32" : "#c62828";
  return { text: `${sign}${delta.toFixed(2)}pp`, color };
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function formatTimestamp(ts: string): string {
  // parse "YYYYMMDD_HHMMSS" format
  if (ts.length === 15 && ts[8] === "_") {
    const y = ts.slice(0, 4);
    const mo = ts.slice(4, 6);
    const d = ts.slice(6, 8);
    const h = ts.slice(9, 11);
    const mi = ts.slice(11, 13);
    const s = ts.slice(13, 15);
    return `${y}-${mo}-${d} ${h}:${mi}:${s}`;
  }
  return ts;
}

export function formatParams(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(0)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(0)}K`;
  return `${count}`;
}
