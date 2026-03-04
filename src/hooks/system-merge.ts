export function mergeSystemMessages(systemMessages: string[]): string[] {
  const normalized = systemMessages
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0);

  const deduped: string[] = [];
  const seen = new Set<string>();
  for (const entry of normalized) {
    if (seen.has(entry)) {
      continue;
    }
    seen.add(entry);
    deduped.push(entry);
  }

  if (deduped.length <= 1) {
    return deduped;
  }

  return [deduped.join("\n\n")];
}
