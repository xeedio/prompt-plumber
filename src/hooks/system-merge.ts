export function mergeSystemMessages(systemMessages: string[]): string[] {
  const normalized = systemMessages
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0);

  if (normalized.length <= 1) {
    return normalized;
  }

  return [normalized.join("\n\n")];
}
