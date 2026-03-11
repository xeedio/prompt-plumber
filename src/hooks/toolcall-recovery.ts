const TOOL_CALL_PATTERN = /<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/i;
const FUNCTION_CALL_PATTERN = /<function=\w+\b[^>]*>/i;

export interface RecoveryTracker {
  attempts: Map<string, number>;
  maxRetries: number;
}

export function createRecoveryTracker(maxRetries: number): RecoveryTracker {
  return {
    attempts: new Map(),
    maxRetries,
  };
}

export function hasTrappedToolCall(
  parts: Array<{ type?: string; text?: string; [key: string]: unknown }>,
): boolean {
  const hasToolInReasoning = parts.some((part) => {
    const type = typeof part.type === "string" ? part.type.toLowerCase() : "";
    return (
      (type === "reasoning" || type === "thinking") &&
      typeof part.text === "string" &&
      (TOOL_CALL_PATTERN.test(part.text) || FUNCTION_CALL_PATTERN.test(part.text))
    );
  });

  const hasRealToolParts = parts.some((part) => {
    const type = typeof part.type === "string" ? part.type.toLowerCase() : "";
    return type === "tool";
  });

  return hasToolInReasoning && !hasRealToolParts;
}

export function canRetry(tracker: RecoveryTracker, sessionID: string): boolean {
  const count = tracker.attempts.get(sessionID) ?? 0;
  return count < tracker.maxRetries;
}

export function recordAttempt(tracker: RecoveryTracker, sessionID: string): void {
  const count = tracker.attempts.get(sessionID) ?? 0;
  tracker.attempts.set(sessionID, count + 1);
}

export const RECOVERY_MESSAGE =
  "Your previous response placed tool calls inside the thinking block. The tool calls were NOT executed. You MUST re-emit the exact same tool calls as real tool calls (outside of thinking). Do NOT wrap them in XML. Just call the tools normally.";
