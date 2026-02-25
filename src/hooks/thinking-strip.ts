import type { ReasoningRetention } from "../config.js";

export interface MessagePart {
  type?: string;
  text?: string;
  content?: string;
  [key: string]: unknown;
}

export interface ChatMessage {
  info?: {
    role?: string;
    [key: string]: unknown;
  };
  parts?: MessagePart[];
  [key: string]: unknown;
}

const THINK_BLOCK_PATTERN = /<think\b[^>]*>[\s\S]*?<\/think>/gi;
const THINK_TAG_PATTERN = /<\/?think\b[^>]*>/gi;

export function stripThinkingText(input: string): string {
  return input.replace(THINK_BLOCK_PATTERN, "").replace(THINK_TAG_PATTERN, "");
}

function isAssistantMessage(message: { info?: { role?: string } }): boolean {
  const role = typeof message.info?.role === "string" ? message.info.role : "";
  return role.toLowerCase() === "assistant";
}

function isReasoningPart(part: MessagePart): boolean {
  const type = typeof part.type === "string" ? part.type.toLowerCase() : "";
  return type === "reasoning" || type === "thinking";
}

function sanitizePart(part: MessagePart): MessagePart {
  const nextPart: MessagePart = { ...part };
  if (typeof nextPart.text === "string") {
    nextPart.text = stripThinkingText(nextPart.text);
  }
  if (typeof nextPart.content === "string") {
    nextPart.content = stripThinkingText(nextPart.content);
  }
  return nextPart;
}

function latestAssistantIndex<T extends { info?: { role?: string } }>(messages: T[]): number {
  for (let idx = messages.length - 1; idx >= 0; idx -= 1) {
    if (isAssistantMessage(messages[idx])) {
      return idx;
    }
  }
  return -1;
}

export function stripAssistantHistoryThinking<
  T extends { info?: { role?: string }; parts?: MessagePart[] },
>(
  messages: T[],
  retention: ReasoningRetention,
): T[] {
  if (retention === "all") {
    return messages;
  }

  const keepReasoningOnIndex =
    retention === "last-message" ? latestAssistantIndex(messages) : -1;

  return messages.map((message, index) => {
    if (!isAssistantMessage(message) || !Array.isArray(message.parts)) {
      return message;
    }

    const nextParts = message.parts
      .map(sanitizePart)
      .filter((part) => {
        if (!isReasoningPart(part)) {
          return true;
        }
        return index === keepReasoningOnIndex;
      });

    return {
      ...message,
      parts: nextParts,
    };
  });
}
