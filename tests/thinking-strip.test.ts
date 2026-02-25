import { describe, expect, it } from "vitest";

import { stripAssistantHistoryThinking, stripThinkingText, type ChatMessage } from "../src/hooks/thinking-strip.js";

function baseMessages(): ChatMessage[] {
  return [
    {
      info: { role: "assistant" },
      parts: [
        { type: "text", text: "answer 1 <think>hidden 1</think> tail" },
        { type: "reasoning", text: "internal reasoning 1" },
      ],
    },
    {
      info: { role: "user" },
      parts: [{ type: "text", text: "question" }],
    },
    {
      info: { role: "assistant" },
      parts: [
        { type: "thinking", content: "internal reasoning 2" },
        { type: "text", text: "answer 2 <think>hidden 2</think>" },
      ],
    },
  ];
}

describe("stripThinkingText", () => {
  it("removes complete think blocks and orphan think tags", () => {
    expect(stripThinkingText("a<think>private</think>b<think>x")).toBe("abx");
  });
});

describe("stripAssistantHistoryThinking", () => {
  it("returns messages unchanged for all retention mode", () => {
    const messages = baseMessages();
    const result = stripAssistantHistoryThinking(messages, "all");
    expect(result).toBe(messages);
  });

  it("drops all assistant reasoning parts for none retention", () => {
    const result = stripAssistantHistoryThinking(baseMessages(), "none");
    expect(result[0].parts).toEqual([{ type: "text", text: "answer 1  tail" }]);
    expect(result[2].parts).toEqual([{ type: "text", text: "answer 2 " }]);
    expect(result[1].parts).toEqual([{ type: "text", text: "question" }]);
  });

  it("keeps reasoning only on latest assistant message for last-message retention", () => {
    const result = stripAssistantHistoryThinking(baseMessages(), "last-message");

    expect(result[0].parts).toEqual([{ type: "text", text: "answer 1  tail" }]);
    expect(result[2].parts).toEqual([
      { type: "thinking", content: "internal reasoning 2" },
      { type: "text", text: "answer 2 " },
    ]);
  });

  it("allows assistant messages to become empty after reasoning removal", () => {
    const messages: ChatMessage[] = [
      { info: { role: "assistant" }, parts: [{ type: "reasoning", text: "r" }] },
    ];

    const result = stripAssistantHistoryThinking(messages, "none");
    expect(result[0].parts).toEqual([]);
  });

  it("handles last-message retention when there are no assistant messages", () => {
    const messages: ChatMessage[] = [
      { info: { role: "user" }, parts: [{ type: "thinking", text: "u" }] },
      { info: { role: "system" }, parts: [{ type: "reasoning", text: "s" }] },
    ];

    const result = stripAssistantHistoryThinking(messages, "last-message");
    expect(result).toEqual(messages);
  });
});
