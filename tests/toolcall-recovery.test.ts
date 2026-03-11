import { describe, expect, it } from "vitest";

import {
  RECOVERY_MESSAGE,
  canRetry,
  createRecoveryTracker,
  hasTrappedToolCall,
  recordAttempt,
} from "../src/hooks/toolcall-recovery.js";

describe("toolcall recovery helpers", () => {
  it("detects trapped tool_call XML in reasoning parts without tool parts", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "reasoning",
        text: "plan <tool_call>{\"name\":\"bash\"}</tool_call> continue",
      },
      {
        type: "text",
        text: "visible",
      },
    ]);

    expect(trapped).toBe(true);
  });

  it("detects trapped function= XML in reasoning parts", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "reasoning",
        text: "plan <function=read><parameter=filePath>/foo</parameter></function> continue",
      },
    ]);

    expect(trapped).toBe(true);
  });

  it("detects unclosed function= XML in reasoning parts", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "reasoning",
        text: "plan <function=read><parameter=filePath>/foo</parameter> continue",
      },
    ]);

    expect(trapped).toBe(true);
  });

  it("detects function= in thinking parts", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "thinking",
        text: "<function=Glob><parameter=pattern>*.ts</parameter></function>",
      },
    ]);

    expect(trapped).toBe(true);
  });

  it("returns false when no reasoning part contains tool_call XML", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "reasoning",
        text: "internal notes only",
      },
      {
        type: "text",
        text: "final response",
      },
    ]);

    expect(trapped).toBe(false);
  });

  it("returns false when a real tool part exists even if reasoning contains tool_call XML", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "thinking",
        text: "step <tool_call><name>bash</name></tool_call>",
      },
      {
        type: "tool",
      },
    ]);

    expect(trapped).toBe(false);
  });

  it("returns false for function= when real tool part exists", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "reasoning",
        text: "<function=read>...</function>",
      },
      {
        type: "tool",
      },
    ]);

    expect(trapped).toBe(false);
  });

  it("detects mixed tool_call and function= patterns", () => {
    const trapped = hasTrappedToolCall([
      {
        type: "reasoning",
        text: "<tool_call><name>bash</name></tool_call><function=read><parameter=filePath>/foo</parameter></function>",
      },
    ]);

    expect(trapped).toBe(true);
  });

  it("handles parts with non-string type values safely", () => {
    const trapped = hasTrappedToolCall([
      {
        type: undefined,
        text: "<tool_call><name>bash</name></tool_call>",
      },
    ]);

    expect(trapped).toBe(false);
  });

  it("allows retries under the max and blocks retries at the max", () => {
    const tracker = createRecoveryTracker(2);

    expect(canRetry(tracker, "s1")).toBe(true);
    recordAttempt(tracker, "s1");
    expect(canRetry(tracker, "s1")).toBe(true);
    recordAttempt(tracker, "s1");
    expect(canRetry(tracker, "s1")).toBe(false);
  });

  it("increments retry attempts for a session", () => {
    const tracker = createRecoveryTracker(3);

    recordAttempt(tracker, "s2");
    recordAttempt(tracker, "s2");

    expect(tracker.attempts.get("s2")).toBe(2);
  });

  it("provides a non-empty recovery message", () => {
    expect(typeof RECOVERY_MESSAGE).toBe("string");
    expect(RECOVERY_MESSAGE.trim().length).toBeGreaterThan(0);
  });
});
