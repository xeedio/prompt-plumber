import { beforeEach, describe, expect, it, vi } from "vitest";

import type { AdapterConfig } from "../src/config.js";

const loadAdapterConfigMock = vi.hoisted(() => vi.fn());
const mockLogger = vi.hoisted(() => ({
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  flush: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../src/config.js", async () => {
  const actual = await vi.importActual<typeof import("../src/config.js")>("../src/config.js");
  return {
    ...actual,
    loadAdapterConfig: loadAdapterConfigMock,
  };
});

vi.mock("../src/hooks/logger.js", async () => {
  const actual = await vi.importActual<typeof import("../src/hooks/logger.js")>("../src/hooks/logger.js");
  return {
    ...actual,
    createHookLogger: vi.fn(() => mockLogger),
  };
});

const activeConfig: AdapterConfig = {
  enabled: true,
  log_level: "info",
  defaults: {
    merge_system_messages: true,
    strip_history_thinking: true,
    strip_stored_thinking_text: true,
    reasoning_retention: "none",
    recover_trapped_tool_calls: true,
    recovery_max_retries: 3,
    system_inject: [],
    auto_compact: true,
    compaction_threshold: 170000,
    compaction_threshold_pct: 0.66,
  },
  rules: [
    {
      name: "vllm-qwen3",
      providers: ["vllm"],
      model_patterns: ["^qwen3([-.]|$)", "qwen3-coder"],
      merge_system_messages: true,
      strip_history_thinking: true,
      strip_stored_thinking_text: true,
      reasoning_retention: "none",
      recover_trapped_tool_calls: true,
      recovery_max_retries: 3,
      system_inject: [],
      auto_compact: true,
      compaction_threshold: 170000,
      compaction_threshold_pct: 0.66,
    },
  ],
};

describe("plugin logging", () => {
  beforeEach(() => {
    loadAdapterConfigMock.mockReset();
    mockLogger.debug.mockReset();
    mockLogger.info.mockReset();
    mockLogger.warn.mockReset();
    mockLogger.error.mockReset();
    mockLogger.flush.mockReset();
    mockLogger.flush.mockResolvedValue(undefined);
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
  });

  it("logs threshold status on chat.params and warns when single-turn tokens exceed threshold", async () => {
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-overflow",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-overflow",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 900, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    expect(mockLogger.info).toHaveBeenCalledWith(
      "chat.params",
      "threshold_status",
      expect.objectContaining({
        sessionID: "s-overflow",
        data: expect.objectContaining({
          currentTokens: 0,
          thresholdTokens: 660,
        }),
      }),
    );
    expect(mockLogger.warn).toHaveBeenCalledWith(
      "event",
      "threshold_approaching",
      expect.objectContaining({
        sessionID: "s-overflow",
        data: expect.objectContaining({
          currentTokens: 900,
          thresholdTokens: 660,
        }),
      }),
    );
  });

  it("logs session compacting lifecycle at info level", async () => {
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["experimental.session.compacting"]({ sessionID: "s-compact-log" });
    await hooks["event"]({
      event: {
        type: "session.compacted",
        properties: {
          sessionID: "s-compact-log",
          beforeTokens: 1200,
          afterTokens: 480,
        },
      },
    });

    expect(mockLogger.info).toHaveBeenCalledWith("experimental.session.compacting", "session_compacting", {
      sessionID: "s-compact-log",
    });
    expect(mockLogger.info).toHaveBeenCalledWith(
      "event",
      "session_compacted",
      expect.objectContaining({
        sessionID: "s-compact-log",
        data: expect.objectContaining({
          beforeTokens: 1200,
          afterTokens: 480,
          properties: expect.objectContaining({
            sessionID: "s-compact-log",
            beforeTokens: 1200,
            afterTokens: 480,
          }),
        }),
      }),
    );
  });
});
