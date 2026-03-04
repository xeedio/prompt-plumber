import { describe, expect, it } from "vitest";

import type { AdapterConfig } from "../src/config.js";
import { ParamsCache } from "../src/hooks/params-cache.js";

const baseConfig: AdapterConfig = {
  enabled: true,
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
      name: "primary",
      providers: ["vllm"],
      model_patterns: ["^qwen3"],
      merge_system_messages: true,
      strip_history_thinking: true,
      strip_stored_thinking_text: true,
      reasoning_retention: "none",
      recover_trapped_tool_calls: true,
      recovery_max_retries: 3,
      system_inject: ["close thinking before tool output"],
      auto_compact: true,
      compaction_threshold: 170000,
      compaction_threshold_pct: 0.66,
    },
    {
      name: "wildcard",
      providers: ["vllm-*"],
      model_patterns: ["coder"],
      merge_system_messages: false,
      strip_history_thinking: false,
      strip_stored_thinking_text: false,
      reasoning_retention: "all",
      recover_trapped_tool_calls: true,
      recovery_max_retries: 2,
      system_inject: ["use tool output outside think tags"],
      auto_compact: false,
      compaction_threshold: 120000,
      compaction_threshold_pct: 0.5,
    },
  ],
};

describe("ParamsCache", () => {
  it("activates when provider/model match configured rule", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.rememberFromChatParams({
      sessionID: "s1",
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    expect(decision.active).toBe(true);
    expect(decision.matchedRule).toBe("primary");
    expect(decision.reasoningRetention).toBe("none");
    expect(decision.recoverTrappedToolCalls).toBe(true);
    expect(decision.recoveryMaxRetries).toBe(3);
    expect(decision.systemInject).toEqual(["close thinking before tool output"]);
    expect(decision.autoCompact).toBe(true);
    expect(decision.compactionThreshold).toBe(170000);
    expect(decision.compactionThresholdPct).toBe(0.66);
  });

  it("skips anthropic providers entirely", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.rememberFromChatParams({
      sessionID: "s2",
      provider: "anthropic",
      model: "qwen3-coder-next",
    });

    expect(decision.active).toBe(false);
    expect(decision.matchedRule).toBeUndefined();
  });

  it("supports wildcard provider matching", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.rememberFromChatParams({
      sessionID: "s3",
      provider: "vllm-edge",
      model: "my-coder-model",
    });

    expect(decision.active).toBe(true);
    expect(decision.matchedRule).toBe("wildcard");
    expect(decision.stripHistoryThinking).toBe(false);
  });

  it("matches rules with wildcard star provider and empty model_patterns", () => {
    const cache = new ParamsCache({
      ...baseConfig,
      rules: [
        {
          ...baseConfig.rules[0],
          name: "star-provider",
          providers: ["*"],
          model_patterns: [],
        },
      ],
    });

    const decision = cache.rememberFromChatParams({
      sessionID: "s-star",
      provider: "custom-provider",
      model: "any-model",
    });

    expect(decision.active).toBe(true);
    expect(decision.matchedRule).toBe("star-provider");
  });

  it("ignores empty provider candidates and falls back to no match", () => {
    const cache = new ParamsCache({
      ...baseConfig,
      rules: [
        {
          ...baseConfig.rules[0],
          name: "empty-provider",
          providers: ["   "],
        },
      ],
    });

    const decision = cache.rememberFromChatParams({
      sessionID: "s-empty-provider",
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    expect(decision.active).toBe(false);
  });

  it("passes through rule system_inject into activation decision", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.rememberFromChatParams({
      sessionID: "s9",
      provider: "vllm-edge",
      model: "my-coder-model",
    });

    expect(decision.systemInject).toEqual(["use tool output outside think tags"]);
    expect(decision.recoverTrappedToolCalls).toBe(true);
    expect(decision.recoveryMaxRetries).toBe(2);
    expect(decision.autoCompact).toBe(false);
    expect(decision.compactionThreshold).toBe(120000);
    expect(decision.compactionThresholdPct).toBe(0.5);
  });

  it("returns cached decision by session id", () => {
    const cache = new ParamsCache(baseConfig);
    cache.rememberFromChatParams({
      sessionID: "s4",
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    const decision = cache.resolve({ sessionID: "s4" });
    expect(decision.active).toBe(true);
    expect(decision.matchedRule).toBe("primary");
  });

  it("falls back to last decision when resolve input is empty", () => {
    const cache = new ParamsCache(baseConfig);
    cache.rememberFromChatParams({
      sessionID: "s5",
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    const decision = cache.resolve({});
    expect(decision.active).toBe(true);
    expect(decision.matchedRule).toBe("primary");
  });

  it("returns inactive when provider/model are missing", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.rememberFromChatParams({ sessionID: "s6" });

    expect(decision).toMatchObject({
      active: false,
      provider: "",
      model: "",
      recoverTrappedToolCalls: false,
      recoveryMaxRetries: 0,
      systemInject: [],
      autoCompact: false,
      compactionThreshold: 0,
      compactionThresholdPct: 0,
    });
    expect(decision.matchedRule).toBeUndefined();
  });

  it("returns inactive when configuration is disabled", () => {
    const cache = new ParamsCache({
      ...baseConfig,
      enabled: false,
    });

    const decision = cache.rememberFromChatParams({
      sessionID: "s7",
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    expect(decision.active).toBe(false);
    expect(decision.matchedRule).toBeUndefined();
  });

  it("treats invalid regex patterns as non-matches", () => {
    const cache = new ParamsCache({
      ...baseConfig,
      rules: [
        {
          name: "broken-pattern",
          providers: ["vllm"],
          model_patterns: ["(["],
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
    });

    const decision = cache.rememberFromChatParams({
      sessionID: "s8",
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    expect(decision.active).toBe(false);
    expect(decision.matchedRule).toBeUndefined();
  });

  it("evaluates from resolve() provider/model input when session is not cached", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.resolve({
      provider: "vllm",
      model: "qwen3-coder-next",
    });

    expect(decision.active).toBe(true);
    expect(decision.matchedRule).toBe("primary");
    expect(cache.resolve({}).matchedRule).toBe("primary");
  });

  it("returns inactive when resolve() has no prior decision", () => {
    const cache = new ParamsCache(baseConfig);
    const decision = cache.resolve({});

    expect(decision).toMatchObject({
      active: false,
      provider: "",
      model: "",
      recoverTrappedToolCalls: false,
      recoveryMaxRetries: 0,
      systemInject: [],
      autoCompact: false,
      compactionThreshold: 0,
      compactionThresholdPct: 0,
    });
  });
});
