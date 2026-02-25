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
    },
    {
      name: "wildcard",
      providers: ["vllm-*"],
      model_patterns: ["coder"],
      merge_system_messages: false,
      strip_history_thinking: false,
      strip_stored_thinking_text: false,
      reasoning_retention: "all",
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
    });
  });
});
