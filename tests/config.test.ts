import { describe, expect, it } from "vitest";

import {
  DEFAULT_CONFIG,
  mergeConfig,
  normalizeRule,
  type AdapterDefaults,
  type PartialAdapterConfig,
} from "../src/config.js";

const defaults: AdapterDefaults = {
  merge_system_messages: true,
  strip_history_thinking: true,
  strip_stored_thinking_text: true,
  reasoning_retention: "none",
};

describe("config defaults and normalization", () => {
  it("ships with expected default config values", () => {
    expect(DEFAULT_CONFIG.enabled).toBe(true);
    expect(DEFAULT_CONFIG.defaults).toEqual(defaults);
    expect(DEFAULT_CONFIG.rules[0]?.providers).toContain("vllm");
  });

  it("normalizes rule fields and falls back to defaults", () => {
    const normalized = normalizeRule(
      {
        name: "partial",
        providers: ["vllm", 1 as unknown as string],
        model_patterns: ["qwen3", null as unknown as string],
        reasoning_retention: "invalid" as unknown as "none",
      },
      defaults,
    );

    expect(normalized.providers).toEqual(["vllm"]);
    expect(normalized.model_patterns).toEqual(["qwen3"]);
    expect(normalized.reasoning_retention).toBe("none");
    expect(normalized.merge_system_messages).toBe(true);
  });

  it("merges partial overrides and applies normalized rule defaults", () => {
    const override: PartialAdapterConfig = {
      enabled: false,
      defaults: {
        strip_history_thinking: false,
        reasoning_retention: "last-message",
      },
      rules: [
        {
          name: "custom",
          providers: ["vllm"],
          model_patterns: ["qwen3-coder"],
        },
      ],
    };

    const merged = mergeConfig(DEFAULT_CONFIG, override);

    expect(merged.enabled).toBe(false);
    expect(merged.defaults.strip_history_thinking).toBe(false);
    expect(merged.defaults.reasoning_retention).toBe("last-message");
    expect(merged.rules[0]).toMatchObject({
      name: "custom",
      providers: ["vllm"],
      model_patterns: ["qwen3-coder"],
      strip_history_thinking: false,
      reasoning_retention: "last-message",
    });
  });
});
