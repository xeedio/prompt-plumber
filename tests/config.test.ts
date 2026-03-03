import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  DEFAULT_CONFIG,
  loadAdapterConfig,
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
  system_inject: [],
};

describe("config defaults and normalization", () => {
  it("ships with expected default config values", () => {
    expect(DEFAULT_CONFIG.enabled).toBe(true);
    expect(DEFAULT_CONFIG.defaults).toEqual(defaults);
    expect(DEFAULT_CONFIG.rules[0]?.providers).toContain("vllm");
  });

  it("sets a non-empty default system_inject instruction on the qwen3 rule", () => {
    const rule = DEFAULT_CONFIG.rules[0];
    expect(rule?.name).toBe("default-vllm-qwen3");
    expect(rule?.system_inject.length).toBeGreaterThan(0);
    expect(rule?.system_inject[0]).toContain("</think>");
    expect(rule?.system_inject[0]).toContain("<tool_call>");
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

  it("normalizes non-array providers/model_patterns to empty arrays", () => {
    const normalized = normalizeRule(
      {
        name: "non-arrays",
        providers: "vllm" as unknown as string[],
        model_patterns: "qwen3" as unknown as string[],
      },
      defaults,
    );

    expect(normalized.providers).toEqual([]);
    expect(normalized.model_patterns).toEqual([]);
  });

  it("normalizes system_inject for string, string[] and undefined", () => {
    const fromString = normalizeRule(
      {
        name: "string-inject",
        system_inject: "close thinking before tool output" as unknown as string[],
      },
      defaults,
    );

    const fromArray = normalizeRule(
      {
        name: "array-inject",
        system_inject: ["one", 2 as unknown as string, "two"],
      },
      defaults,
    );

    const fromUndefined = normalizeRule(
      {
        name: "undefined-inject",
      },
      defaults,
    );

    expect(fromString.system_inject).toEqual(["close thinking before tool output"]);
    expect(fromArray.system_inject).toEqual(["one", "two"]);
    expect(fromUndefined.system_inject).toEqual([]);
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

describe("loadAdapterConfig", () => {
  async function withTempHome<T>(
    callback: (paths: { homeDir: string; projectDir: string }) => Promise<T>,
  ): Promise<T> {
    const homeDir = await mkdtemp(path.join(os.tmpdir(), "pp-home-"));
    const projectDir = await mkdtemp(path.join(os.tmpdir(), "pp-project-"));
    const previousHome = process.env.HOME;
    process.env.HOME = homeDir;

    try {
      return await callback({ homeDir, projectDir });
    } finally {
      if (previousHome === undefined) {
        delete process.env.HOME;
      } else {
        process.env.HOME = previousHome;
      }

      await rm(homeDir, { recursive: true, force: true });
      await rm(projectDir, { recursive: true, force: true });
    }
  }

  it("returns defaults when no global/project config files exist", async () => {
    await withTempHome(async ({ projectDir }) => {
      const loaded = await loadAdapterConfig(projectDir);
      expect(loaded).toEqual(DEFAULT_CONFIG);
    });
  });

  it("loads only global config when project directory is omitted", async () => {
    await withTempHome(async ({ homeDir }) => {
      const globalDir = path.join(homeDir, ".config", "opencode");
      await mkdir(globalDir, { recursive: true });
      await writeFile(
        path.join(globalDir, "vllm-adapter.jsonc"),
        `{
          "enabled": false
        }`,
      );

      const loaded = await loadAdapterConfig();
      expect(loaded.enabled).toBe(false);
      expect(loaded.defaults).toEqual(DEFAULT_CONFIG.defaults);
    });
  });

  it("merges global config first and then project overrides", async () => {
    await withTempHome(async ({ homeDir, projectDir }) => {
      const globalDir = path.join(homeDir, ".config", "opencode");
      await mkdir(globalDir, { recursive: true });
      await writeFile(
        path.join(globalDir, "vllm-adapter.jsonc"),
        `{
          "defaults": {
            "strip_history_thinking": false,
            "reasoning_retention": "last-message"
          },
          "rules": [
            {
              "name": "global",
              "providers": ["vllm"],
              "model_patterns": ["qwen3"]
            }
          ]
        }`,
      );

      const projectConfigDir = path.join(projectDir, ".opencode");
      await mkdir(projectConfigDir, { recursive: true });
      await writeFile(
        path.join(projectConfigDir, "vllm-adapter.jsonc"),
        `{
          "enabled": false,
          "defaults": {
            "reasoning_retention": "all"
          },
          "rules": [
            {
              "name": "project",
              "providers": ["vllm"],
              "model_patterns": ["qwen3-coder"],
              "merge_system_messages": false
            }
          ]
        }`,
      );

      const loaded = await loadAdapterConfig(projectDir);
      expect(loaded.enabled).toBe(false);
      expect(loaded.defaults.strip_history_thinking).toBe(false);
      expect(loaded.defaults.reasoning_retention).toBe("all");
      expect(loaded.rules).toEqual([
        {
          name: "project",
          providers: ["vllm"],
          model_patterns: ["qwen3-coder"],
          merge_system_messages: false,
          strip_history_thinking: false,
          strip_stored_thinking_text: true,
          reasoning_retention: "all",
          system_inject: [],
        },
      ]);
    });
  });

  it("throws a parse error when config JSONC is invalid", async () => {
    await withTempHome(async ({ homeDir, projectDir }) => {
      const globalDir = path.join(homeDir, ".config", "opencode");
      await mkdir(globalDir, { recursive: true });
      const brokenPath = path.join(globalDir, "vllm-adapter.jsonc");
      await writeFile(brokenPath, "{\n  defaults: {\n");

      await expect(loadAdapterConfig(projectDir)).rejects.toThrow(
        `Failed to parse JSONC config ${brokenPath}`,
      );
    });
  });
});
