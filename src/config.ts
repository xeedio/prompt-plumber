import { access, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";

import { parse, printParseErrorCode } from "jsonc-parser";

export type ReasoningRetention = "none" | "last-message" | "all";

export interface AdapterDefaults {
  merge_system_messages: boolean;
  strip_history_thinking: boolean;
  strip_stored_thinking_text: boolean;
  reasoning_retention: ReasoningRetention;
  system_inject: string[];
}

export interface ActivationRule {
  name: string;
  providers: string[];
  model_patterns: string[];
  merge_system_messages: boolean;
  strip_history_thinking: boolean;
  strip_stored_thinking_text: boolean;
  reasoning_retention: ReasoningRetention;
  system_inject: string[];
}

export interface AdapterConfig {
  enabled: boolean;
  defaults: AdapterDefaults;
  rules: ActivationRule[];
}

export type PartialAdapterConfig = Partial<
  Omit<AdapterConfig, "defaults" | "rules"> & {
    defaults: Partial<AdapterDefaults>;
    rules: Array<Partial<ActivationRule>>;
  }
>;

export const DEFAULT_CONFIG: AdapterConfig = {
  enabled: true,
  defaults: {
    merge_system_messages: true,
    strip_history_thinking: true,
    strip_stored_thinking_text: true,
    reasoning_retention: "none",
    system_inject: [],
  },
  rules: [
    {
      name: "default-vllm-qwen3",
      providers: ["vllm"],
      model_patterns: ["^qwen3([-.]|$)", "qwen3-coder"],
      merge_system_messages: true,
      strip_history_thinking: true,
      strip_stored_thinking_text: true,
      reasoning_retention: "none",
      system_inject: [],
    },
  ],
};

export function asReasoningRetention(value: unknown, fallback: ReasoningRetention): ReasoningRetention {
  if (value === "none" || value === "last-message" || value === "all") {
    return value;
  }
  return fallback;
}

function normalizeStringArray(value: unknown): string[] {
  if (typeof value === "string") {
    return [value];
  }

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((entry): entry is string => typeof entry === "string");
}

export function normalizeRule(rule: Partial<ActivationRule>, defaults: AdapterDefaults): ActivationRule {
  const systemInject = normalizeStringArray(
    (rule as Partial<ActivationRule> & { system_inject?: unknown }).system_inject,
  );

  return {
    name: rule.name ?? "unnamed-rule",
    providers: Array.isArray(rule.providers)
      ? rule.providers.filter((entry): entry is string => typeof entry === "string")
      : [],
    model_patterns: Array.isArray(rule.model_patterns)
      ? rule.model_patterns.filter((entry): entry is string => typeof entry === "string")
      : [],
    merge_system_messages: rule.merge_system_messages ?? defaults.merge_system_messages,
    strip_history_thinking: rule.strip_history_thinking ?? defaults.strip_history_thinking,
    strip_stored_thinking_text:
      rule.strip_stored_thinking_text ?? defaults.strip_stored_thinking_text,
    reasoning_retention: asReasoningRetention(rule.reasoning_retention, defaults.reasoning_retention),
    system_inject: systemInject,
  };
}

export function mergeConfig(base: AdapterConfig, override: PartialAdapterConfig): AdapterConfig {
  const hasSystemInjectOverride =
    override.defaults !== undefined &&
    Object.prototype.hasOwnProperty.call(override.defaults, "system_inject");

  const defaults: AdapterDefaults = {
    ...base.defaults,
    ...(override.defaults ?? {}),
    reasoning_retention: asReasoningRetention(
      override.defaults?.reasoning_retention,
      base.defaults.reasoning_retention,
    ),
    system_inject: hasSystemInjectOverride
      ? normalizeStringArray(
          (override.defaults as Partial<AdapterDefaults> & { system_inject?: unknown })
            .system_inject,
        )
      : normalizeStringArray(base.defaults.system_inject),
  };

  const rawRules = Array.isArray(override.rules) ? override.rules : base.rules;
  const normalizedRules = rawRules.map((rule) => normalizeRule(rule, defaults));

  return {
    enabled: override.enabled ?? base.enabled,
    defaults,
    rules: normalizedRules,
  };
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function loadJsonc(filePath: string): Promise<PartialAdapterConfig | null> {
  if (!(await fileExists(filePath))) {
    return null;
  }

  const raw = await readFile(filePath, "utf8");
  const errors: { error: number; offset: number; length: number }[] = [];
  const parsed = parse(raw, errors, { allowTrailingComma: true, disallowComments: false }) as
    | PartialAdapterConfig
    | undefined;

  if (errors.length > 0) {
    const detail = errors
      .map((entry) => `${printParseErrorCode(entry.error)}@${entry.offset}`)
      .join(", ");
    throw new Error(`Failed to parse JSONC config ${filePath}: ${detail}`);
  }

  return parsed ?? null;
}

export async function loadAdapterConfig(projectDirectory?: string): Promise<AdapterConfig> {
  let config = DEFAULT_CONFIG;

  const globalConfigPath = path.join(homedir(), ".config", "opencode", "vllm-adapter.jsonc");
  const globalConfig = await loadJsonc(globalConfigPath);
  if (globalConfig) {
    config = mergeConfig(config, globalConfig);
  }

  if (projectDirectory) {
    const projectConfigPath = path.join(projectDirectory, ".opencode", "vllm-adapter.jsonc");
    const projectConfig = await loadJsonc(projectConfigPath);
    if (projectConfig) {
      config = mergeConfig(config, projectConfig);
    }
  }

  return {
    ...config,
    rules: config.rules.map((rule) => normalizeRule(rule, config.defaults)),
    defaults: {
      ...config.defaults,
      reasoning_retention: asReasoningRetention(
        config.defaults.reasoning_retention,
        DEFAULT_CONFIG.defaults.reasoning_retention,
      ),
      system_inject: normalizeStringArray(config.defaults.system_inject),
    },
  };
}
