import { access, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";

import { parse, printParseErrorCode } from "jsonc-parser";

export type ReasoningRetention = "none" | "last-message" | "all";
export type LogLevel = "debug" | "info" | "warn" | "error";

export interface AdapterDefaults {
  merge_system_messages: boolean;
  strip_history_thinking: boolean;
  strip_stored_thinking_text: boolean;
  reasoning_retention: ReasoningRetention;
  recover_trapped_tool_calls: boolean;
  recovery_max_retries: number;
  system_inject: string[];
  auto_compact: boolean;
  compaction_threshold: number;
  compaction_threshold_pct: number;
}

export interface ActivationRule {
  name: string;
  providers: string[];
  model_patterns: string[];
  merge_system_messages: boolean;
  strip_history_thinking: boolean;
  strip_stored_thinking_text: boolean;
  reasoning_retention: ReasoningRetention;
  recover_trapped_tool_calls: boolean;
  recovery_max_retries: number;
  system_inject: string[];
  auto_compact: boolean;
  compaction_threshold: number;
  compaction_threshold_pct: number;
}

export interface AdapterConfig {
  enabled: boolean;
  log_level: LogLevel;
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
      name: "default-vllm-qwen3",
      providers: ["vllm"],
      model_patterns: ["^qwen3([-.]|$)", "qwen3-coder"],
      merge_system_messages: true,
      strip_history_thinking: true,
      strip_stored_thinking_text: true,
      reasoning_retention: "none",
      recover_trapped_tool_calls: true,
      recovery_max_retries: 3,
      system_inject: [
        "You MUST close your thinking with </think> BEFORE any tool calls, function calls, or structured output. NEVER place <tool_call>, function invocations, or action tags inside <think> blocks. Correct sequence: <think>reasoning</think> then tool calls.",
      ],
      auto_compact: true,
      compaction_threshold: 170000,
      compaction_threshold_pct: 0.66,
    },
  ],
};

export function asLogLevel(value: unknown, fallback: LogLevel): LogLevel {
  if (value === "debug" || value === "info" || value === "warn" || value === "error") {
    return value;
  }
  return fallback;
}

export function asReasoningRetention(value: unknown, fallback: ReasoningRetention): ReasoningRetention {
  if (value === "none" || value === "last-message" || value === "all") {
    return value;
  }
  return fallback;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  return fallback;
}

function asRecoveryMaxRetries(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.floor(value);
  }
  return fallback;
}

function asCompactionThreshold(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.floor(value);
  }
  return fallback;
}

function asCompactionThresholdPct(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1) {
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
    recover_trapped_tool_calls: asBoolean(
      rule.recover_trapped_tool_calls,
      defaults.recover_trapped_tool_calls,
    ),
    recovery_max_retries: asRecoveryMaxRetries(
      rule.recovery_max_retries,
      defaults.recovery_max_retries,
    ),
    system_inject: systemInject,
    auto_compact: asBoolean(rule.auto_compact, defaults.auto_compact),
    compaction_threshold: asCompactionThreshold(
      rule.compaction_threshold,
      defaults.compaction_threshold,
    ),
    compaction_threshold_pct: asCompactionThresholdPct(
      rule.compaction_threshold_pct,
      defaults.compaction_threshold_pct,
    ),
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
    recover_trapped_tool_calls: asBoolean(
      override.defaults?.recover_trapped_tool_calls,
      base.defaults.recover_trapped_tool_calls,
    ),
    recovery_max_retries: asRecoveryMaxRetries(
      override.defaults?.recovery_max_retries,
      base.defaults.recovery_max_retries,
    ),
    system_inject: hasSystemInjectOverride
      ? normalizeStringArray(
          (override.defaults as Partial<AdapterDefaults> & { system_inject?: unknown })
            .system_inject,
        )
      : normalizeStringArray(base.defaults.system_inject),
    auto_compact: asBoolean(override.defaults?.auto_compact, base.defaults.auto_compact),
    compaction_threshold: asCompactionThreshold(
      override.defaults?.compaction_threshold,
      base.defaults.compaction_threshold,
    ),
    compaction_threshold_pct: asCompactionThresholdPct(
      override.defaults?.compaction_threshold_pct,
      base.defaults.compaction_threshold_pct,
    ),
  };

  const rawRules = Array.isArray(override.rules) ? override.rules : base.rules;
  const normalizedRules = rawRules.map((rule) => normalizeRule(rule, defaults));

  return {
    enabled: override.enabled ?? base.enabled,
    log_level: asLogLevel(override.log_level, base.log_level),
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

async function loadJsoncWithLegacyFallback(
  primaryPath: string,
  legacyPath: string,
): Promise<PartialAdapterConfig | null> {
  const primary = await loadJsonc(primaryPath);
  if (primary) {
    return primary;
  }

  const legacy = await loadJsonc(legacyPath);
  if (legacy) {
    console.warn(
      `prompt-plumber: using legacy config path ${legacyPath}; migrate to ${primaryPath}`,
    );
  }

  return legacy;
}

export async function loadAdapterConfig(projectDirectory?: string): Promise<AdapterConfig> {
  let config = DEFAULT_CONFIG;

  const globalConfigPath = path.join(homedir(), ".config", "opencode", "prompt-plumber.jsonc");
  const globalLegacyConfigPath = path.join(homedir(), ".config", "opencode", "vllm-adapter.jsonc");
  const globalConfig = await loadJsoncWithLegacyFallback(globalConfigPath, globalLegacyConfigPath);
  if (globalConfig) {
    config = mergeConfig(config, globalConfig);
  }

  if (projectDirectory) {
    const projectConfigPath = path.join(projectDirectory, ".opencode", "prompt-plumber.jsonc");
    const projectLegacyConfigPath = path.join(projectDirectory, ".opencode", "vllm-adapter.jsonc");
    const projectConfig = await loadJsoncWithLegacyFallback(projectConfigPath, projectLegacyConfigPath);
    if (projectConfig) {
      config = mergeConfig(config, projectConfig);
    }
  }

  return {
    ...config,
    log_level: asLogLevel(config.log_level, DEFAULT_CONFIG.log_level),
    rules: config.rules.map((rule) => normalizeRule(rule, config.defaults)),
    defaults: {
      ...config.defaults,
      reasoning_retention: asReasoningRetention(
        config.defaults.reasoning_retention,
        DEFAULT_CONFIG.defaults.reasoning_retention,
      ),
      recover_trapped_tool_calls: asBoolean(
        config.defaults.recover_trapped_tool_calls,
        DEFAULT_CONFIG.defaults.recover_trapped_tool_calls,
      ),
      recovery_max_retries: asRecoveryMaxRetries(
        config.defaults.recovery_max_retries,
        DEFAULT_CONFIG.defaults.recovery_max_retries,
      ),
      system_inject: normalizeStringArray(config.defaults.system_inject),
      auto_compact: asBoolean(config.defaults.auto_compact, DEFAULT_CONFIG.defaults.auto_compact),
      compaction_threshold: asCompactionThreshold(
        config.defaults.compaction_threshold,
        DEFAULT_CONFIG.defaults.compaction_threshold,
      ),
      compaction_threshold_pct: asCompactionThresholdPct(
        config.defaults.compaction_threshold_pct,
        DEFAULT_CONFIG.defaults.compaction_threshold_pct,
      ),
    },
  };
}
