import type { ActivationRule, AdapterConfig, ReasoningRetention } from "../config.js";

export interface ActivationDecision {
  active: boolean;
  provider: string;
  model: string;
  mergeSystemMessages: boolean;
  stripHistoryThinking: boolean;
  stripStoredThinkingText: boolean;
  reasoningRetention: ReasoningRetention;
  recoverTrappedToolCalls: boolean;
  recoveryMaxRetries: number;
  systemInject: string[];
  autoCompact: boolean;
  compactionThreshold: number;
  compactionThresholdPct: number;
  matchedRule?: string;
}

interface ResolveInput {
  sessionID?: string;
  provider?: string;
  model?: string;
}

const ANTHROPIC_PROVIDER = /anthropic/i;

function normalizeValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function wildcardToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  return new RegExp(`^${escaped}$`, "i");
}

function providerMatches(candidates: string[], provider: string): boolean {
  if (candidates.length === 0) {
    return true;
  }

  return candidates.some((candidate) => {
    const normalized = candidate.trim();
    if (!normalized) {
      return false;
    }

    if (normalized === "*") {
      return true;
    }

    if (normalized.includes("*")) {
      return wildcardToRegex(normalized).test(provider);
    }

    return normalized.toLowerCase() === provider.toLowerCase();
  });
}

function modelMatches(patterns: string[], model: string): boolean {
  if (patterns.length === 0) {
    return true;
  }

  return patterns.some((pattern) => {
    try {
      return new RegExp(pattern, "i").test(model);
    } catch {
      return false;
    }
  });
}

function inactiveDecision(provider: string, model: string): ActivationDecision {
  return {
    active: false,
    provider,
    model,
    mergeSystemMessages: false,
    stripHistoryThinking: false,
    stripStoredThinkingText: false,
    reasoningRetention: "none",
    recoverTrappedToolCalls: false,
    recoveryMaxRetries: 0,
    systemInject: [],
    autoCompact: false,
    compactionThreshold: 0,
    compactionThresholdPct: 0,
  };
}

function decisionFromRule(
  rule: ActivationRule,
  provider: string,
  model: string,
): ActivationDecision {
  return {
    active: true,
    provider,
    model,
    mergeSystemMessages: rule.merge_system_messages,
    stripHistoryThinking: rule.strip_history_thinking,
    stripStoredThinkingText: rule.strip_stored_thinking_text,
    reasoningRetention: rule.reasoning_retention,
    recoverTrappedToolCalls: rule.recover_trapped_tool_calls,
    recoveryMaxRetries: rule.recovery_max_retries,
    systemInject: rule.system_inject,
    autoCompact: rule.auto_compact,
    compactionThreshold: rule.compaction_threshold,
    compactionThresholdPct: rule.compaction_threshold_pct,
    matchedRule: rule.name,
  };
}

export class ParamsCache {
  private readonly bySession = new Map<string, ActivationDecision>();

  private lastDecision: ActivationDecision | null = null;

  constructor(private readonly config: AdapterConfig) {}

  private evaluate(providerInput: unknown, modelInput: unknown): ActivationDecision {
    const provider = normalizeValue(providerInput);
    const model = normalizeValue(modelInput);

    if (!provider || !model || !this.config.enabled || ANTHROPIC_PROVIDER.test(provider)) {
      return inactiveDecision(provider, model);
    }

    for (const rule of this.config.rules) {
      if (!providerMatches(rule.providers, provider)) {
        continue;
      }
      if (!modelMatches(rule.model_patterns, model)) {
        continue;
      }
      return decisionFromRule(rule, provider, model);
    }

    return inactiveDecision(provider, model);
  }

  rememberFromChatParams(input: ResolveInput): ActivationDecision {
    const decision = this.evaluate(input.provider, input.model);
    this.lastDecision = decision;
    if (input.sessionID) {
      this.bySession.set(input.sessionID, decision);
    }
    return decision;
  }

  resolve(input: ResolveInput): ActivationDecision {
    if (input.sessionID && this.bySession.has(input.sessionID)) {
      return this.bySession.get(input.sessionID)!;
    }

    if (input.provider || input.model) {
      return this.rememberFromChatParams(input);
    }

    return this.lastDecision ?? inactiveDecision("", "");
  }
}
