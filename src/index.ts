import type { Plugin } from "@opencode-ai/plugin";

import { asLogLevel, loadAdapterConfig } from "./config.js";
import { createHookLogger } from "./hooks/logger.js";
import { ParamsCache } from "./hooks/params-cache.js";
import { mergeSystemMessages } from "./hooks/system-merge.js";
import {
  RECOVERY_MESSAGE,
  canRetry,
  createRecoveryTracker,
  hasTrappedToolCall,
  recordAttempt,
} from "./hooks/toolcall-recovery.js";
import { stripAssistantHistoryThinking, stripThinkingText } from "./hooks/thinking-strip.js";

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function providerIdFromChatParams(input: {
  provider?: { info?: { id?: string }; id?: string };
}): string {
  return asString(input.provider?.info?.id) || asString(input.provider?.id);
}

function providerIdFromModel(input: { model?: { providerID?: string; providerId?: string } }): string {
  return asString(input.model?.providerID ?? input.model?.providerId);
}

type SessionTokenState = {
  total: number;
  providerID?: string;
  modelID?: string;
  contextWindow?: number;
  thresholdTokens: number;
  thresholdPct: number;
  warnedContextUnknown: boolean;
};

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function tokenTotal(tokens: unknown): number | undefined {
  if (!tokens || typeof tokens !== "object") return undefined;
  const typed = tokens as {
    total?: unknown;
    input?: unknown;
    output?: unknown;
    reasoning?: unknown;
    cache?: {
      read?: unknown;
      write?: unknown;
    };
  };

  const total = asNumber(typed.total);
  if (total !== undefined) return total;

  const input = asNumber(typed.input) ?? 0;
  const output = asNumber(typed.output) ?? 0;
  const reasoning = asNumber(typed.reasoning) ?? 0;
  const cacheRead = asNumber(typed.cache?.read) ?? 0;
  const cacheWrite = asNumber(typed.cache?.write) ?? 0;
  const computed = input + output + reasoning + cacheRead + cacheWrite;
  return computed > 0 ? computed : undefined;
}

function percentUsed(currentTokens: number, thresholdTokens: number): number {
  if (thresholdTokens <= 0) return 1;
  return Number((currentTokens / thresholdTokens).toFixed(4));
}

function extractContextWindow(model: unknown): number | undefined {
  if (!model || typeof model !== "object") return undefined;
  const typed = model as {
    limit?: {
      context?: unknown;
    };
    context?: unknown;
  };
  const byLimit = asNumber(typed.limit?.context);
  if (byLimit !== undefined && byLimit > 0) return Math.floor(byLimit);
  const byContext = asNumber(typed.context);
  if (byContext !== undefined && byContext > 0) return Math.floor(byContext);
  return undefined;
}

function thresholdFromContext(contextWindow: number, thresholdPct: number): number {
  return Math.floor(contextWindow * thresholdPct);
}

const plugin = (async (ctx) => {
  const config = await loadAdapterConfig(ctx.directory);
  const logger = createHookLogger({
    level: asLogLevel(process.env.PROMPT_PLUMBER_LOG_LEVEL ?? config.log_level, "info"),
  });

  if (!config.enabled) {
    logger.info("plugin", "prompt-plumber disabled by configuration");
    return {};
  }

  logger.info("plugin", "prompt-plumber initialized", {
    data: {
      directory: ctx.directory,
      enabled: config.enabled,
      logLevel: process.env.PROMPT_PLUMBER_LOG_LEVEL ?? config.log_level,
    },
  });

  // Monkey-patch exploration:
  // Plugins execute in the same process, but PluginInput only exposes SDK client,
  // project/worktree paths, server URL, and shell helpers. It does not expose
  // Provider/Model internals or mutable references to OpenCode's in-memory model
  // registry. Runtime deep-import patching would be brittle and unsafe, so we avoid
  // monkey-patching model limits from this plugin.
  logger.debug("plugin", "monkey-patch exploration", {
    data: {
      canAccessProviderInternals: false,
      reason: "PluginInput does not expose internal provider/model references",
    },
  });

  const cache = new ParamsCache(config);
  const tracker = createRecoveryTracker(config.defaults.recovery_max_retries);
  const tokensBySession = new Map<string, SessionTokenState>();
  const compactingSessions = new Set<string>();

  return {
    event: async (input) => {
      const event = input.event as {
        type?: string;
        properties?: {
          sessionID?: string;
          info?: {
            sessionID?: string;
            role?: string;
            providerID?: string;
            modelID?: string;
            tokens?: unknown;
          };
          part?: {
            sessionID?: string;
            type?: string;
            tokens?: unknown;
          };
          status?: {
            type?: string;
          };
        };
      };

      if (event.type === "message.updated") {
        const info = event.properties?.info;
        if (info?.role === "assistant" && info.sessionID) {
          const total = tokenTotal(info.tokens);
          if (total !== undefined) {
            const previous = tokensBySession.get(info.sessionID);
            tokensBySession.set(info.sessionID, {
              total,
              providerID: asString(info.providerID) || previous?.providerID,
              modelID: asString(info.modelID) || previous?.modelID,
              contextWindow: previous?.contextWindow,
              thresholdTokens: previous?.thresholdTokens ?? 0,
              thresholdPct: previous?.thresholdPct ?? 0,
              warnedContextUnknown: previous?.warnedContextUnknown ?? false,
            });
            logger.debug("event", "token_update", {
              sessionID: info.sessionID,
              data: {
                currentTokens: total,
                thresholdTokens: previous?.thresholdTokens ?? 0,
                percentUsed: percentUsed(total, previous?.thresholdTokens ?? 0),
                providerID: asString(info.providerID) || previous?.providerID,
                modelID: asString(info.modelID) || previous?.modelID,
              },
            });
          }
        }
      }

      if (event.type === "message.part.updated") {
        const part = event.properties?.part;
        if (part?.sessionID && part.type === "step-finish") {
          const total = tokenTotal(part.tokens);
          if (total !== undefined) {
            const previous = tokensBySession.get(part.sessionID);
            tokensBySession.set(part.sessionID, {
              total,
              providerID: previous?.providerID,
              modelID: previous?.modelID,
              contextWindow: previous?.contextWindow,
              thresholdTokens: previous?.thresholdTokens ?? 0,
              thresholdPct: previous?.thresholdPct ?? 0,
              warnedContextUnknown: previous?.warnedContextUnknown ?? false,
            });
            logger.debug("event", "token_update", {
              sessionID: part.sessionID,
              data: {
                currentTokens: total,
                thresholdTokens: previous?.thresholdTokens ?? 0,
                percentUsed: percentUsed(total, previous?.thresholdTokens ?? 0),
              },
            });
          }
        }
      }

      if (event.type === "session.compacted" && event.properties?.sessionID) {
        compactingSessions.delete(event.properties.sessionID);
        tokensBySession.delete(event.properties.sessionID);
      }

      let sessionID: string | undefined;
      if (event.type === "session.idle" && event.properties?.sessionID) {
        sessionID = event.properties.sessionID;
      } else if (
        event.type === "session.status" &&
        event.properties?.status?.type === "idle" &&
        event.properties?.sessionID
      ) {
        sessionID = event.properties.sessionID;
      }

      if (!sessionID) {
        return;
      }

      const decision = cache.resolve({ sessionID });
      logger.debug("event", "idle event decision", {
        sessionID,
        data: {
          active: decision.active,
          matchedRule: decision.matchedRule,
          autoCompact: decision.autoCompact,
          compactionThreshold: decision.compactionThreshold,
          recoverTrappedToolCalls: decision.recoverTrappedToolCalls,
        },
      });

      if (decision.active && decision.autoCompact) {
        const tracked = tokensBySession.get(sessionID);
        const currentTokens = tracked?.total ?? 0;
        const thresholdTokens = tracked?.thresholdTokens ?? decision.compactionThreshold;
        const thresholdPct = tracked?.thresholdPct ?? decision.compactionThresholdPct;

        if (!tracked?.contextWindow) {
          if (!tracked?.warnedContextUnknown) {
            logger.warn("event", "context_unknown", {
              sessionID,
              data: {
                provider: tracked?.providerID || decision.provider,
                model: tracked?.modelID || decision.model,
                thresholdTokens,
                thresholdPct,
              },
            });
            if (tracked) {
              tracked.warnedContextUnknown = true;
              tokensBySession.set(sessionID, tracked);
            }
          }
        }

        if (currentTokens < thresholdTokens) {
          logger.debug("event", "compaction_not_needed", {
            sessionID,
            data: {
              currentTokens,
              thresholdTokens,
              percentUsed: percentUsed(currentTokens, thresholdTokens),
            },
          });
        } else {
          const providerID = tracked?.providerID || decision.provider;
          const modelID = tracked?.modelID || decision.model;
          if (providerID && modelID) {
            if (compactingSessions.has(sessionID)) {
              logger.info("event", "compaction_skipped", {
                sessionID,
                data: {
                  reason: "in_flight_guard",
                },
              });
            } else {
              compactingSessions.add(sessionID);
              logger.info("event", "compaction_triggered", {
                sessionID,
                data: {
                  currentTokens,
                  thresholdTokens,
                  percentUsed: percentUsed(currentTokens, thresholdTokens),
                  providerID,
                  modelID,
                  reason: "tokens_at_or_above_threshold",
                  method: "session.summarize",
                },
              });
              try {
                await ctx.client.session.summarize({
                  path: { id: sessionID },
                  body: { providerID, modelID },
                });
              } catch (error) {
                logger.error("event", "failed proactive compaction", {
                  sessionID,
                  data: { error: error instanceof Error ? error.message : String(error) },
                });
                compactingSessions.delete(sessionID);
              }
            }
          }
        }
      }

      if (!decision.active || !decision.recoverTrappedToolCalls) {
        return;
      }

      tracker.maxRetries = decision.recoveryMaxRetries;
      if (!canRetry(tracker, sessionID)) {
        logger.debug("event", "recovery skipped due to retry limit", {
          sessionID,
          data: { maxRetries: tracker.maxRetries },
        });
        return;
      }

      try {
        const result = await ctx.client.session.messages({ path: { id: sessionID } });
        if (!Array.isArray(result.data) || result.data.length === 0) {
          return;
        }

        const lastAssistant = [...result.data]
          .reverse()
          .find((message) => message.info?.role === "assistant");
        if (!lastAssistant || !Array.isArray(lastAssistant.parts)) {
          return;
        }

        if (!hasTrappedToolCall(lastAssistant.parts)) {
          logger.debug("event", "no trapped tool call found", { sessionID });
          return;
        }

        recordAttempt(tracker, sessionID);
        logger.info("event", "tool-call recovery triggered", {
          sessionID,
          data: { attempts: tracker.attempts.get(sessionID) },
        });
        await ctx.client.session.promptAsync({
          path: { id: sessionID },
          body: {
            parts: [{ type: "text", text: RECOVERY_MESSAGE }],
          },
        });
      } catch (error) {
        logger.error("event", "tool-call recovery failed", {
          sessionID,
          data: { error: error instanceof Error ? error.message : String(error) },
        });
      }
    },

    "chat.params": async (input, _output) => {
      const freshConfig = await loadAdapterConfig(ctx.directory);
      cache.updateConfig(freshConfig);
      logger.debug("chat.params", "config_reloaded");

      const decision = cache.rememberFromChatParams({
        sessionID: input.sessionID,
        provider: providerIdFromChatParams(input),
        model: asString(input.model?.id),
      });
      tracker.maxRetries = decision.recoveryMaxRetries;
      const existing = tokensBySession.get(input.sessionID);
      const provider = providerIdFromChatParams(input);
      const model = asString(input.model?.id);
      const contextWindow = extractContextWindow(input.model);
      const thresholdPct = decision.compactionThresholdPct;
      const thresholdTokens =
        contextWindow !== undefined
          ? thresholdFromContext(contextWindow, thresholdPct)
          : decision.compactionThreshold;
      const warnedContextUnknown = contextWindow === undefined;

      const hasPreviousContext = existing?.contextWindow !== undefined;
      if (contextWindow !== undefined && !hasPreviousContext) {
        logger.info("chat.params", "context_discovered", {
          sessionID: input.sessionID,
          data: {
            provider,
            model,
            contextWindow,
            thresholdTokens,
            thresholdPct,
          },
        });
      }

      if (existing?.contextWindow !== undefined && contextWindow !== undefined && existing.contextWindow !== contextWindow) {
        logger.info("chat.params", "context_changed", {
          sessionID: input.sessionID,
          data: {
            provider,
            model,
            oldContextWindow: existing.contextWindow,
            newContextWindow: contextWindow,
            oldThresholdTokens: existing.thresholdTokens,
            newThresholdTokens: thresholdTokens,
            thresholdPct,
          },
        });
      }

      if (existing?.contextWindow !== undefined && contextWindow === undefined) {
        logger.info("chat.params", "context_changed", {
          sessionID: input.sessionID,
          data: {
            provider,
            model,
            oldContextWindow: existing.contextWindow,
            newContextWindow: undefined,
            oldThresholdTokens: existing.thresholdTokens,
            newThresholdTokens: thresholdTokens,
            thresholdPct,
          },
        });
      }

      if (contextWindow === undefined) {
        logger.warn("chat.params", "context_unknown", {
          sessionID: input.sessionID,
          data: {
            provider,
            model,
            thresholdTokens,
            thresholdPct,
          },
        });
      }

      tokensBySession.set(input.sessionID, {
        total: existing?.total ?? 0,
        providerID: provider,
        modelID: model,
        contextWindow,
        thresholdTokens,
        thresholdPct,
        warnedContextUnknown,
      });
      logger.debug("chat.params", "hook fired", {
        sessionID: input.sessionID,
        data: {
          provider,
          model,
          active: decision.active,
          autoCompact: decision.autoCompact,
          thresholdTokens,
          thresholdPct,
          contextWindow,
          usedFallbackThreshold: contextWindow === undefined,
          matchedRule: decision.matchedRule,
        },
      });
    },

    "experimental.chat.system.transform": async (input, output) => {
      const decision = cache.resolve({
        sessionID: input.sessionID,
        provider: providerIdFromModel(input),
        model: asString(input.model?.id),
      });
      if (!decision.active || !decision.mergeSystemMessages) {
        return;
      }

      const before = Array.isArray(output.system) ? output.system.length : 0;
      output.system = mergeSystemMessages(Array.isArray(output.system) ? output.system : []);
      if (decision.systemInject && decision.systemInject.length > 0) {
        output.system.push(...decision.systemInject);
      }
      logger.debug("experimental.chat.system.transform", "system prompt transformed", {
        sessionID: input.sessionID,
        data: {
          matchedRule: decision.matchedRule,
          before,
          after: output.system.length,
          injected: decision.systemInject.length,
        },
      });
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const decision = cache.resolve({});
      if (!decision.active || !decision.stripHistoryThinking || !Array.isArray(output.messages)) {
        return;
      }

      const before = output.messages.length;
      output.messages = stripAssistantHistoryThinking(output.messages, decision.reasoningRetention);
      logger.debug("experimental.chat.messages.transform", "history thinking transformed", {
        data: {
          matchedRule: decision.matchedRule,
          before,
          after: output.messages.length,
          retention: decision.reasoningRetention,
        },
      });
    },

    "experimental.text.complete": async (input, output) => {
      const decision = cache.resolve({ sessionID: input.sessionID });
      if (!decision.active || !decision.stripStoredThinkingText || typeof output.text !== "string") {
        return;
      }

      const beforeLength = output.text.length;
      output.text = stripThinkingText(output.text);
      logger.debug("experimental.text.complete", "stripped thinking text", {
        sessionID: input.sessionID,
        data: {
          beforeLength,
          afterLength: output.text.length,
        },
      });
    },

    "chat.headers": async (input, output) => {
      if (input.sessionID) {
        output.headers["x-litellm-session-id"] = input.sessionID;
        output.headers["x-opencode-session"] = input.sessionID;
        logger.debug("chat.headers", "added session affinity headers", {
          sessionID: input.sessionID,
        });
      }
    },
  };
}) satisfies Plugin;

export default plugin;
