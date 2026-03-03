import type { Plugin } from "@opencode-ai/plugin";

import { loadAdapterConfig } from "./config.js";
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
  provider?: { info?: { id?: string } };
}): string {
  return asString(input.provider?.info?.id);
}

function providerIdFromModel(input: { model?: { providerID?: string; providerId?: string } }): string {
  return asString(input.model?.providerID ?? input.model?.providerId);
}

const plugin = (async (ctx) => {
  const config = await loadAdapterConfig(ctx.directory);
  if (!config.enabled) {
    return {};
  }

  const cache = new ParamsCache(config);
  const tracker = createRecoveryTracker(3);

  return {
    event: async (input) => {
      const event = input.event as {
        type?: string;
        properties?: {
          sessionID?: string;
          status?: {
            type?: string;
          };
        };
      };

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
      if (!decision.active || !decision.recoverTrappedToolCalls) {
        return;
      }

      const sessionTracker = {
        attempts: tracker.attempts,
        maxRetries: decision.recoveryMaxRetries,
      };
      if (!canRetry(sessionTracker, sessionID)) {
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
          return;
        }

        recordAttempt(sessionTracker, sessionID);
        await ctx.client.session.promptAsync({
          path: { id: sessionID },
          body: {
            parts: [{ type: "text", text: RECOVERY_MESSAGE }],
          },
        });
      } catch {}
    },

    "chat.params": async (input, _output) => {
      cache.rememberFromChatParams({
        sessionID: input.sessionID,
        provider: providerIdFromChatParams(input),
        model: asString(input.model?.id),
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

      output.system = mergeSystemMessages(Array.isArray(output.system) ? output.system : []);
      if (decision.systemInject && decision.systemInject.length > 0) {
        output.system.push(...decision.systemInject);
      }
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const decision = cache.resolve({});
      if (!decision.active || !decision.stripHistoryThinking || !Array.isArray(output.messages)) {
        return;
      }

      output.messages = stripAssistantHistoryThinking(output.messages, decision.reasoningRetention);
    },

    "experimental.text.complete": async (input, output) => {
      const decision = cache.resolve({ sessionID: input.sessionID });
      if (!decision.active || !decision.stripStoredThinkingText || typeof output.text !== "string") {
        return;
      }

      output.text = stripThinkingText(output.text);
    },

    "chat.headers": async (input, output) => {
      if (input.sessionID) {
        output.headers["x-litellm-session-id"] = input.sessionID;
        output.headers["x-opencode-session"] = input.sessionID;
      }
    },
  };
}) satisfies Plugin;

export default plugin;
