import type { Plugin } from "@opencode-ai/plugin";

import { loadAdapterConfig } from "./config.js";
import { ParamsCache } from "./hooks/params-cache.js";
import { mergeSystemMessages } from "./hooks/system-merge.js";
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

  return {
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
  };
}) satisfies Plugin;

export default plugin;
