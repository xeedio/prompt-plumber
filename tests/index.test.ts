import { beforeEach, describe, expect, it, vi } from "vitest";

import type { AdapterConfig } from "../src/config.js";

const loadAdapterConfigMock = vi.hoisted(() => vi.fn());

vi.mock("../src/config.js", async () => {
  const actual = await vi.importActual<typeof import("../src/config.js")>("../src/config.js");
  return {
    ...actual,
    loadAdapterConfig: loadAdapterConfigMock,
  };
});

const activeConfig: AdapterConfig = {
  enabled: true,
  defaults: {
    merge_system_messages: true,
    strip_history_thinking: true,
    strip_stored_thinking_text: true,
    reasoning_retention: "none",
  },
  rules: [
    {
      name: "vllm-qwen3",
      providers: ["vllm"],
      model_patterns: ["^qwen3([-.]|$)", "qwen3-coder"],
      merge_system_messages: true,
      strip_history_thinking: true,
      strip_stored_thinking_text: true,
      reasoning_retention: "none",
    },
  ],
};

describe("plugin hooks", () => {
  beforeEach(() => {
    loadAdapterConfigMock.mockReset();
  });

  it("returns no hooks when adapter config is disabled", async () => {
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      enabled: false,
    });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = await pluginFactory({ directory: "/tmp/project" } as never);

    expect(hooks).toEqual({});
    expect(loadAdapterConfigMock).toHaveBeenCalledWith("/tmp/project");
  });

  it("sets session affinity chat headers for vllm and anthropic", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    const vllmOutput = { headers: {} as Record<string, string> };
    await hooks["chat.headers"](
      {
        sessionID: "s-vllm",
        provider: { info: { id: "vllm" } },
      } as never,
      vllmOutput,
    );
    expect(vllmOutput.headers).toEqual({
      "x-litellm-session-id": "s-vllm",
      "x-opencode-session": "s-vllm",
    });

    const anthropicOutput = { headers: {} as Record<string, string> };
    await hooks["chat.headers"](
      {
        sessionID: "s-anthropic",
        provider: { info: { id: "anthropic" } },
      } as never,
      anthropicOutput,
    );
    expect(anthropicOutput.headers).toEqual({
      "x-litellm-session-id": "s-anthropic",
      "x-opencode-session": "s-anthropic",
    });
  });

  it("does not set session affinity chat headers without sessionID", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    const emptySessionOutput = { headers: {} as Record<string, string> };
    await hooks["chat.headers"](
      {
        sessionID: "",
      } as never,
      emptySessionOutput,
    );
    expect(emptySessionOutput.headers).toEqual({});

    const undefinedSessionOutput = { headers: {} as Record<string, string> };
    await hooks["chat.headers"](
      {
        sessionID: undefined,
      } as never,
      undefinedSessionOutput,
    );
    expect(undefinedSessionOutput.headers).toEqual({});
  });

  it("wires all hooks and mutates outputs only for active vllm sessions", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-active",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    const systemOutput = { system: [" alpha ", "beta"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-active",
        model: { providerId: "vllm", id: "qwen3-coder-next" },
      },
      systemOutput,
    );
    expect(systemOutput.system).toEqual(["alpha\n\nbeta"]);

    const messageOutput = {
      messages: [
        {
          info: { role: "assistant" },
          parts: [
            { type: "reasoning", text: "secret" },
            { type: "text", text: "visible<think>hidden</think>" },
          ],
        },
        {
          info: { role: "assistant" },
          parts: [{ type: "thinking", text: "last secret" }],
        },
      ],
    };

    await hooks["experimental.chat.messages.transform"]({}, messageOutput);
    expect(messageOutput.messages[0].parts).toEqual([{ type: "text", text: "visible" }]);
    expect(messageOutput.messages[1].parts).toEqual([]);

    const textOutput = { text: "head <think>private</think> tail" };
    await hooks["experimental.text.complete"]({ sessionID: "s-active" }, textOutput);
    expect(textOutput.text).toBe("head  tail");

    const hooksWithoutDecision = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;
    const noSessionMessagesOutput = {
      messages: [
        {
          info: { role: "assistant" },
          parts: [{ type: "reasoning", text: "must-stay-without-decision" }],
        },
      ],
    };
    await hooksWithoutDecision["experimental.chat.messages.transform"]({}, noSessionMessagesOutput);
    expect(noSessionMessagesOutput.messages[0].parts).toEqual([
      { type: "reasoning", text: "must-stay-without-decision" },
    ]);

    await hooks["chat.params"](
      {
        sessionID: "s-bad-types",
        provider: { info: { id: 101 } },
        model: { id: { nested: true } },
      } as never,
      {},
    );

    const nonArraySystem = { system: "not-an-array" };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-active",
        model: { providerID: "vllm", id: "qwen3-coder-next" },
      },
      nonArraySystem as never,
    );
    expect(nonArraySystem.system).toEqual([]);

    const badMessagesOutput = { messages: "not-an-array" };
    await hooks["experimental.chat.messages.transform"]({}, badMessagesOutput as never);
    expect(badMessagesOutput.messages).toBe("not-an-array");

    const nonStringText = { text: 123 as never };
    await hooks["experimental.text.complete"]({ sessionID: "s-active" }, nonStringText);
    expect(nonStringText.text).toBe(123);

    await hooks["chat.params"](
      {
        sessionID: "s-anthropic",
        provider: { info: { id: "anthropic" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    const anthropicSystemOutput = { system: ["one", "two"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-anthropic",
        model: { providerID: "anthropic", id: "qwen3-coder-next" },
      },
      anthropicSystemOutput,
    );
    expect(anthropicSystemOutput.system).toEqual(["one", "two"]);
  });
});
