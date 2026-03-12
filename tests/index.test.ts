import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AdapterConfig } from "../src/config.js";
import { RECOVERY_MESSAGE } from "../src/hooks/toolcall-recovery.js";

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
      name: "vllm-qwen3",
      providers: ["vllm"],
      model_patterns: ["^qwen3([-.]|$)", "qwen3-coder"],
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
  ],
};

function createMockClient() {
  return {
    session: {
      messages: vi.fn(),
      promptAsync: vi.fn(),
      summarize: vi.fn(),
    },
  };
}

describe("plugin hooks", () => {
  let testLogDir = "";

  beforeEach(async () => {
    loadAdapterConfigMock.mockReset();
    testLogDir = await mkdtemp(path.join(os.tmpdir(), "pp-test-"));
    process.env.PROMPT_PLUMBER_LOG_DIR = testLogDir;
    process.env.PROMPT_PLUMBER_LOG_LEVEL = "error";
  });

  afterEach(async () => {
    delete process.env.PROMPT_PLUMBER_LOG_LEVEL;
    delete process.env.PROMPT_PLUMBER_LOG_DIR;
    await rm(testLogDir, { recursive: true, force: true }).catch(() => {});
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

  it("extracts provider from provider.id in chat.params", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-provider-id",
        provider: { id: "vllm" },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    const systemOutput = { system: ["left", "right"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-provider-id",
        model: { id: "qwen3-coder-next" },
      },
      systemOutput,
    );

    expect(systemOutput.system).toEqual(["left\n\nright"]);
  });

  it("matches litellm rules when chat.params provider uses provider.id", async () => {
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [
        {
          ...activeConfig.rules[0],
          name: "litellm-coder",
          providers: ["litellm"],
          model_patterns: ["coder"],
        },
      ],
    });
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-litellm-provider-id",
        provider: { id: "litellm" },
        model: { id: "coder" },
      },
      {},
    );

    const textOutput = { text: "head <think>private</think> tail" };
    await hooks["experimental.text.complete"]({ sessionID: "s-litellm-provider-id" }, textOutput);

    expect(textOutput.text).toBe("head  tail");
  });

  it("triggers recovery prompt on session.idle when assistant has trapped tool_call XML", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({
      data: [
        {
          info: { role: "assistant" },
          parts: [
            {
              type: "reasoning",
              text: "thinking <tool_call>{\"name\":\"bash\"}</tool_call>",
            },
          ],
        },
      ],
    });
    client.session.promptAsync.mockResolvedValue({});

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-recovery",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-recovery",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 1000, output: 100, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-recovery" } },
    });

    expect(client.session.messages).toHaveBeenCalledWith({ path: { id: "s-recovery" } });
    expect(client.session.promptAsync).toHaveBeenCalledTimes(1);
    expect(client.session.promptAsync).toHaveBeenCalledWith({
      path: { id: "s-recovery" },
      body: {
        parts: [{ type: "text", text: RECOVERY_MESSAGE }],
      },
    });
  });

  it("triggers recovery prompt on session.status idle events", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({
      data: [
        {
          info: { role: "assistant" },
          parts: [{ type: "reasoning", text: "x <tool_call>bad</tool_call> y" }],
        },
      ],
    });
    client.session.promptAsync.mockResolvedValue({});

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-status",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "session.status",
        properties: { sessionID: "s-status", status: { type: "idle" } },
      },
    });

    expect(client.session.promptAsync).toHaveBeenCalledTimes(1);
  });

  it("does not trigger recovery prompt when no trapped tool_call XML exists", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({
      data: [
        {
          info: { role: "assistant" },
          parts: [{ type: "reasoning", text: "plain reasoning" }],
        },
      ],
    });
    client.session.promptAsync.mockResolvedValue({});

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-no-trap",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-no-trap" } },
    });

    expect(client.session.messages).toHaveBeenCalledWith({ path: { id: "s-no-trap" } });
    expect(client.session.promptAsync).not.toHaveBeenCalled();
  });

  it("does not trigger recovery prompt for inactive anthropic sessions", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({
      data: [
        {
          info: { role: "assistant" },
          parts: [
            {
              type: "reasoning",
              text: "thinking <tool_call>{\"name\":\"bash\"}</tool_call>",
            },
          ],
        },
      ],
    });
    client.session.promptAsync.mockResolvedValue({});

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-anthropic-idle",
        provider: { info: { id: "anthropic" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-anthropic-idle" } },
    });

    expect(client.session.messages).not.toHaveBeenCalled();
    expect(client.session.promptAsync).not.toHaveBeenCalled();
  });

  it("respects per-session max recovery retries", async () => {
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [
        {
          ...activeConfig.rules[0],
          recovery_max_retries: 2,
        },
      ],
    });
    const client = createMockClient();
    client.session.messages.mockResolvedValue({
      data: [
        {
          info: { role: "assistant" },
          parts: [
            {
              type: "reasoning",
              text: "thinking <tool_call>{\"name\":\"bash\"}</tool_call>",
            },
          ],
        },
      ],
    });
    client.session.promptAsync.mockResolvedValue({});

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-limit",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-limit",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 1000, output: 100, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-limit" } },
    });
    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-limit" } },
    });
    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-limit" } },
    });

    expect(client.session.messages).toHaveBeenCalledTimes(2);
    expect(client.session.promptAsync).toHaveBeenCalledTimes(2);
  });

  it("appends system_inject entries after merged system messages", async () => {
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [
        {
          ...activeConfig.rules[0],
          system_inject: [
            "Close thinking before action output.",
            "Emit tool calls outside <think> blocks.",
          ],
        },
      ],
    });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-inject",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    const systemOutput = { system: ["alpha", "beta"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-inject",
        model: { providerID: "vllm", id: "qwen3-coder-next" },
      },
      systemOutput,
    );

    expect(systemOutput.system).toEqual([
      "alpha\n\nbeta",
      "Close thinking before action output.",
      "Emit tool calls outside <think> blocks.",
    ]);
  });

  it("does not append system_inject entries for inactive anthropic sessions", async () => {
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [
        {
          ...activeConfig.rules[0],
          system_inject: ["Close thinking before action output."],
        },
      ],
    });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-anthropic-inactive",
        provider: { info: { id: "anthropic" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    const systemOutput = { system: ["one", "two"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-anthropic-inactive",
        model: { providerID: "anthropic", id: "qwen3-coder-next" },
      },
      systemOutput,
    );

    expect(systemOutput.system).toEqual(["one", "two"]);
  });

  it("keeps merged system output unchanged when system_inject is empty", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project" } as never)) as Record<
      string,
      (input: any, output: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-empty-inject",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    const systemOutput = { system: ["left", "right"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "s-empty-inject",
        model: { providerId: "vllm", id: "qwen3-coder-next" },
      },
      systemOutput,
    );

    expect(systemOutput.system).toEqual(["left\n\nright"]);
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

  it("triggers proactive compaction via session.summarize when threshold is exceeded", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-compact",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-compact",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: {
              input: 169000,
              output: 2000,
              reasoning: 0,
              cache: { read: 1000, write: 0 },
            },
          },
        },
      },
    });

    await hooks["event"]({
      event: {
        type: "session.idle",
        properties: { sessionID: "s-compact" },
      },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
    expect(client.session.summarize).toHaveBeenCalledWith({
      path: { id: "s-compact" },
      body: {
        providerID: "vllm",
        modelID: "qwen3-coder-next",
      },
    });
  });

  it("does not trigger proactive compaction when auto_compact is false", async () => {
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [
        {
          ...activeConfig.rules[0],
          auto_compact: false,
        },
      ],
    });
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-no-compact",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-no-compact",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: {
              input: 180000,
              output: 2000,
              cache: { read: 0, write: 0 },
            },
          },
        },
      },
    });

    await hooks["event"]({
      event: {
        type: "session.idle",
        properties: { sessionID: "s-no-compact" },
      },
    });

    expect(client.session.summarize).not.toHaveBeenCalled();
  });

  it("hot-reloads config on chat.params and applies updated decisions", async () => {
    const noCompactConfig: AdapterConfig = {
      ...activeConfig,
      rules: [
        {
          ...activeConfig.rules[0],
          auto_compact: false,
        },
      ],
    };

    loadAdapterConfigMock
      .mockResolvedValueOnce(activeConfig)
      .mockResolvedValueOnce(activeConfig)
      .mockResolvedValueOnce(noCompactConfig);

    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-hot-reload",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-hot-reload",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 170000, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks["event"]({
      event: {
        type: "session.idle",
        properties: { sessionID: "s-hot-reload" },
      },
    });
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    await hooks["event"]({
      event: {
        type: "session.compacted",
        properties: { sessionID: "s-hot-reload" },
      },
    });

    await hooks["chat.params"](
      {
        sessionID: "s-hot-reload",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-hot-reload",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 200000, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks["event"]({
      event: {
        type: "session.idle",
        properties: { sessionID: "s-hot-reload" },
      },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
    expect(loadAdapterConfigMock).toHaveBeenCalledTimes(3);
  });

  it("uses dynamic context from chat.params model.limit.context", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-dynamic",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-dynamic",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 660, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-dynamic" } },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
  });

  it("prefers model.limit.input over model.limit.context for threshold", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-input-limit",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000, input: 500 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-input-limit",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 400, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-input-limit" } },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
    expect(client.session.summarize).toHaveBeenCalledWith({
      path: { id: "s-input-limit" },
      body: { providerID: "vllm", modelID: "qwen3-coder-next" },
    });
  });

  it("does not compact when tokens are below dynamic threshold", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-dynamic-below",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-dynamic-below",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 659, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-dynamic-below" } },
    });

    expect(client.session.summarize).not.toHaveBeenCalled();
  });

  it("falls back to absolute threshold when context is unknown", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-fallback",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-fallback",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 170000, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-fallback" } },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
  });

  it("recalculates threshold when model context changes mid-session", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-switch",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );

    await hooks["chat.params"](
      {
        sessionID: "s-switch",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 2000 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-switch",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 1200, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-switch" } },
    });

    expect(client.session.summarize).not.toHaveBeenCalled();
  });

  it("uses fallback threshold for a session with no chat.params", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-seed",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-no-chat-params",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 180000, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-no-chat-params" } },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
    expect(client.session.summarize).toHaveBeenCalledWith({
      path: { id: "s-no-chat-params" },
      body: { providerID: "vllm", modelID: "qwen3-coder-next" },
    });
  });

  it("falls back when context window is zero", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-zero-context",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 0 } },
      },
      {},
    );
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-zero-context",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 170000, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-zero-context" } } });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
  });

  it("tracks thresholds independently across sessions", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-a",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks["chat.params"](
      {
        sessionID: "s-b",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 2000 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-a",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 700, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-b",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 700, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-a" } } });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-b" } } });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
    expect(client.session.summarize).toHaveBeenCalledWith({
      path: { id: "s-a" },
      body: { providerID: "vllm", modelID: "qwen3-coder-next" },
    });
  });

  it("clears in-flight guard on session.compacted", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-guard",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-guard",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 800, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });

    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-guard" } } });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-guard" } } });
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    await hooks["event"]({ event: { type: "session.compacted", properties: { sessionID: "s-guard" } } });
    await hooks["chat.params"](
      {
        sessionID: "s-guard",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-guard",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 800, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-guard" } } });

    expect(client.session.summarize).toHaveBeenCalledTimes(2);
  });

  it("supports compaction threshold pct edge values", async () => {
    const cfg = {
      ...activeConfig,
      rules: [{ ...activeConfig.rules[0], compaction_threshold_pct: 0, compaction_threshold: 170000 }],
    };
    loadAdapterConfigMock.mockResolvedValue(cfg);

    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-pct-0",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-pct-0" } } });
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    client.session.summarize.mockClear();
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [{ ...activeConfig.rules[0], compaction_threshold_pct: 1 }],
    });
    const hooks2 = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;
    await hooks2["chat.params"](
      {
        sessionID: "s-pct-1",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks2["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-pct-1",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 999, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks2["event"]({ event: { type: "session.idle", properties: { sessionID: "s-pct-1" } } });
    expect(client.session.summarize).not.toHaveBeenCalled();

    await hooks2["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-pct-1",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 1000, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks2["event"]({ event: { type: "session.idle", properties: { sessionID: "s-pct-1" } } });
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    client.session.summarize.mockClear();
    loadAdapterConfigMock.mockResolvedValue({
      ...activeConfig,
      rules: [{ ...activeConfig.rules[0], compaction_threshold_pct: 0.5 }],
    });
    const hooks3 = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;
    await hooks3["chat.params"](
      {
        sessionID: "s-pct-half",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks3["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-pct-half",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 500, output: 0, cache: { read: 0, write: 0 } },
          },
        },
      },
    });
    await hooks3["event"]({ event: { type: "session.idle", properties: { sessionID: "s-pct-half" } } });
    expect(client.session.summarize).toHaveBeenCalledTimes(1);
  });

  it("handles context transitions and model.context fallback", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-context-change",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", context: 1000 },
      },
      {},
    );
    await hooks["chat.params"](
      {
        sessionID: "s-context-change",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );
    await hooks["chat.params"](
      {
        sessionID: "s-context-non-object",
        provider: { info: { id: "vllm" } },
        model: "qwen3-coder-next",
      } as never,
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-context-change",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { total: 170000 },
          },
        },
      },
    });
    await hooks["event"]({
      event: { type: "session.idle", properties: { sessionID: "s-context-change" } },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
  });

  it("handles message.part.updated and ignores malformed token events", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize.mockResolvedValue({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-step-finish",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );

    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-step-finish",
            providerID: 12,
            modelID: null,
            tokens: null,
          },
        },
      },
    });

    await hooks["event"]({
      event: {
        type: "message.part.updated",
        properties: {
          part: {
            sessionID: "s-step-finish",
            type: "step-finish",
            tokens: { input: 500, output: 160 },
          },
        },
      },
    });

    await hooks["event"]({
      event: {
        type: "session.status",
        properties: {
          sessionID: "s-step-finish",
          status: { type: "running" },
        },
      },
    });

    await hooks["event"]({
      event: {
        type: "session.idle",
        properties: { sessionID: "s-step-finish" },
      },
    });

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
  });

  it("clears in-flight guard when summarize throws", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.messages.mockResolvedValue({ data: [] });
    client.session.summarize
      .mockRejectedValueOnce(new Error("temporary summarize failure"))
      .mockResolvedValueOnce({ data: true });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-summarize-error",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next", limit: { context: 1000 } },
      },
      {},
    );
    await hooks["event"]({
      event: {
        type: "message.updated",
        properties: {
          info: {
            role: "assistant",
            sessionID: "s-summarize-error",
            providerID: "vllm",
            modelID: "qwen3-coder-next",
            tokens: { input: 1000 },
          },
        },
      },
    });

    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-summarize-error" } } });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-summarize-error" } } });

    expect(client.session.summarize).toHaveBeenCalledTimes(2);
  });

  it("handles recovery message fetch failures and malformed assistant payloads", async () => {
    loadAdapterConfigMock.mockResolvedValue(activeConfig);
    const client = createMockClient();
    client.session.promptAsync.mockResolvedValue({});
    client.session.messages
      .mockRejectedValueOnce(new Error("message fetch failed"))
      .mockResolvedValueOnce({ data: [{ info: { role: "assistant" } }] })
      .mockResolvedValueOnce({ data: [{ info: { role: "assistant" }, parts: [{ type: "reasoning", text: "ok" }] }] });

    const pluginFactory = (await import("../src/index.js")).default;
    const hooks = (await pluginFactory({ directory: "/tmp/project", client } as never)) as Record<
      string,
      (input: any, output?: any) => Promise<void>
    >;

    await hooks["chat.params"](
      {
        sessionID: "s-recovery-edge",
        provider: { info: { id: "vllm" } },
        model: { id: "qwen3-coder-next" },
      },
      {},
    );

    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-recovery-edge" } } });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-recovery-edge" } } });
    await hooks["event"]({ event: { type: "session.idle", properties: { sessionID: "s-recovery-edge" } } });

    expect(client.session.promptAsync).not.toHaveBeenCalled();
    expect(client.session.messages).toHaveBeenCalledTimes(3);
  });
});
