import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AdapterConfig } from "../src/config.js";

const loadAdapterConfigMock = vi.hoisted(() => vi.fn());

vi.mock("../src/config.js", async () => {
  const actual = await vi.importActual<typeof import("../src/config.js")>("../src/config.js");
  return {
    ...actual,
    loadAdapterConfig: loadAdapterConfigMock,
  };
});

type Hooks = Record<string, (input: any, output?: any) => Promise<void>>;

type LogEntry = {
  level: string;
  hook: string;
  message: string;
  sessionID?: string;
  data?: Record<string, unknown>;
};

const simConfig: AdapterConfig = {
  enabled: true,
  log_level: "debug",
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
      name: "litellm-coder",
      providers: ["litellm"],
      model_patterns: ["coder"],
      merge_system_messages: true,
      strip_history_thinking: true,
      strip_stored_thinking_text: true,
      reasoning_retention: "none",
      recover_trapped_tool_calls: true,
      recovery_max_retries: 3,
      system_inject: ["You MUST close your thinking with </think> BEFORE any tool calls."],
      auto_compact: true,
      compaction_threshold: 170000,
      compaction_threshold_pct: 0.66,
    },
  ],
};

function createMockClient() {
  return {
    session: {
      messages: vi.fn().mockResolvedValue({ data: [] }),
      promptAsync: vi.fn().mockResolvedValue({}),
      summarize: vi.fn().mockResolvedValue({ data: true }),
    },
  };
}

function tokenEvent(sessionID: string, input: number, output = 0) {
  return {
    event: {
      type: "message.updated",
      properties: {
        info: {
          role: "assistant",
          sessionID,
          providerID: "litellm",
          modelID: "coder",
          tokens: { input, output, reasoning: 0, cache: { read: 0, write: 0 } },
        },
      },
    },
  };
}

function idleEvent(sessionID: string) {
  return { event: { type: "session.idle", properties: { sessionID } } };
}

function compactedEvent(sessionID: string) {
  return { event: { type: "session.compacted", properties: { sessionID } } };
}

function chatParamsInput(sessionID: string, context?: number) {
  return {
    sessionID,
    provider: { info: { id: "litellm" } },
    model: {
      id: "coder",
      ...(context !== undefined ? { limit: { context } } : {}),
    },
  };
}

async function waitForLogEntries(logDir: string, predicate: (entries: LogEntry[]) => boolean) {
  const day = new Date().toISOString().slice(0, 10);
  const filePath = path.join(logDir, `${day}.jsonl`);
  let entries: LogEntry[] = [];

  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const raw = await readFile(filePath, "utf8");
      entries = raw
        .split("\n")
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as LogEntry);
      if (predicate(entries)) {
        return entries;
      }
    } catch {
      // no-op
    }

    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  return entries;
}

async function createHooks(client: ReturnType<typeof createMockClient>, config: AdapterConfig = simConfig) {
  loadAdapterConfigMock.mockResolvedValue(config);
  const pluginFactory = (await import("../src/index.js")).default;
  return (await pluginFactory({ directory: "/tmp/project", client } as never)) as Hooks;
}

describe("integration simulation", () => {
  let testLogDir = "";

  beforeEach(async () => {
    loadAdapterConfigMock.mockReset();
    testLogDir = await mkdtemp(path.join(os.tmpdir(), "pp-integration-sim-"));
    process.env.PROMPT_PLUMBER_LOG_DIR = testLogDir;
    process.env.PROMPT_PLUMBER_LOG_LEVEL = "debug";
  });

  afterEach(async () => {
    delete process.env.PROMPT_PLUMBER_LOG_DIR;
    delete process.env.PROMPT_PLUMBER_LOG_LEVEL;
    await rm(testLogDir, { recursive: true, force: true }).catch(() => {});
  });

  it("simulates full session lifecycle with context discovery and compaction", async () => {
    const client = createMockClient();
    const hooks = await createHooks(client);

    await hooks["chat.params"](chatParamsInput("sim-full", 262144), {});
    await hooks["event"](tokenEvent("sim-full", 50000));
    await hooks["event"](idleEvent("sim-full"));
    await hooks["event"](tokenEvent("sim-full", 100000));
    await hooks["event"](idleEvent("sim-full"));
    await hooks["event"](tokenEvent("sim-full", 150000));
    await hooks["event"](idleEvent("sim-full"));

    expect(client.session.summarize).not.toHaveBeenCalled();

    await hooks["event"](tokenEvent("sim-full", 175000));
    await hooks["event"](idleEvent("sim-full"));
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    await hooks["event"](compactedEvent("sim-full"));
    await hooks["chat.params"](chatParamsInput("sim-full", 262144), {});
    await hooks["event"](tokenEvent("sim-full", 175000));
    await hooks["event"](idleEvent("sim-full"));
    expect(client.session.summarize).toHaveBeenCalledTimes(2);

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) =>
        logs.some((entry) => entry.sessionID === "sim-full" && entry.message === "context_discovered") &&
        logs.filter((entry) => entry.sessionID === "sim-full" && entry.message === "compaction_triggered")
          .length >= 2,
    );

    const contextDiscovered = entries.find(
      (entry) => entry.sessionID === "sim-full" && entry.message === "context_discovered",
    );
    expect(contextDiscovered?.data?.thresholdTokens).toBe(
      Math.floor(262144 * simConfig.defaults.compaction_threshold_pct),
    );
  });

  it("simulates model switch mid-session changing compaction threshold", async () => {
    const client = createMockClient();
    const hooks = await createHooks(client);

    await hooks["chat.params"](chatParamsInput("sim-switch", 262144), {});
    await hooks["event"](tokenEvent("sim-switch", 100000));
    await hooks["event"](idleEvent("sim-switch"));
    expect(client.session.summarize).not.toHaveBeenCalled();

    await hooks["chat.params"](chatParamsInput("sim-switch", 131072), {});
    await hooks["event"](idleEvent("sim-switch"));
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) =>
        logs.some((entry) => entry.sessionID === "sim-switch" && entry.message === "context_changed") &&
        logs.some((entry) => entry.sessionID === "sim-switch" && entry.message === "compaction_triggered"),
    );

    const changed = entries.find(
      (entry) => entry.sessionID === "sim-switch" && entry.message === "context_changed",
    );
    expect(changed?.data?.newThresholdTokens).toBe(86507);
  });

  it("simulates session reaching compaction with unknown context (fallback threshold)", async () => {
    const client = createMockClient();
    const hooks = await createHooks(client);

    await hooks["chat.params"](chatParamsInput("sim-fallback"), {});
    await hooks["event"](tokenEvent("sim-fallback", 160000));
    await hooks["event"](idleEvent("sim-fallback"));
    expect(client.session.summarize).not.toHaveBeenCalled();

    await hooks["event"](tokenEvent("sim-fallback", 175000));
    await hooks["event"](idleEvent("sim-fallback"));
    expect(client.session.summarize).toHaveBeenCalledTimes(1);

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) =>
        logs.some((entry) => entry.sessionID === "sim-fallback" && entry.message === "context_unknown") &&
        logs.some((entry) => entry.sessionID === "sim-fallback" && entry.message === "compaction_triggered"),
    );

    const unknown = entries.find(
      (entry) => entry.sessionID === "sim-fallback" && entry.message === "context_unknown",
    );
    expect(unknown?.level).toBe("warn");
  });

  it("simulates trapped tool-call recovery with retry exhaustion", async () => {
    const recoveryConfig: AdapterConfig = {
      ...simConfig,
      defaults: {
        ...simConfig.defaults,
        recovery_max_retries: 2,
      },
      rules: [
        {
          ...simConfig.rules[0],
          recovery_max_retries: 2,
        },
      ],
    };

    const client = createMockClient();
    client.session.messages.mockResolvedValue({
      data: [
        {
          info: { role: "assistant" },
          parts: [{ type: "reasoning", text: "thinking <tool_call>{\"name\":\"bash\"}</tool_call>" }],
        },
      ],
    });

    const hooks = await createHooks(client, recoveryConfig);
    await hooks["chat.params"](chatParamsInput("sim-retry", 262144), {});

    await hooks["event"](idleEvent("sim-retry"));
    await hooks["event"](idleEvent("sim-retry"));
    await hooks["event"](idleEvent("sim-retry"));

    expect(client.session.promptAsync).toHaveBeenCalledTimes(2);

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) =>
        logs.filter((entry) => entry.sessionID === "sim-retry" && entry.message === "tool-call recovery triggered")
          .length === 2 &&
        logs.some(
          (entry) =>
            entry.sessionID === "sim-retry" && entry.message === "recovery skipped due to retry limit",
        ),
    );

    expect(
      entries.some(
        (entry) =>
          entry.sessionID === "sim-retry" && entry.message === "recovery skipped due to retry limit",
      ),
    ).toBe(true);
  });

  it("simulates concurrent sessions with independent thresholds", async () => {
    const client = createMockClient();
    const hooks = await createHooks(client);

    await hooks["chat.params"](chatParamsInput("sim-a", 100000), {});
    await hooks["chat.params"](chatParamsInput("sim-b", 262144), {});

    await hooks["event"](tokenEvent("sim-a", 70000));
    await hooks["event"](tokenEvent("sim-b", 70000));

    await hooks["event"](idleEvent("sim-a"));
    await hooks["event"](idleEvent("sim-b"));

    expect(client.session.summarize).toHaveBeenCalledTimes(1);
    expect(client.session.summarize).toHaveBeenCalledWith({
      path: { id: "sim-a" },
      body: { providerID: "litellm", modelID: "coder" },
    });

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) =>
        logs.some((entry) => entry.sessionID === "sim-a" && entry.message === "compaction_triggered") &&
        logs.some((entry) => entry.sessionID === "sim-b" && entry.message === "compaction_not_needed"),
    );

    expect(entries.some((entry) => entry.sessionID === "sim-b" && entry.message === "compaction_not_needed")).toBe(
      true,
    );
  });

  it("simulates compaction failure with retry on next idle", async () => {
    const client = createMockClient();
    client.session.summarize
      .mockRejectedValueOnce(new Error("summarize failed"))
      .mockResolvedValueOnce({ data: true });

    const hooks = await createHooks(client);
    await hooks["chat.params"](chatParamsInput("sim-failure", 100000), {});
    await hooks["event"](tokenEvent("sim-failure", 70000));

    await hooks["event"](idleEvent("sim-failure"));
    await hooks["event"](idleEvent("sim-failure"));

    expect(client.session.summarize).toHaveBeenCalledTimes(2);

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) => logs.some((entry) => entry.sessionID === "sim-failure" && entry.message === "failed proactive compaction"),
    );

    expect(entries.some((entry) => entry.sessionID === "sim-failure" && entry.message === "failed proactive compaction")).toBe(
      true,
    );
  });

  it("simulates system prompt merge and thinking strip in same session", async () => {
    const client = createMockClient();
    const hooks = await createHooks(client);

    await hooks["chat.params"](chatParamsInput("sim-transform", 262144), {});

    const systemOutput = { system: [" alpha ", "alpha", "beta"] };
    await hooks["experimental.chat.system.transform"](
      {
        sessionID: "sim-transform",
        model: { providerID: "litellm", id: "coder" },
      },
      systemOutput,
    );

    expect(systemOutput.system).toEqual([
      "alpha\n\nbeta",
      "You MUST close your thinking with </think> BEFORE any tool calls.",
    ]);

    const messagesOutput = {
      messages: [
        {
          info: { role: "assistant" },
          parts: [
            { type: "reasoning", text: "hidden" },
            { type: "text", text: "answer <think>internal</think> done" },
          ],
        },
        {
          info: { role: "user" },
          parts: [{ type: "text", text: "keep-user" }],
        },
      ],
    };
    await hooks["experimental.chat.messages.transform"]({}, messagesOutput);
    expect(messagesOutput.messages[0].parts).toEqual([{ type: "text", text: "answer  done" }]);
    expect(messagesOutput.messages[1].parts).toEqual([{ type: "text", text: "keep-user" }]);

    const textOutput = { text: "prefix <think>secret</think> suffix" };
    await hooks["experimental.text.complete"]({ sessionID: "sim-transform" }, textOutput);
    expect(textOutput.text).toBe("prefix  suffix");

    const entries = await waitForLogEntries(
      testLogDir,
      (logs) =>
        logs.some((entry) => entry.sessionID === "sim-transform" && entry.message === "system prompt transformed") &&
        logs.some((entry) => entry.message === "history thinking transformed") &&
        logs.some((entry) => entry.sessionID === "sim-transform" && entry.message === "stripped thinking text"),
    );

    expect(entries.some((entry) => entry.sessionID === "sim-transform" && entry.message === "system prompt transformed")).toBe(
      true,
    );
  });
});
