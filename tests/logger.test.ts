import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createHookLogger } from "../src/hooks/logger.js";

const envToRestore = {
  PROMPT_PLUMBER_LOG_DIR: process.env.PROMPT_PLUMBER_LOG_DIR,
  PROMPT_PLUMBER_LOG_LEVEL: process.env.PROMPT_PLUMBER_LOG_LEVEL,
};

afterEach(() => {
  if (envToRestore.PROMPT_PLUMBER_LOG_DIR === undefined) delete process.env.PROMPT_PLUMBER_LOG_DIR;
  else process.env.PROMPT_PLUMBER_LOG_DIR = envToRestore.PROMPT_PLUMBER_LOG_DIR;

  if (envToRestore.PROMPT_PLUMBER_LOG_LEVEL === undefined) delete process.env.PROMPT_PLUMBER_LOG_LEVEL;
  else process.env.PROMPT_PLUMBER_LOG_LEVEL = envToRestore.PROMPT_PLUMBER_LOG_LEVEL;
});

describe("hook logger", () => {
  it("writes all log levels when level is debug", async () => {
    const logDir = await mkdtemp(path.join(os.tmpdir(), "pp-logger-"));
    const logger = createHookLogger({ level: "debug", logDir });

    logger.debug("event", "debug-entry");
    logger.info("event", "info-entry");
    logger.warn("event", "warn-entry");
    logger.error("event", "error-entry");
    await logger.flush();

    const day = new Date().toISOString().slice(0, 10);
    const content = await readFile(path.join(logDir, `${day}.jsonl`), "utf8");
    const lines = content.trim().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(lines.map((line) => line.level)).toEqual(["debug", "info", "warn", "error"]);
    await rm(logDir, { recursive: true, force: true });
  });

  it("writes jsonl entries using the daily file name", async () => {
    const logDir = await mkdtemp(path.join(os.tmpdir(), "pp-logger-"));
    const logger = createHookLogger({ level: "debug", logDir });

    logger.info("plugin", "initialized", { sessionID: "s1", data: { ok: true } });
    logger.debug("chat.params", "decision", { data: { matchedRule: "vllm-default" } });
    await logger.flush();

    const day = new Date().toISOString().slice(0, 10);
    const content = await readFile(path.join(logDir, `${day}.jsonl`), "utf8");
    const lines = content.trim().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(lines).toHaveLength(2);
    expect(lines[0].level).toBe("info");
    expect(lines[0].hook).toBe("plugin");
    expect(lines[0].sessionID).toBe("s1");
    expect(lines[1].level).toBe("debug");

    await rm(logDir, { recursive: true, force: true });
  });

  it("filters entries below configured level", async () => {
    const logDir = await mkdtemp(path.join(os.tmpdir(), "pp-logger-"));
    const logger = createHookLogger({ level: "warn", logDir });

    logger.debug("plugin", "debug");
    logger.info("plugin", "info");
    logger.warn("plugin", "warn");
    await logger.flush();

    const day = new Date().toISOString().slice(0, 10);
    const content = await readFile(path.join(logDir, `${day}.jsonl`), "utf8");
    const lines = content.trim().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(lines).toHaveLength(1);
    expect(lines[0].level).toBe("warn");

    await rm(logDir, { recursive: true, force: true });
  });

  it("writes valid json format for each line", async () => {
    const logDir = await mkdtemp(path.join(os.tmpdir(), "pp-logger-"));
    const logger = createHookLogger({ level: "info", logDir });

    logger.info("chat.params", "context_discovered", {
      sessionID: "s-json",
      data: { contextWindow: 1000, thresholdTokens: 660 },
    });
    await logger.flush();

    const day = new Date().toISOString().slice(0, 10);
    const content = await readFile(path.join(logDir, `${day}.jsonl`), "utf8");
    const parsed = JSON.parse(content.trim()) as Record<string, unknown>;

    expect(typeof parsed.timestamp).toBe("string");
    expect(parsed.level).toBe("info");
    expect(parsed.hook).toBe("chat.params");
    expect(parsed.sessionID).toBe("s-json");
    expect(parsed.message).toBe("context_discovered");
    expect(parsed.data).toEqual({ contextWindow: 1000, thresholdTokens: 660 });

    await rm(logDir, { recursive: true, force: true });
  });

  it("handles burst async writes without losing entries", async () => {
    const logDir = await mkdtemp(path.join(os.tmpdir(), "pp-logger-"));
    const logger = createHookLogger({ level: "debug", logDir });

    for (let i = 0; i < 50; i += 1) {
      logger.debug("event", `burst-${i}`);
    }
    await logger.flush();

    const day = new Date().toISOString().slice(0, 10);
    const content = await readFile(path.join(logDir, `${day}.jsonl`), "utf8");
    const lines = content.trim().split("\n");

    expect(lines).toHaveLength(50);
    expect(lines[0]).toContain("burst-0");
    expect(lines[49]).toContain("burst-49");

    await rm(logDir, { recursive: true, force: true });
  });
});
