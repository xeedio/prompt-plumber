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
});
