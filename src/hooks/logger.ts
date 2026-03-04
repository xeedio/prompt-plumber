import { appendFile, mkdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import type { LogLevel } from "../config.js";

type LogEntry = {
  timestamp: string;
  level: LogLevel;
  hook: string;
  sessionID?: string;
  message: string;
  data?: unknown;
};

const LOG_LEVEL_ORDER: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

function defaultLogDir(): string {
  return path.join(os.homedir(), ".config", "opencode", "prompt-plumber-logs");
}

function asLogLevel(value: unknown): LogLevel {
  if (value === "debug" || value === "info" || value === "warn" || value === "error") {
    return value;
  }
  return "info";
}

function toDateString(timestamp: string): string {
  return timestamp.slice(0, 10);
}

export type HookLoggerInput = {
  level?: LogLevel;
  logDir?: string;
};

export type HookLogger = {
  debug(hook: string, message: string, input?: { sessionID?: string; data?: unknown }): void;
  info(hook: string, message: string, input?: { sessionID?: string; data?: unknown }): void;
  warn(hook: string, message: string, input?: { sessionID?: string; data?: unknown }): void;
  error(hook: string, message: string, input?: { sessionID?: string; data?: unknown }): void;
  flush(): Promise<void>;
};

export function createHookLogger(input: HookLoggerInput = {}): HookLogger {
  const minLevel = asLogLevel(input.level ?? process.env.PROMPT_PLUMBER_LOG_LEVEL);
  const logDir = input.logDir ?? process.env.PROMPT_PLUMBER_LOG_DIR ?? defaultLogDir();

  let queue = Promise.resolve();
  let initialized = false;

  const ensureDir = async () => {
    if (initialized) return;
    await mkdir(logDir, { recursive: true });
    initialized = true;
  };

  const write = (entry: LogEntry) => {
    if (LOG_LEVEL_ORDER[entry.level] < LOG_LEVEL_ORDER[minLevel]) {
      return;
    }

    const day = toDateString(entry.timestamp);
    const filepath = path.join(logDir, `${day}.jsonl`);
    const line = JSON.stringify(entry) + "\n";

    queue = queue
      .then(async () => {
        await ensureDir();
        await appendFile(filepath, line, "utf8");
      })
      .catch(() => {});
  };

  const log = (level: LogLevel, hook: string, message: string, input?: { sessionID?: string; data?: unknown }) => {
    write({
      timestamp: new Date().toISOString(),
      level,
      hook,
      sessionID: input?.sessionID,
      message,
      data: input?.data,
    });
  };

  return {
    debug(hook, message, input) {
      log("debug", hook, message, input);
    },
    info(hook, message, input) {
      log("info", hook, message, input);
    },
    warn(hook, message, input) {
      log("warn", hook, message, input);
    },
    error(hook, message, input) {
      log("error", hook, message, input);
    },
    flush() {
      return queue;
    },
  };
}
