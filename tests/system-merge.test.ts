import { describe, expect, it } from "vitest";

import { mergeSystemMessages } from "../src/hooks/system-merge.js";

describe("mergeSystemMessages", () => {
  it("returns empty array for no input", () => {
    expect(mergeSystemMessages([])).toEqual([]);
  });

  it("returns a single trimmed message unchanged", () => {
    expect(mergeSystemMessages(["  one message  "])).toEqual(["one message"]);
  });

  it("merges two messages with double newline", () => {
    expect(mergeSystemMessages(["alpha", "beta"])).toEqual(["alpha\n\nbeta"]);
  });

  it("merges three messages and drops whitespace-only entries", () => {
    expect(mergeSystemMessages(["  first", "", "   ", "second", "third  "])).toEqual([
      "first\n\nsecond\n\nthird",
    ]);
  });
});
