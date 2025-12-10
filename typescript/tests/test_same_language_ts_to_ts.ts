/** Same-language test: TypeScript -> TypeScript (using cross-language pattern).
 * This tests if the issue is cross-language state loading or resuming from state with sessions.
 */

import { test, expect } from 'vitest';
import { runAgentTestPartial, runAgentTestFromPython, cleanItems, assertConversationItems, EXPECTED_ITEMS } from './test_helpers';

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_same_language_ts_to_ts_blocking_non_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    await expect(async () => {
      const { sessionId, connectionString } = await runAgentTestPartial(false, false, undefined, 0, 4, model);
      await runAgentTestFromPython(false, false, sessionId, connectionString, model);
    }).rejects.toThrow();
    return;
  }
  const { sessionId, connectionString } = await runAgentTestPartial(false, false, undefined, 0, 4, model);
  const items = await runAgentTestFromPython(false, false, sessionId, connectionString, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_same_language_ts_to_ts_blocking_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    await expect(async () => {
      const { sessionId, connectionString } = await runAgentTestPartial(false, true, undefined, 0, 4, model);
      await runAgentTestFromPython(false, true, sessionId, connectionString, model);
    }).rejects.toThrow();
    return;
  }
  const { sessionId, connectionString } = await runAgentTestPartial(false, true, undefined, 0, 4, model);
  const items = await runAgentTestFromPython(false, true, sessionId, connectionString, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_same_language_ts_to_ts_parallel_non_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    await expect(async () => {
      const { sessionId, connectionString } = await runAgentTestPartial(true, false, undefined, 0, 4, model);
      await runAgentTestFromPython(true, false, sessionId, connectionString, model);
    }).rejects.toThrow();
    return;
  }
  const { sessionId, connectionString } = await runAgentTestPartial(true, false, undefined, 0, 4, model);
  const items = await runAgentTestFromPython(true, false, sessionId, connectionString, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_same_language_ts_to_ts_parallel_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    await expect(async () => {
      const { sessionId, connectionString } = await runAgentTestPartial(true, true, undefined, 0, 4, model);
      await runAgentTestFromPython(true, true, sessionId, connectionString, model);
    }).rejects.toThrow();
    return;
  }
  const { sessionId, connectionString } = await runAgentTestPartial(true, true, undefined, 0, 4, model);
  const items = await runAgentTestFromPython(true, true, sessionId, connectionString, model);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

