/** Same-language test: TypeScript -> TypeScript (using cross-language pattern).
 * This tests if the issue is cross-language state loading or resuming from state with sessions.
 */

import { test } from 'vitest';
import { runAgentTestPartial, runAgentTestFromPython, cleanItems, assertConversationItems, EXPECTED_ITEMS } from './test_helpers';

test('test_same_language_ts_to_ts_blocking_non_streaming', async () => {
  const { sessionId, connectionString } = await runAgentTestPartial(false, false, undefined, 0, 4);
  const items = await runAgentTestFromPython(false, false, sessionId, connectionString);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test('test_same_language_ts_to_ts_blocking_streaming', async () => {
  const { sessionId, connectionString } = await runAgentTestPartial(false, true, undefined, 0, 4);
  const items = await runAgentTestFromPython(false, true, sessionId, connectionString);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test('test_same_language_ts_to_ts_parallel_non_streaming', async () => {
  const { sessionId, connectionString } = await runAgentTestPartial(true, false, undefined, 0, 4);
  const items = await runAgentTestFromPython(true, false, sessionId, connectionString);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

test('test_same_language_ts_to_ts_parallel_streaming', async () => {
  const { sessionId, connectionString } = await runAgentTestPartial(true, true, undefined, 0, 4);
  const items = await runAgentTestFromPython(true, true, sessionId, connectionString);
  const cleaned = cleanItems(items);
  assertConversationItems(cleaned, EXPECTED_ITEMS);
});

