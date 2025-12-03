/** Same-language test: TypeScript -> TypeScript (using cross-language pattern).
 * This tests if the issue is cross-language state loading or resuming from state with sessions.
 */

import { runAgentTestPartial, runAgentTestFromPython, cleanItems, assertConversationItems, EXPECTED_ITEMS } from './test_run_agent';

async function runTest(runInParallel: boolean, stream: boolean): Promise<void> {
  const testName = `test_same_language_ts_to_ts_${runInParallel ? 'parallel' : 'blocking'}_${stream ? 'streaming' : 'non_streaming'}`;
  try {
    console.log(`Running test: ${testName}`);
    
    // Step 1: Run TypeScript partial test (inputs 0-3) which stops at interruption
    const sessionId = await runAgentTestPartial(runInParallel, stream, undefined, 0, 4);
    console.log(`TypeScript test completed, session ID: ${sessionId}`);
    
    // Step 2: Resume in TypeScript (instead of Python) using the same pattern as cross-language
    // This simulates what Python does: load state and resume
    const items = await runAgentTestFromPython(runInParallel, stream, sessionId);
    const cleaned = cleanItems(items);
    
    // Items should match exactly - use assertConversationItems which handles key ordering
    if (cleaned.length !== EXPECTED_ITEMS.length) {
      console.log('\n' + '='.repeat(80));
      console.log('SAME-LANGUAGE TEST MISMATCH DETECTED');
      console.log('='.repeat(80));
      console.log(`Got ${cleaned.length} items, expected ${EXPECTED_ITEMS.length} items\n`);
      
      // Log detailed differences
      const maxLen = Math.max(cleaned.length, EXPECTED_ITEMS.length);
      for (let i = 0; i < Math.min(maxLen, 25); i++) {
        console.log(`\n--- Position ${i} ---`);
        if (i < cleaned.length) {
          console.log(`ACTUAL:   ${JSON.stringify(cleaned[i], null, 2)}`);
        } else {
          console.log(`ACTUAL:   <missing>`);
        }
        
        if (i < EXPECTED_ITEMS.length) {
          console.log(`EXPECTED: ${JSON.stringify(EXPECTED_ITEMS[i], null, 2)}`);
        } else {
          console.log(`EXPECTED: <missing>`);
        }
      }
      
      throw new Error(`Same-language test failed: items don't match. Got ${cleaned.length} items, expected ${EXPECTED_ITEMS.length} items.`);
    }
    
    // Use assertConversationItems which handles key ordering differences
    assertConversationItems(cleaned, EXPECTED_ITEMS);
    console.log(`✓ ${testName} passed`);
  } catch (error: any) {
    console.error(`✗ ${testName} failed:`, error.message);
    if (error.stack) {
      console.error(error.stack);
    }
    console.error('\nTests failed!');
    process.exit(1);
  }
}

(async () => {
  await runTest(false, false);
  await runTest(false, true);
  await runTest(true, false);
  await runTest(true, true);
  console.log('\nAll tests passed!');
})();

