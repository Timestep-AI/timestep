/** Cross-language tests: Python -> TypeScript state persistence.
 * This script is called by the Python orchestration script after Python saves state.
 * It receives the session ID as a command line argument.
 */

import { runAgentTestFromPython, cleanItems, assertConversationItems, EXPECTED_ITEMS } from './test_helpers';

function logItemDifferences(cleaned: any[], expected: any[], maxItems: number = 25): void {
  console.log('\n' + '='.repeat(80));
  console.log('CROSS-LANGUAGE TEST MISMATCH DETECTED');
  console.log('='.repeat(80));
  console.log(`Got ${cleaned.length} items, expected ${expected.length} items\n`);
  
  // Log item types comparison
  const actualTypes = cleaned.map(item => item.type || 'unknown');
  const expectedTypes = expected.map(item => item.type || 'unknown');
  console.log(`Actual item types:  ${actualTypes.join(', ')}`);
  console.log(`Expected item types: ${expectedTypes.join(', ')}\n`);
  
  // Log detailed comparison for each position
  const maxLen = Math.max(cleaned.length, expected.length);
  for (let i = 0; i < Math.min(maxLen, maxItems); i++) {
    console.log(`\n--- Position ${i} ---`);
    if (i < cleaned.length) {
      console.log(`ACTUAL:   ${JSON.stringify(cleaned[i], null, 2)}`);
    } else {
      console.log(`ACTUAL:   <missing>`);
    }
    
    if (i < expected.length) {
      console.log(`EXPECTED: ${JSON.stringify(expected[i], null, 2)}`);
    } else {
      console.log(`EXPECTED: <missing>`);
    }
  }
}

async function runTest(runInParallel: boolean, stream: boolean, sessionId: string): Promise<void> {
  const testName = `test_cross_language_py_to_ts_${runInParallel ? 'parallel' : 'blocking'}_${stream ? 'streaming' : 'non_streaming'}`;
  try {
    console.log(`Running test: ${testName}`);
    const items = await runAgentTestFromPython(runInParallel, stream, sessionId);
    const cleaned = cleanItems(items);
    
    if (cleaned.length !== EXPECTED_ITEMS.length) {
      logItemDifferences(cleaned, EXPECTED_ITEMS);
      throw new Error(`Cross-language test failed: items don't match. Got ${cleaned.length} items, expected ${EXPECTED_ITEMS.length} items.`);
    }

    try {
      assertConversationItems(cleaned, EXPECTED_ITEMS);
    } catch (error: any) {
      logItemDifferences(cleaned, EXPECTED_ITEMS);
      throw error;
    }
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

// Get test variant and session ID from command line args
const args = process.argv.slice(2);
if (args.length >= 3) {
  const runInParallel = args[0] === 'true';
  const stream = args[1] === 'true';
  const sessionId = args[2];
  (async () => {
    await runTest(runInParallel, stream, sessionId);
  })();
} else {
  console.error('Usage: npx tsx test_cross_language_py_to_ts.ts <runInParallel> <stream> <sessionId>');
  process.exit(1);
}
