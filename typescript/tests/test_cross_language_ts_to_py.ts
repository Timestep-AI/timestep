/** Orchestration script for TypeScript -> Python cross-language tests. */

import { runAgentTestPartial } from './test_run_agent';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function runTest(name: string, runInParallel: boolean, stream: boolean): Promise<void> {
  try {
    console.log(`Running test: ${name}`);
    
    // Step 1: Run TypeScript partial test (inputs 0-3) which stops at interruption
    const sessionId = await runAgentTestPartial(runInParallel, stream, undefined, 0, 4);
    console.log(`TypeScript test completed, session ID: ${sessionId}`);
    
    // Step 2: Run Python test that loads the state and continues, passing session ID via environment variable
    const pythonTestName = `test_cross_language_ts_to_py_${runInParallel ? 'parallel' : 'blocking'}_${stream ? 'streaming' : 'non_streaming'}`;
    const pythonDir = path.join(__dirname, '../../python');
    const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
    
    console.log(`Running Python test: ${pythonTestName}`);
    const { stdout, stderr } = await execAsync(pythonTestCmd, {
      cwd: pythonDir,
      env: { ...process.env, CROSS_LANG_SESSION_ID: sessionId }
    });
    console.log(stdout);
    if (stderr && !stderr.includes('PytestWarning')) {
      console.error(stderr);
    }
    
    console.log(`✓ ${name} passed`);
  } catch (error: any) {
    console.error(`✗ ${name} failed:`, error.message);
    if (error.stdout) console.error('stdout:', error.stdout);
    if (error.stderr) console.error('stderr:', error.stderr);
    console.error('\nTests failed!');
    process.exit(1);
  }
}

(async () => {
  await runTest('test_cross_language_ts_to_py_blocking_non_streaming', false, false);
  await runTest('test_cross_language_ts_to_py_blocking_streaming', false, true);
  await runTest('test_cross_language_ts_to_py_parallel_non_streaming', true, false);
  await runTest('test_cross_language_ts_to_py_parallel_streaming', true, true);
  console.log('\nAll tests passed!');
})();

