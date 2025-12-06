/** Orchestration script for TypeScript -> Python cross-language tests. */

import { test } from 'vitest';
import { runAgentTestPartial } from './test_run_agent';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('test_cross_language_ts_to_py_blocking_non_streaming', async () => {
  const sessionId = await runAgentTestPartial(false, false, undefined, 0, 4);
  const pythonTestName = 'test_cross_language_ts_to_py_blocking_non_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env: { ...process.env, CROSS_LANG_SESSION_ID: sessionId }
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

test('test_cross_language_ts_to_py_blocking_streaming', async () => {
  const sessionId = await runAgentTestPartial(false, true, undefined, 0, 4);
  const pythonTestName = 'test_cross_language_ts_to_py_blocking_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env: { ...process.env, CROSS_LANG_SESSION_ID: sessionId }
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

test('test_cross_language_ts_to_py_parallel_non_streaming', async () => {
  const sessionId = await runAgentTestPartial(true, false, undefined, 0, 4);
  const pythonTestName = 'test_cross_language_ts_to_py_parallel_non_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env: { ...process.env, CROSS_LANG_SESSION_ID: sessionId }
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

test('test_cross_language_ts_to_py_parallel_streaming', async () => {
  const sessionId = await runAgentTestPartial(true, true, undefined, 0, 4);
  const pythonTestName = 'test_cross_language_ts_to_py_parallel_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env: { ...process.env, CROSS_LANG_SESSION_ID: sessionId }
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

