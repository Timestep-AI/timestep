/** Orchestration script for TypeScript -> Python cross-language tests. */

import { test } from 'vitest';
import { runAgentTestPartial } from './test_helpers';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_cross_language_ts_to_py_blocking_non_streaming with %s', async (model) => {
  const result = await runAgentTestPartial(false, false, undefined, 0, 4, model);
  const pythonTestName = 'test_cross_language_ts_to_py_blocking_non_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
  if (result.connectionString) {
    env.PG_CONNECTION_URI = result.connectionString;
  }
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_cross_language_ts_to_py_blocking_streaming with %s', async (model) => {
  const result = await runAgentTestPartial(false, true, undefined, 0, 4, model);
  const pythonTestName = 'test_cross_language_ts_to_py_blocking_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
  if (result.connectionString) {
    env.PG_CONNECTION_URI = result.connectionString;
  }
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_cross_language_ts_to_py_parallel_non_streaming with %s', async (model) => {
  const result = await runAgentTestPartial(true, false, undefined, 0, 4, model);
  const pythonTestName = 'test_cross_language_ts_to_py_parallel_non_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
  if (result.connectionString) {
    env.PG_CONNECTION_URI = result.connectionString;
  }
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"]])('test_cross_language_ts_to_py_parallel_streaming with %s', async (model) => {
  const result = await runAgentTestPartial(true, true, undefined, 0, 4, model);
  const pythonTestName = 'test_cross_language_ts_to_py_parallel_streaming';
  const pythonDir = path.join(__dirname, '../../python');
  const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
  
  const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
  if (result.connectionString) {
    env.PG_CONNECTION_URI = result.connectionString;
  }
  
  const { stdout, stderr } = await execAsync(pythonTestCmd, {
    cwd: pythonDir,
    env
  });
  if (stderr && !stderr.includes('PytestWarning')) {
    console.error(stderr);
  }
});

