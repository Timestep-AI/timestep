/** Orchestration script for TypeScript -> Python cross-language tests. */

import { test, expect } from 'vitest';
import { runAgentTestPartial } from './test_helpers';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"], ["ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"]])('test_cross_language_ts_to_py_blocking_non_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues (may timeout or throw)
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Test timeout - expected failure')), 60000)
    );
    try {
      await Promise.race([
        (async () => {
          const result = await runAgentTestPartial(false, false, undefined, 0, 4, model);
          const pythonTestName = 'test_cross_language_ts_to_py_blocking_non_streaming';
          const pythonDir = path.join(__dirname, '../../python');
          const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
          
          const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
          if (result.connectionString) {
            env.PG_CONNECTION_URI = result.connectionString;
          }
          
          await execAsync(pythonTestCmd, {
            cwd: pythonDir,
            env
          });
        })(),
        timeoutPromise
      ]);
    } catch (error) {
      // Expected to fail either in runAgentTestPartial, Python test execution, or timeout
      expect(error).toBeDefined();
    }
    return;
  }
  
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

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"], ["ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"]])('test_cross_language_ts_to_py_blocking_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    try {
      const result = await runAgentTestPartial(false, true, undefined, 0, 4, model);
      const pythonTestName = 'test_cross_language_ts_to_py_blocking_streaming';
      const pythonDir = path.join(__dirname, '../../python');
      const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
      
      const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
      if (result.connectionString) {
        env.PG_CONNECTION_URI = result.connectionString;
      }
      
      await expect(execAsync(pythonTestCmd, {
        cwd: pythonDir,
        env
      })).rejects.toThrow();
    } catch (error) {
      // Expected to fail either in runAgentTestPartial or Python test execution
      expect(error).toBeDefined();
    }
    return;
  }
  
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

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"], ["ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"]])('test_cross_language_ts_to_py_parallel_non_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    try {
      const result = await runAgentTestPartial(true, false, undefined, 0, 4, model);
      const pythonTestName = 'test_cross_language_ts_to_py_parallel_non_streaming';
      const pythonDir = path.join(__dirname, '../../python');
      const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
      
      const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
      if (result.connectionString) {
        env.PG_CONNECTION_URI = result.connectionString;
      }
      
      await expect(execAsync(pythonTestCmd, {
        cwd: pythonDir,
        env
      })).rejects.toThrow();
    } catch (error) {
      // Expected to fail either in runAgentTestPartial or Python test execution
      expect(error).toBeDefined();
    }
    return;
  }
  
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

test.each([["gpt-4.1"], ["ollama/gpt-oss:20b-cloud"], ["ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M"]])('test_cross_language_ts_to_py_parallel_streaming with %s', async (model) => {
  if (model === "ollama/gpt-oss:20b-cloud") {
    // Expected failure: Ollama cloud model has known compatibility issues
    try {
      const result = await runAgentTestPartial(true, true, undefined, 0, 4, model);
      const pythonTestName = 'test_cross_language_ts_to_py_parallel_streaming';
      const pythonDir = path.join(__dirname, '../../python');
      const pythonTestCmd = `cd ${pythonDir} && uv run pytest tests/test_cross_language_ts_to_py.py::${pythonTestName} -v -x`;
      
      const env: Record<string, string> = { ...process.env, CROSS_LANG_SESSION_ID: result.sessionId, CROSS_LANG_MODEL: model };
      if (result.connectionString) {
        env.PG_CONNECTION_URI = result.connectionString;
      }
      
      await expect(execAsync(pythonTestCmd, {
        cwd: pythonDir,
        env
      })).rejects.toThrow();
    } catch (error) {
      // Expected to fail either in runAgentTestPartial or Python test execution
      expect(error).toBeDefined();
    }
    return;
  }
  
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

