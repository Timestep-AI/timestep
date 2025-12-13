"""Orchestration script for Python -> TypeScript cross-language tests."""

import pytest
import subprocess
import sys
import os
from pathlib import Path
from test_run_agent import run_agent_test_partial


async def run_test(test_name: str, run_in_parallel: bool, stream: bool, model: str = "gpt-4.1"):
    """Run a single cross-language test."""
    print(f"Running test: {test_name} with model: {model}")
    
    # Step 1: Run Python partial test (inputs 0-3) which stops at interruption
    result = await run_agent_test_partial(
        run_in_parallel=run_in_parallel,
        stream=stream,
        session_id=None,
        start_index=0,
        end_index=4,
        model=model
    )
    
    # Handle both dict return (new format) and string return (old format for backwards compatibility)
    if isinstance(result, dict):
        session_id = result["session_id"]
        connection_string = result.get("connection_string")
    else:
        # Old format - just session_id string
        session_id = result
        connection_string = None
    
    print(f"Python test completed, session ID: {session_id}")
    if connection_string:
        print(f"Using connection string: {connection_string}")
    
    # Step 2: Run TypeScript test that loads the state and continues, passing session ID and model as parameters
    ts_test_path = Path(__file__).parent.parent.parent / "typescript" / "tests" / "test_cross_language_py_to_ts.ts"
    ts_dir = ts_test_path.parent.parent
    
    # Prepare environment with connection string if available
    env = {**os.environ}
    if connection_string:
        env["PG_CONNECTION_URI"] = connection_string
    
    print(f"Running TypeScript test: {run_in_parallel}, {stream}, model: {model}")
    print(f"Working directory: {ts_dir}")
    print(f"Test file: {ts_test_path}")
    # Use pnpm exec to ensure we use the locally installed tsx
    test_file_rel = ts_test_path.relative_to(ts_dir)
    print(f"Command: pnpm exec tsx {test_file_rel} {str(run_in_parallel).lower()} {str(stream).lower()} {session_id} {model}")
    
    result = subprocess.run(
        ["pnpm", "exec", "tsx", str(test_file_rel), str(run_in_parallel).lower(), str(stream).lower(), session_id, model],
        cwd=str(ts_dir),
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.stdout:
        print("=== TypeScript stdout ===")
        print(result.stdout)
    if result.stderr:
        print("=== TypeScript stderr ===", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        error_msg = f"TypeScript test failed with return code {result.returncode}"
        if result.stdout:
            error_msg += f"\n\nStdout:\n{result.stdout}"
        if result.stderr:
            error_msg += f"\n\nStderr:\n{result.stderr}"
        raise RuntimeError(error_msg)
    
    print(f"✓ {test_name} passed")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M", "openai/gpt-5.2"])
async def test_cross_language_py_to_ts_blocking_non_streaming(model):
    """Test Python -> TypeScript: blocking, non-streaming."""
    if model == "ollama/gpt-oss:20b-cloud" or model == "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_test("test_cross_language_py_to_ts_blocking_non_streaming", False, False, model)
        return
    await run_test("test_cross_language_py_to_ts_blocking_non_streaming", False, False, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M", "openai/gpt-5.2"])
async def test_cross_language_py_to_ts_blocking_streaming(model):
    """Test Python -> TypeScript: blocking, streaming."""
    if model == "ollama/gpt-oss:20b-cloud" or model == "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_test("test_cross_language_py_to_ts_blocking_streaming", False, True, model)
        return
    await run_test("test_cross_language_py_to_ts_blocking_streaming", False, True, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M", "openai/gpt-5.2"])
async def test_cross_language_py_to_ts_parallel_non_streaming(model):
    """Test Python -> TypeScript: parallel, non-streaming."""
    if model == "ollama/gpt-oss:20b-cloud" or model == "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_test("test_cross_language_py_to_ts_parallel_non_streaming", True, False, model)
        return
    await run_test("test_cross_language_py_to_ts_parallel_non_streaming", True, False, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1", "ollama/gpt-oss:20b-cloud", "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M", "openai/gpt-5.2"])
async def test_cross_language_py_to_ts_parallel_streaming(model):
    """Test Python -> TypeScript: parallel, streaming."""
    if model == "ollama/gpt-oss:20b-cloud" or model == "ollama/hf.co/mjschock/SmolVLM2-500M-Video-Instruct-GGUF:Q4_K_M":
        # Expected failure: Ollama cloud model has known compatibility issues
        with pytest.raises(Exception):
            await run_test("test_cross_language_py_to_ts_parallel_streaming", True, True, model)
        return
    await run_test("test_cross_language_py_to_ts_parallel_streaming", True, True, model)

