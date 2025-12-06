"""Same-language tests: Python -> Python (using cross-language pattern).
This tests if the issue is cross-language state loading or resuming from state with sessions.
"""

import pytest
import os
import json
import logging
from test_run_agent import (
    run_agent_test_partial,
    run_agent_test_from_typescript,
    clean_items,
    assert_conversation_items,
    EXPECTED_ITEMS,
    truncate_image_data,
)

# Enable DEBUG logging to see debug logs
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openai.agents").setLevel(logging.DEBUG)


def log_item_differences(cleaned, expected, max_items=None):
    """Log detailed differences between actual and expected items."""
    print(f"\n{'='*80}")
    print(f"SAME-LANGUAGE TEST MISMATCH DETECTED")
    print(f"{'='*80}")
    print(f"Got {len(cleaned)} items, expected {len(expected)} items\n")
    
    # Log item types comparison
    actual_types = [item.get('type', 'unknown') for item in cleaned]
    expected_types = [item.get('type', 'unknown') for item in expected]
    print(f"Actual item types:  {actual_types}")
    print(f"Expected item types: {expected_types}\n")
    
    # Log detailed comparison for each position - show ALL items
    max_len = max(len(cleaned), len(expected))
    if max_items is None:
        max_items = max_len
    for i in range(min(max_len, max_items)):
        print(f"\n--- Position {i} ---")
        if i < len(cleaned):
            actual_item = cleaned[i]
            truncated_actual = truncate_image_data(actual_item)
            print(f"ACTUAL:   {json.dumps(truncated_actual, indent=2)}")
        else:
            print(f"ACTUAL:   <missing>")
        
        if i < len(expected):
            expected_item = expected[i]
            truncated_expected = truncate_image_data(expected_item)
            print(f"EXPECTED: {json.dumps(truncated_expected, indent=2)}")
        else:
            print(f"EXPECTED: <missing>")


@pytest.mark.asyncio
async def test_same_language_py_to_py_blocking_non_streaming():
    """Test Python -> Python: blocking, non-streaming."""
    # Step 1: Run Python partial test (inputs 0-3) which stops at interruption
    # Explicitly pass session_id=None to ensure a fresh session for each test
    session_id = await run_agent_test_partial(run_in_parallel=False, stream=False, session_id=None, start_index=0, end_index=4)
    print(f"Python test completed, session ID: {session_id}")
    
    # Step 2: Resume in Python (instead of TypeScript) using the same pattern as cross-language
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=False, stream=False)
    cleaned = clean_items(items)
    
    try:
        assert_conversation_items(cleaned, EXPECTED_ITEMS)
    except AssertionError:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        raise


@pytest.mark.asyncio
async def test_same_language_py_to_py_blocking_streaming():
    """Test Python -> Python: blocking, streaming."""
    # Explicitly pass session_id=None to ensure a fresh session for each test
    session_id = await run_agent_test_partial(run_in_parallel=False, stream=True, session_id=None, start_index=0, end_index=4)
    print(f"Python test completed, session ID: {session_id}")
    
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=False, stream=True)
    cleaned = clean_items(items)
    
    try:
        assert_conversation_items(cleaned, EXPECTED_ITEMS)
    except AssertionError:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        raise


@pytest.mark.asyncio
async def test_same_language_py_to_py_parallel_non_streaming():
    """Test Python -> Python: parallel, non-streaming."""
    # Explicitly pass session_id=None to ensure a fresh session for each test
    session_id = await run_agent_test_partial(run_in_parallel=True, stream=False, session_id=None, start_index=0, end_index=4)
    print(f"Python test completed, session ID: {session_id}")
    
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=True, stream=False)
    cleaned = clean_items(items)
    
    try:
        assert_conversation_items(cleaned, EXPECTED_ITEMS)
    except AssertionError:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        raise


@pytest.mark.asyncio
async def test_same_language_py_to_py_parallel_streaming():
    """Test Python -> Python: parallel, streaming."""
    # Explicitly pass session_id=None to ensure a fresh session for each test
    session_id = await run_agent_test_partial(run_in_parallel=True, stream=True, session_id=None, start_index=0, end_index=4)
    print(f"Python test completed, session ID: {session_id}")
    
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=True, stream=True)
    cleaned = clean_items(items)
    
    try:
        assert_conversation_items(cleaned, EXPECTED_ITEMS)
    except AssertionError:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        raise

