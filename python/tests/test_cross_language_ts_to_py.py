"""Cross-language tests: TypeScript -> Python state persistence."""

import pytest
import os
import json
import logging
from test_run_agent import (
    run_agent_test_from_typescript,
    clean_items,
    assert_conversation_items,
    EXPECTED_ITEMS,
)

# Enable DEBUG logging to see cross-language debug logs
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openai.agents").setLevel(logging.DEBUG)


def get_session_id_from_env() -> str:
    """Get session ID from environment variable."""
    session_id = os.environ.get('CROSS_LANG_SESSION_ID')
    if not session_id:
        raise ValueError("Session ID not provided. CROSS_LANG_SESSION_ID environment variable must be set.")
    return session_id


def log_item_differences(cleaned, expected, max_items=None):
    """Log detailed differences between actual and expected items."""
    print(f"\n{'='*80}")
    print(f"CROSS-LANGUAGE TEST MISMATCH DETECTED")
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
            print(f"ACTUAL:   {json.dumps(actual_item, indent=2)}")
        else:
            print(f"ACTUAL:   <missing>")
        
        if i < len(expected):
            expected_item = expected[i]
            print(f"EXPECTED: {json.dumps(expected_item, indent=2)}")
        else:
            print(f"EXPECTED: <missing>")


@pytest.mark.asyncio
async def test_cross_language_ts_to_py_blocking_non_streaming():
    """Test TypeScript -> Python: blocking, non-streaming."""
    session_id = get_session_id_from_env()
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=False, stream=False)
    cleaned = clean_items(items)
    
    # Items should match exactly - if they don't, log and fail
    if len(cleaned) != len(EXPECTED_ITEMS) or cleaned != EXPECTED_ITEMS:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        pytest.fail(f"Cross-language test failed: items don't match. Got {len(cleaned)} items, expected {len(EXPECTED_ITEMS)} items.")
    
    assert_conversation_items(cleaned, EXPECTED_ITEMS)


@pytest.mark.asyncio
async def test_cross_language_ts_to_py_blocking_streaming():
    """Test TypeScript -> Python: blocking, streaming."""
    session_id = get_session_id_from_env()
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=False, stream=True)
    cleaned = clean_items(items)
    
    if len(cleaned) != len(EXPECTED_ITEMS) or cleaned != EXPECTED_ITEMS:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        pytest.fail(f"Cross-language test failed: items don't match. Got {len(cleaned)} items, expected {len(EXPECTED_ITEMS)} items.")
    
    assert_conversation_items(cleaned, EXPECTED_ITEMS)


@pytest.mark.asyncio
async def test_cross_language_ts_to_py_parallel_non_streaming():
    """Test TypeScript -> Python: parallel, non-streaming."""
    session_id = get_session_id_from_env()
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=True, stream=False)
    cleaned = clean_items(items)
    
    if len(cleaned) != len(EXPECTED_ITEMS) or cleaned != EXPECTED_ITEMS:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        pytest.fail(f"Cross-language test failed: items don't match. Got {len(cleaned)} items, expected {len(EXPECTED_ITEMS)} items.")
    
    assert_conversation_items(cleaned, EXPECTED_ITEMS)


@pytest.mark.asyncio
async def test_cross_language_ts_to_py_parallel_streaming():
    """Test TypeScript -> Python: parallel, streaming."""
    session_id = get_session_id_from_env()
    items = await run_agent_test_from_typescript(session_id=session_id, run_in_parallel=True, stream=True)
    cleaned = clean_items(items)
    
    if len(cleaned) != len(EXPECTED_ITEMS) or cleaned != EXPECTED_ITEMS:
        log_item_differences(cleaned, EXPECTED_ITEMS)
        pytest.fail(f"Cross-language test failed: items don't match. Got {len(cleaned)} items, expected {len(EXPECTED_ITEMS)} items.")
    
    assert_conversation_items(cleaned, EXPECTED_ITEMS)

