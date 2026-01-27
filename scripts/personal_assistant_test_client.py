# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "a2a-sdk",
#   "httpx",
#   "openai",
# ]
# ///

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

# Fix package import: lib/python/ contains the timestep package
# but Python needs to import it as 'timestep', not 'python'
script_dir = Path(__file__).parent
lib_dir = script_dir.parent / "lib"
lib_python_dir = lib_dir / "python"

# Add lib/python to path
if str(lib_python_dir) not in sys.path:
    sys.path.insert(0, str(lib_python_dir))

# Create a 'timestep' module that points to the python directory
# This allows imports like 'from timestep.utils import ...' to work
import types
timestep_module = types.ModuleType('timestep')
timestep_module.__path__ = [str(lib_python_dir)]
sys.modules['timestep'] = timestep_module

# Create timestep.utils namespace
import types
utils_module = types.ModuleType('timestep.utils')
utils_module.__path__ = [str(lib_python_dir / "utils")]
sys.modules['timestep.utils'] = utils_module

# Load event_helpers module directly (avoids loading __init__.py which requires mcp)
import importlib.util
event_helpers_path = lib_python_dir / "utils" / "event_helpers.py"
if event_helpers_path.exists():
    spec = importlib.util.spec_from_file_location("timestep.utils.event_helpers", event_helpers_path)
    event_helpers_module = importlib.util.module_from_spec(spec)
    sys.modules['timestep.utils.event_helpers'] = event_helpers_module
    spec.loader.exec_module(event_helpers_module)

from a2a.client import ClientFactory
from timestep.utils.event_helpers import extract_event_data
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Get a logger instance

    base_url = 'http://localhost:9999'

    # Use ClientFactory (supports JSON-RPC by default)
    client = await ClientFactory.connect(base_url)
    logger.info('Client initialized.')

    send_message_payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': "What's the weather in Oakland?"}
            ],
            'messageId': uuid4().hex,
        },
    }

    logger.info('\n=== start:send_message ===')
    # Client.send_message expects just the message object, not SendMessageRequest
    message_obj = send_message_payload['message']

    async for event in client.send_message(message_obj):
        # Extract event data (handles both tuple and direct event objects)
        event_data = extract_event_data(event)
        
        # Print event data
        if hasattr(event_data, 'model_dump'):
            print(event_data.model_dump(mode='json', exclude_none=True))
        else:
            print(event_data)
    logger.info('=== end:send_message ===\n')
    # --8<-- [end:send_message]

    # --8<-- [start:send_message_streaming]

    logger.info('=== start:send_message_streaming ===')
    # Note: Client.send_message() already returns an async generator (streaming)
    # There is no separate send_message_streaming method in the new Client API
    # send_message() already streams events as they arrive
    
    async for event in client.send_message(message_obj):
        # Extract event data (handles both tuple and direct event objects)
        event_data = extract_event_data(event)
        
        # Print event data
        if hasattr(event_data, 'model_dump'):
            print(event_data.model_dump(mode='json', exclude_none=True))
        else:
            print(event_data)
    logger.info('=== end:send_message_streaming ===')
    # --8<-- [end:send_message_streaming]

    # --8<-- [start:responses_non_streaming]
    logger.info('\n=== start:responses_non_streaming ===')
    openai_client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="dummy-key"  # Not used for local endpoint
    )
    
    response = await openai_client.responses.create(
        model="gpt-4o-mini",
        input="What's the weather in Oakland?"
    )
    print(response.model_dump_json(indent=2, exclude_none=True))
    logger.info('=== end:responses_non_streaming ===\n')
    # --8<-- [end:responses_non_streaming]

    # --8<-- [start:responses_streaming]
    logger.info('=== start:responses_streaming ===')
    openai_client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="dummy-key"  # Not used for local endpoint
    )
    
    stream = await openai_client.responses.create(
        model="gpt-4o-mini",
        input="What's the weather in Oakland?",
        stream=True
    )
    
    async for chunk in stream:
        print(chunk.model_dump_json(indent=2, exclude_none=True))
    logger.info('=== end:responses_streaming ===')
    # --8<-- [end:responses_streaming]


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
