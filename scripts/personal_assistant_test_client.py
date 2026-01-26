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

from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Get a logger instance

    base_url = 'http://localhost:9999'

    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        # Fetch agent card
        agent_card = await resolver.get_agent_card()
        logger.info('Successfully fetched agent card')
        
        # Use A2AClient for JSON-RPC agents (ClientFactory doesn't support JSON-RPC yet)
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        logger.info('A2AClient initialized.')

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
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)
        print(response.model_dump(mode='json', exclude_none=True))
        logger.info('=== end:send_message ===\n')
        # --8<-- [end:send_message]

        # --8<-- [start:send_message_streaming]

        logger.info('=== start:send_message_streaming ===')
        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        stream_response = client.send_message_streaming(streaming_request)

        async for chunk in stream_response:
            print(chunk.model_dump(mode='json', exclude_none=True))
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
