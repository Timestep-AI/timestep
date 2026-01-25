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

import httpx
from openai import OpenAI

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

    # --8<-- [start:A2ACardResolver]

    base_url = 'http://localhost:9999'

    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        # --8<-- [end:A2ACardResolver]

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = (
                await resolver.get_agent_card()
            )  # Fetches from default public path
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        f'\nPublic card supports authenticated extended card. Attempting to fetch from: {base_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = (
                        _extended_card  # Update to use the extended card
                    )
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
                        exc_info=True,
                    )
            elif (
                _public_card
            ):  # supports_authenticated_extended_card is False or None
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        # --8<-- [start:send_message]
        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
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

        # --8<-- [start:openai_chat_completions]
        logger.info('\n=== start:openai_chat_completions ===')
        
        # Same message payload that was sent to A2A agent
        message_content = "What's the weather in Oakland?"
        
        openai_client = OpenAI(
            api_key="dummy-api-key",
            base_url="http://localhost:9999/v1",
        )
        
        logger.info('=== start:openai_chat_completions (non-streaming) ===')
        response = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "user", "content": message_content}
                ],
                stream=False,
            )
        )
        
        print(response.model_dump_json(indent=2, exclude_none=True))
        logger.info('=== end:openai_chat_completions (non-streaming) ===\n')
        # --8<-- [end:openai_chat_completions (non-streaming)]
        
        # --8<-- [start:openai_streaming]
        logger.info('=== start:openai_chat_completions (streaming) ===')
        
        stream = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "user", "content": message_content}
                ],
                stream=True,
            )
        )
        
        # Process stream in async context
        async def process_stream():
            for chunk in stream:
                print(chunk.model_dump_json(indent=2, exclude_none=True))
        
        await process_stream()
        
        logger.info('=== end:openai_chat_completions (streaming) ===')
        # --8<-- [end:openai_streaming]
        
        # --8<-- [start:openai_responses]
        logger.info('\n=== start:openai_responses ===')
        
        # Same message payload that was sent to A2A agent
        message_content = "What's the weather in Oakland?"
        
        logger.info('=== start:openai_responses (non-streaming) ===')
        # Use httpx to call our custom /v1/responses endpoint
        response = await httpx_client.post(
            "http://localhost:9999/v1/responses",
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "user", "content": message_content}
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        response_data = response.json()
        print(json.dumps(response_data, indent=2))
        logger.info('=== end:openai_responses (non-streaming) ===\n')
        # --8<-- [end:openai_responses (non-streaming)]
        
        # --8<-- [start:openai_responses_streaming]
        logger.info('=== start:openai_responses (streaming) ===')
        
        # Use httpx to call our custom /v1/responses endpoint with streaming
        async with httpx_client.stream(
            "POST",
            "http://localhost:9999/v1/responses",
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "user", "content": message_content}
                ],
                "stream": True,
            },
        ) as stream_response:
            stream_response.raise_for_status()
            async for line in stream_response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        print(json.dumps(chunk, indent=2))
                    except json.JSONDecodeError:
                        pass
        
        logger.info('=== end:openai_responses (streaming) ===')
        # --8<-- [end:openai_responses_streaming]


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
