"""Command-line interface for running agents via A2A client."""

import argparse
import asyncio
import logging
from typing import Any
from uuid import uuid4

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
    """Main CLI entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an agent via A2A client",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:9999",
        help="Base URL of the A2A agent server (default: http://localhost:9999)"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Message to send to the agent"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Use streaming mode (default: True)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        help="Authorization token for extended agent card"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging to show INFO level messages
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)  # Get a logger instance

    # --8<-- [start:A2ACardResolver]

    base_url = args.base_url

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
                if not args.auth_token:
                    raise ValueError("--auth-token is required when extended card is supported")
                
                logger.info(
                    f'\nPublic card supports authenticated extended card. Attempting to fetch from: {base_url}{EXTENDED_AGENT_CARD_PATH}'
                )
                auth_headers_dict = {'Authorization': f'Bearer {args.auth_token}'}
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
                final_agent_card_to_use = _extended_card
                logger.info(
                    '\nUsing AUTHENTICATED EXTENDED agent card for client initialization.'
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

        # Get message from user
        if args.message:
            message_text = args.message
        else:
            # Interactive mode
            try:
                message_text = input("You: ").strip()
                if not message_text:
                    logger.warning("Empty message, exiting.")
                    return
            except (EOFError, KeyboardInterrupt):
                logger.info("\nExiting...")
                return

        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message_text}
                ],
                'messageId': uuid4().hex,
            },
        }
        
        use_streaming = args.stream and not args.no_stream

        if use_streaming:
            # --8<-- [start:send_message_streaming]

            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            stream_response = client.send_message_streaming(streaming_request)

            async for chunk in stream_response:
                print(chunk.model_dump(mode='json', exclude_none=True))
            # --8<-- [end:send_message_streaming]
        else:
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            response = await client.send_message(request)
            print(response.model_dump(mode='json', exclude_none=True))
            # --8<-- [end:send_message]


def cli_main() -> None:
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == '__main__':
    cli_main()
