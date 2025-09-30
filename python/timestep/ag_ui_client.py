import asyncio
import os
import sys
from uuid import uuid4
from ag_ui.core import RunAgentInput, UserMessage, EventType
from ag_ui_server import TimestepAgent


async def chat_loop():
    """Interactive chat loop for AG-UI client."""
    print("🤖 Timestep Assistant started!")
    print("Type your messages and press Enter. Press Ctrl+D to quit.\n")

    model_id = os.getenv("MODEL_ID") or "ollama/gpt-oss:120b-cloud"
    openai_use_responses = (
        os.getenv("OPENAI_USE_RESPONSES", "false").lower() == "true"
    )

    agent = TimestepAgent(model_id, openai_use_responses)

    messages = []
    thread_id = str(uuid4())

    while True:
        try:
            # Get user input
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("> ")
            )

            if not user_input.strip():
                continue

            print()

            # Add user message to conversation
            user_message = UserMessage(
                id=str(uuid4()), role="user", content=user_input.strip()
            )
            messages.append(user_message)

            # Create run input for AG-UI
            run_input = RunAgentInput(
                thread_id=thread_id,
                run_id=str(uuid4()),
                messages=messages,
                state={},
                tools=[],
                context=[],
                forwarded_props={},
            )

            # Stream events from the agent
            has_started_message = False

            async for event in agent.run(run_input):
                if event.type == EventType.TEXT_MESSAGE_START:
                    if not has_started_message:
                        sys.stdout.write("🤖 Assistant: ")
                        sys.stdout.flush()
                        has_started_message = True

                elif event.type == EventType.TEXT_MESSAGE_CHUNK:
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()

                elif event.type == EventType.TEXT_MESSAGE_END:
                    # Message complete
                    pass

                elif event.type == EventType.TOOL_CALL_START:
                    sys.stdout.write(f"\n🔧 Calling tool: {event.tool_call_name}\n")
                    sys.stdout.flush()

                elif event.type == EventType.TOOL_CALL_ARGS:
                    sys.stdout.write(f"   Arguments: {event.delta}\n")
                    sys.stdout.flush()

                elif event.type == EventType.TOOL_CALL_RESULT:
                    sys.stdout.write(f"   Result: {event.content}\n")
                    sys.stdout.flush()

                elif event.type == EventType.TOOL_CALL_END:
                    sys.stdout.write("✅ Tool call complete\n\n")
                    sys.stdout.flush()

            if has_started_message:
                print("\n")

        except EOFError:
            # Ctrl+D pressed
            print("\n👋 Thanks for using AG-UI Assistant!")
            break
        except KeyboardInterrupt:
            # Ctrl+C pressed
            print("\n👋 Thanks for using AG-UI Assistant!")
            break
        except Exception as error:
            print(f"❌ Error: {error}")
            continue


async def main():
    await chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
