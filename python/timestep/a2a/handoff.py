"""Handoff tool for agent-to-agent communication via A2A protocol."""

import json
import os
import re
import uuid
from typing import Any, Callable

import httpx


class HandoffCallbacks:
    """Callbacks for handoff tool."""

    def __init__(
        self,
        on_approval_required: Callable[[dict[str, Any]], Any] | None = None,
        on_child_message: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize callbacks.

        Args:
            on_approval_required: Optional callback for handling approvals from target agent
            on_child_message: Optional callback for handling child messages
        """
        self.on_approval_required = on_approval_required
        self.on_child_message = on_child_message


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse SSE line and extract JSON data.

    Args:
        line: SSE line starting with "data: "

    Returns:
        Parsed JSON data or None
    """
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None


def _extract_task_info(result: dict[str, Any]) -> dict[str, str | None]:
    """Extract task ID and context ID from SSE event.

    Args:
        result: SSE event result dictionary

    Returns:
        Dictionary with task_id and context_id
    """
    if result.get("kind") == "task" and result.get("id"):
        return {
            "task_id": result["id"],
            "context_id": result.get("context_id"),
        }
    elif result.get("id") and result.get("context_id"):
        return {
            "task_id": result["id"],
            "context_id": result["context_id"],
        }
    return {"task_id": None, "context_id": None}


async def _handle_child_approval(
    approval_message: dict[str, Any],
    message_endpoint: str,
    child_context_id: str,
    task_id: str,
    on_approval_required: Callable[[dict[str, Any]], Any],
) -> None:
    """Handle approval request from child agent.

    Args:
        approval_message: Approval message from child agent
        message_endpoint: Endpoint for sending approval response
        child_context_id: Child agent context ID
        task_id: Child agent task ID
        on_approval_required: Callback for approval handling
    """
    if "parts" not in approval_message:
        return

    approval_text = "".join(
        part.get("text", "")
        for part in approval_message["parts"]
        if part.get("kind") == "text" and part.get("text")
    )

    tool_match = re.search(r"Tool: (.+)", approval_text)
    args_match = re.search(r"Arguments: (.+)", approval_text, re.DOTALL)

    tool_name = tool_match.group(1).strip() if tool_match else "unknown"
    tool_args = args_match.group(1).strip() if args_match else "{}"

    synthetic_tool_call: dict[str, Any] = {
        "id": f"handoff-approval-{task_id}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": tool_args,
        },
    }

    approved = await on_approval_required(synthetic_tool_call)
    approval_response_text = "approve" if approved else "reject"
    approval_message_id = str(uuid.uuid4())
    approval_request_id = str(int(uuid.uuid4().int % 1e10))

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                message_endpoint,
                json={
                    "jsonrpc": "2.0",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "messageId": approval_message_id,
                            "role": "user",
                            "parts": [{"kind": "text", "text": approval_response_text}],
                            "contextId": child_context_id,
                            "taskId": task_id,
                        },
                    },
                    "id": approval_request_id,
                },
            )
    except Exception as error:
        print(f"[Handoff] Error sending approval: {error}")


async def _establish_parent_relationship(
    parent_context_id: str,
    child_context_id: str,
) -> None:
    """Establish parent-child relationship between contexts.

    Args:
        parent_context_id: Parent context ID
        child_context_id: Child context ID
    """
    try:
        a2a_server_url = os.environ.get("A2A_SERVER_URL", "http://localhost:8080")
        update_url = f"{a2a_server_url}/contexts/{child_context_id}"

        print(
            f"[Handoff] Establishing parent relationship: {parent_context_id} -> {child_context_id}"
        )

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                update_url,
                json={"parent_context_id": parent_context_id},
            )

            if response.status_code != 200:
                print(
                    f"[Handoff] Failed to set parent relationship: {response.status_code} {response.text}"
                )
            else:
                print("[Handoff] Parent relationship established successfully")
    except Exception as error:
        print(f"[Handoff] Error establishing parent relationship: {error}")


async def _process_sse_stream(
    response: httpx.Response,
    args: dict[str, Any],
    callbacks: HandoffCallbacks,
    message_endpoint: str,
) -> dict[str, str]:
    """Process SSE stream from child agent and forward messages to parent.

    Args:
        response: HTTP response with SSE stream
        args: Handoff arguments
        callbacks: Handoff callbacks
        message_endpoint: Endpoint for sending messages

    Returns:
        Dictionary with task_id and child_context_id

    Raises:
        ValueError: If task ID cannot be found in stream
    """
    task_id: str | None = None
    child_context_id: str | None = None
    child_context_id_from_args = str(uuid.uuid4())  # Fallback if not found in stream

    buffer = ""
    async for line_bytes in response.aiter_lines():
        if not line_bytes:
            continue

        line = line_bytes.decode("utf-8")
        buffer += line

        lines = buffer.split("\n")
        buffer = lines.pop() if lines else ""

        for line_item in lines:
            data = _parse_sse_line(line_item)
            if not data or not isinstance(data, dict):
                continue

            if "result" not in data:
                continue

            result = data["result"]

            # Extract task ID and context ID from first event
            if not task_id:
                task_info = _extract_task_info(result)
                if task_info["task_id"]:
                    task_id = task_info["task_id"]
                    child_context_id = task_info["context_id"] or child_context_id_from_args
                    print(
                        f"[Handoff] Found task ID from SSE: {task_id}, context: {child_context_id}"
                    )

            # Handle status updates
            if (
                isinstance(result, dict)
                and result.get("kind") == "status-update"
                and result.get("status")
            ):
                status_result = result["status"]

                # Handle approvals from child agent
                if (
                    status_result.get("state") == "input-required"
                    and callbacks.on_approval_required
                    and message_endpoint
                    and child_context_id
                    and task_id
                ):
                    approval_message = status_result.get("message")
                    if approval_message:
                        await _handle_child_approval(
                            approval_message,
                            message_endpoint,
                            child_context_id,
                            task_id,
                            callbacks.on_approval_required,
                        )

                # Forward child messages to parent context
                if (
                    status_result.get("message")
                    and args.get("source_context_id")
                    and callbacks.on_child_message
                    and status_result.get("state") != "input-required"
                ):
                    child_message = status_result["message"]
                    if child_message.get("role") in ("agent", "tool"):
                        child_message_with_extras = child_message.copy()
                        print(
                            f"[Handoff] Forwarding child message to parent: {child_message.get('role')}",
                            {
                                "tool_name": child_message_with_extras.get("tool_name"),
                                "has_tool_calls": bool(
                                    child_message_with_extras.get("tool_calls")
                                ),
                            },
                        )
                        callbacks.on_child_message({
                            "kind": "message",
                            **child_message,
                            "contextId": args["source_context_id"],
                            "toolName": child_message_with_extras.get("tool_name"),
                            "tool_calls": child_message_with_extras.get("tool_calls"),
                        })

                # Establish parent relationship when child completes
                if (
                    status_result.get("state") == "completed"
                    and args.get("source_context_id")
                    and child_context_id
                ):
                    await _establish_parent_relationship(
                        args["source_context_id"], child_context_id
                    )

                # Stop when child completes or fails
                if status_result.get("state") in ("completed", "failed"):
                    print(
                        f"[Handoff] Child agent {status_result.get('state')}, stopping"
                    )
                    if not task_id or not child_context_id:
                        raise ValueError(
                            "Task ID or context ID missing when child completed"
                        )
                    return {"task_id": task_id, "child_context_id": child_context_id}

    if not task_id or not child_context_id:
        raise ValueError(
            "Failed to get task ID from target agent - task not found in SSE stream"
        )

    return {"task_id": task_id, "child_context_id": child_context_id}


async def handoff(
    args: dict[str, Any],
    callbacks: HandoffCallbacks | None = None,
) -> str:
    """Hand off a message to another agent via A2A protocol.

    Args:
        args: Handoff arguments with 'message', 'agent_id', and optional 'source_context_id'
        callbacks: Optional callbacks for handoff events

    Returns:
        Handoff initiation result as JSON string

    Raises:
        ValueError: If handoff fails
    """
    if callbacks is None:
        callbacks = HandoffCallbacks()

    agent_id = args.get("agent_id") or args.get("agentId", "")
    message = args.get("message", "")

    print(
        f"[Handoff] Starting handoff to agent: {agent_id}",
        {"has_approval_callback": callbacks.on_approval_required is not None},
    )

    try:
        # Get A2A server URL from environment, default to localhost:8080
        a2a_server_url = os.environ.get("A2A_SERVER_URL", "http://localhost:8080")
        agent_card_url = f"{a2a_server_url}/agents/{agent_id}/.well-known/agent-card.json"

        print(f"[Handoff] Connecting to agent card: {agent_card_url}")

        # Try to fetch the agent card first to verify connectivity
        try:
            async with httpx.AsyncClient() as client:
                card_response = await client.get(agent_card_url)
                if card_response.status_code != 200:
                    raise ValueError(
                        f"Failed to fetch agent card: {card_response.status_code} {card_response.text}"
                    )
                agent_card = card_response.json()
                print(
                    f"[Handoff] Agent card fetched successfully: {agent_card.get('name', 'Unknown')}"
                )
        except Exception as fetch_error:
            error_message = str(fetch_error) if fetch_error else "Unknown error"
            print(f"[Handoff] Failed to fetch agent card: {error_message}")
            raise ValueError(
                f"Failed to fetch agent card from {agent_card_url}: {error_message}"
            )

        # Create new context for child agent (allows parallel handoffs)
        child_context_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        # Return the same JSON that was passed in as arguments
        handoff_result = json.dumps({
            "message": message,
            "agentId": agent_id,
        })

        # Send message via direct HTTP POST to child agent
        agent_base_url = agent_card_url.replace("/.well-known/agent-card.json", "")
        message_endpoint = agent_base_url
        request_id = str(int(uuid.uuid4().int % 1e10))

        print(
            "[Handoff] Sending message to target agent via HTTP POST...",
            {"endpoint": message_endpoint, "child_context_id": child_context_id},
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                message_endpoint,
                json={
                    "jsonrpc": "2.0",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "messageId": message_id,
                            "role": "user",
                            "parts": [{"kind": "text", "text": message}],
                            "contextId": child_context_id,
                        },
                    },
                    "id": request_id,
                },
            )

            if response.status_code != 200:
                raise ValueError(
                    f"Failed to send message to target agent: {response.status_code} {response.text}"
                )

            # Read SSE stream to forward child messages to parent and establish relationship
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                raise ValueError(f"Expected SSE stream response, got: {content_type}")

            # Process SSE stream
            await _process_sse_stream(
                response,
                {
                    **args,
                    "source_context_id": args.get("source_context_id"),
                },
                callbacks,
                message_endpoint,
            )

        print("[Handoff] Handoff completed successfully")
        return handoff_result
    except Exception as error:
        error_message = str(error) if error else "Unknown error"
        error_stack = None
        if hasattr(error, "__traceback__"):
            import traceback
            error_stack = traceback.format_tb(error.__traceback__)

        print(f"[Handoff] Error during handoff to {agent_id}: {error_message}")
        if error_stack:
            print(f"[Handoff] Stack trace: {error_stack}")

        raise ValueError(f"Error during handoff: {error_message}")

