"""A2A AgentExecutor implementation for Timestep agents."""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, completed_task
from a2a.types import TaskState, Artifact, TextPart, Part

from timestep.services.agent import Agent
from timestep.stores.session import Session
from timestep.utils.types import ChatMessage


class TimestepAgentExecutor(AgentExecutor):
    """A2A AgentExecutor implementation for Timestep agents."""

    def __init__(
        self,
        agent_config: Dict[str, Any],
        session: Session,
    ):
        """Initialize Timestep agent executor.

        Args:
            agent_config: Agent configuration dictionary (must include "id" field)
            session: Session for conversation persistence
        """
        self.agent_config = agent_config
        self.agent_id = agent_config.get("id")
        self.session = session
        self.agent = Agent(
            model=agent_config["model"],
            api_key=None
        )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent with A2A context and event queue.

        Args:
            context: A2A request context
            event_queue: A2A event queue for streaming events
        """
        message = context.message
        if len(message.parts) != 1:
            raise ValueError(f"Expected exactly 1 part, got {len(message.parts)}")
        if message.parts[0].root.kind != "text":
            raise ValueError(f"Expected text part, got {message.parts[0].root.kind}")
        
        # Create task from message (Task-generating agent pattern)
        task = new_task(message)
        task_id = task.id
        context_id = context.context_id or task.context_id
        
        # Enqueue task creation
        await event_queue.enqueue_event({"kind": "task", "task": task})
        
        messages = [{"role": message.role, "content": message.parts[0].root.text}]
        
        # Run agent and collect response (suppress message events, collect content)
        response_content = ""
        async for event in self._run_agent_events(messages):
            if event.get("type") == "content_delta":
                response_content += event.get("content", "")
            elif event.get("type") == "message":
                response_content = event.get("content", "")
        
        # Create artifact with response
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="response",
            description="Agent response",
            parts=[Part(root=TextPart(text=response_content))]
        )
        
        # Create completed task
        completed = completed_task(
            task_id=task_id,
            context_id=context_id,
            artifacts=[artifact],
            history=None
        )
        completed.status.state = TaskState.completed
        
        # Enqueue completed task
        await event_queue.enqueue_event({"kind": "task", "task": completed})
    
    async def _run_agent_events(self, messages: list[ChatMessage]):
        """Run agent and yield events without enqueueing to event_queue."""
        from timestep.services.environment import Environment
        from timestep.services.agent import ExecutionLoop
        
        environment = Environment(self.agent_config, self.session, api_key=None)
        self.agent.tools = environment.get_tools()
        
        existing_messages = await self.session.get_items()
        if messages or not existing_messages:
            await environment.reset(initial_messages=messages)
        
        loop = ExecutionLoop(self.agent, environment)
        async for event in loop.run():
            yield event

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel agent execution.

        Args:
            context: A2A request context
            event_queue: A2A event queue
        """
        raise Exception("cancel not supported")
