"""Session implementation for storing and retrieving agent conversations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from timestep.utils.constants import ROLE_SYSTEM
from timestep.utils.types import ChatMessage

logger = logging.getLogger(__name__)


@runtime_checkable
class Session(Protocol):
    """Protocol for session implementations."""

    session_id: str

    async def get_items(self, limit: int | None = None) -> list[ChatMessage]:
        """
        Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history.
        """
        ...

    async def add_items(self, items: list[ChatMessage]) -> None:
        """
        Add new items to the conversation history.

        Args:
            items: List of input items to add to the history.
        """
        ...

    async def pop_item(self) -> ChatMessage | None:
        """
        Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty.
        """
        ...

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        ...


class FileSession:
    """File-based session implementation using JSONL storage."""

    def __init__(
        self,
        agent_name: str,
        conversation_id: str,
        agent_instructions: str | None = None,
        storage_dir: str = "conversations",
    ):
        """
        Initialize session for a specific agent and conversation.

        Args:
            agent_name: Name of the agent
            conversation_id: Unique identifier for this conversation
            agent_instructions: Optional system instructions for the agent
            storage_dir: Directory to store conversation files
        """
        self._agent_name = agent_name
        self._conversation_id = conversation_id
        self._agent_instructions = agent_instructions
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(exist_ok=True)
        self._items: List[ChatMessage] = []
        self._load_conversation()

    @property
    def session_id(self) -> str:
        """Return the session ID (conversation_id)."""
        return self._conversation_id

    def _get_conversation_path(self) -> Path:
        """Get the file path for this conversation."""
        return self._storage_dir / f"{self._agent_name}_{self._conversation_id}.jsonl"

    def _load_conversation(self) -> None:
        """Load conversation from disk if it exists."""
        file_path = self._get_conversation_path()
        if not file_path.exists():
            # Initialize with system message if instructions provided
            if self._agent_instructions:
                self._items = [{"role": ROLE_SYSTEM, "content": self._agent_instructions}]
                self._save_conversation()
            return

        messages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)
                messages.append(message)

        self._items = messages

    def _save_conversation(self) -> None:
        """Save conversation to disk as JSONL (one message per line)."""
        file_path = self._get_conversation_path()
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for message in self._items:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")

    async def get_items(self, limit: int | None = None) -> list[ChatMessage]:
        """Retrieve the conversation history for this session."""
        if limit is None:
            return self._items.copy()
        # Return the latest N items in chronological order
        return self._items[-limit:] if limit > 0 else []

    async def add_items(self, items: list[ChatMessage]) -> None:
        """Add new items to the conversation history."""
        self._items.extend(items)
        self._save_conversation()

    async def pop_item(self) -> ChatMessage | None:
        """Remove and return the most recent item from the session."""
        if not self._items:
            return None
        item = self._items.pop()
        self._save_conversation()
        return item

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        self._items = []
        # Re-initialize with system message if instructions provided
        if self._agent_instructions:
            self._items = [{"role": "system", "content": self._agent_instructions}]
        self._save_conversation()




