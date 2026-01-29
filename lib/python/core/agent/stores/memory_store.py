"""Memory store for context-level conversation history."""

from abc import ABC, abstractmethod
from typing import List, Optional
from a2a.types import Message


class MemoryStore(ABC):
    """Abstract interface for storing context-level conversation history.
    
    Memory is keyed by context_id, meaning all tasks within the same context
    share the same conversation history.
    """
    
    @abstractmethod
    async def get_history(self, context_id: str) -> List[Message]:
        """Get conversation history for a context_id (context-level, all tasks share).
        
        Args:
            context_id: The context ID to get history for
            
        Returns:
            List of messages in the conversation history
        """
        pass
    
    @abstractmethod
    async def add_message(self, context_id: str, message: Message) -> None:
        """Add a message to conversation history for a context_id.
        
        Args:
            context_id: The context ID to add the message to
            message: The message to add
        """
        pass
    
    @abstractmethod
    async def clear_history(self, context_id: str) -> None:
        """Clear conversation history for a context_id.
        
        Args:
            context_id: The context ID to clear history for
        """
        pass


class InMemoryMemoryStore(MemoryStore):
    """In-memory implementation of MemoryStore.
    
    Stores conversation history in memory, keyed by context_id.
    This is the default implementation, but custom storage backends
    (database, file system, etc.) can be implemented by extending MemoryStore.
    """
    
    def __init__(self):
        """Initialize the in-memory memory store."""
        # Dictionary mapping context_id to list of messages
        self._history: dict[str, List[Message]] = {}
    
    async def get_history(self, context_id: str) -> List[Message]:
        """Get conversation history for a context_id.
        
        Args:
            context_id: The context ID to get history for
            
        Returns:
            List of messages in the conversation history (empty list if no history)
        """
        return self._history.get(context_id, []).copy()
    
    async def add_message(self, context_id: str, message: Message) -> None:
        """Add a message to conversation history for a context_id.
        
        Args:
            context_id: The context ID to add the message to
            message: The message to add
        """
        if context_id not in self._history:
            self._history[context_id] = []
        self._history[context_id].append(message)
    
    async def clear_history(self, context_id: str) -> None:
        """Clear conversation history for a context_id.
        
        Args:
            context_id: The context ID to clear history for
        """
        if context_id in self._history:
            del self._history[context_id]
