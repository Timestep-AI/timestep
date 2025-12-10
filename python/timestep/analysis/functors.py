"""Functor implementations for agent systems."""

from typing import Generic, TypeVar, Callable, List, Dict, Any, Optional
from abc import ABC, abstractmethod
from .categories import Category, AgentCategory, ToolCategory

T = TypeVar('T')
U = TypeVar('U')


class Functor(ABC, Generic[T, U]):
    """
    Base functor class.
    
    A functor F: C -> D maps objects and morphisms from category C to category D,
    preserving composition and identity.
    """
    
    @abstractmethod
    def map_object(self, obj: T) -> U:
        """Map an object from source category to target category."""
        pass
    
    @abstractmethod
    def map_morphism(self, morphism: T, source: T, target: T) -> U:
        """
        Map a morphism from source category to target category.
        
        Args:
            morphism: The morphism to map
            source: Source object in source category
            target: Target object in source category
            
        Returns:
            Mapped morphism in target category
        """
        pass


class AgentToolFunctor(Functor):
    """
    Functor from Agent category to Tool category.
    
    Maps agents to their available tools.
    """
    
    def __init__(self, agent_category: AgentCategory, tool_category: ToolCategory):
        self.agent_category = agent_category
        self.tool_category = tool_category
        self._agent_to_tools: Dict[str, List[str]] = {}
    
    def map_object(self, agent_id: str) -> List[str]:
        """
        Map an agent to its list of tool IDs.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of tool IDs available to the agent
        """
        return self._agent_to_tools.get(agent_id, [])
    
    def add_agent_tools(self, agent_id: str, tool_ids: List[str]):
        """Associate tools with an agent."""
        self._agent_to_tools[agent_id] = tool_ids
        for tool_id in tool_ids:
            if tool_id not in self.tool_category.objects():
                # Tool should already be in category, but ensure it exists
                pass
    
    def map_morphism(self, handoff: str, source_agent: str, target_agent: str) -> Optional[List[str]]:
        """
        Map a handoff morphism to tool composition.
        
        When agent A handoffs to agent B, the tools of A and B form a composition.
        
        Args:
            handoff: Handoff target agent ID
            source_agent: Source agent ID
            target_agent: Target agent ID
            
        Returns:
            List of tool IDs representing the composition, or None
        """
        source_tools = self.map_object(source_agent)
        target_tools = self.map_object(target_agent)
        # Composition of tools: tools from both agents
        return source_tools + target_tools


class HandoffFunctor(Functor):
    """
    Functor from Agent category to Agent category.
    
    Maps agents to their handoff targets (delegation structure).
    """
    
    def __init__(self, agent_category: AgentCategory):
        self.agent_category = agent_category
    
    def map_object(self, agent_id: str) -> List[str]:
        """
        Map an agent to its handoff targets.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of agent IDs that this agent can handoff to
        """
        return self.agent_category._handoffs.get(agent_id, [])
    
    def map_morphism(self, handoff: str, source_agent: str, target_agent: str) -> Optional[str]:
        """
        Map a handoff morphism (identity on handoffs).
        
        Args:
            handoff: Handoff target agent ID
            source_agent: Source agent ID
            target_agent: Target agent ID
            
        Returns:
            Target agent ID (handoff is preserved)
        """
        if target_agent in self.map_object(source_agent):
            return target_agent
        return None


class StateFunctor(Functor):
    """
    Functor from Agent × Session category to RunState.
    
    Maps agent-session pairs to execution states.
    """
    
    def __init__(self):
        self._agent_session_states: Dict[tuple, str] = {}  # (agent_id, session_id) -> state_id
    
    def map_object(self, agent_session: tuple) -> Optional[str]:
        """
        Map an agent-session pair to a state ID.
        
        Args:
            agent_session: Tuple of (agent_id, session_id)
            
        Returns:
            State ID, or None if no state exists
        """
        return self._agent_session_states.get(agent_session)
    
    def add_state(self, agent_id: str, session_id: str, state_id: str):
        """Associate a state with an agent-session pair."""
        self._agent_session_states[(agent_id, session_id)] = state_id
    
    def map_morphism(self, state_transition: str, source: tuple, target: tuple) -> Optional[str]:
        """
        Map a state transition morphism.
        
        Args:
            state_transition: State transition ID
            source: Source (agent_id, session_id) tuple
            target: Target (agent_id, session_id) tuple
            
        Returns:
            State transition ID
        """
        # State transitions preserve the state mapping
        return state_transition

