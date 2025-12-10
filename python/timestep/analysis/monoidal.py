"""Monoidal category structures for agent composition."""

from typing import Generic, TypeVar, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from .categories import Category, AgentCategory

T = TypeVar('T')


class MonoidalCategory(ABC, Generic[T]):
    """
    Base class for monoidal categories.
    
    A monoidal category has:
    - Tensor product (parallel composition)
    - Unit object (identity)
    - Associativity and unit laws
    """
    
    @abstractmethod
    def tensor(self, obj1: T, obj2: T) -> T:
        """
        Tensor product (parallel composition).
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Tensor product of the two objects
        """
        pass
    
    @abstractmethod
    def unit(self) -> T:
        """
        Get the unit object (identity for tensor).
        
        Returns:
            Unit object
        """
        pass
    
    @abstractmethod
    def compose(self, f: T, g: T) -> Optional[T]:
        """
        Sequential composition.
        
        Args:
            f: First morphism
            g: Second morphism
            
        Returns:
            Composed morphism
        """
        pass


class AgentComposition(MonoidalCategory):
    """
    Monoidal structure for agent composition.
    
    - Tensor product: Parallel agent execution
    - Unit: Empty agent (identity)
    - Composition: Sequential agent workflows
    """
    
    def __init__(self, agent_category: AgentCategory):
        self.agent_category = agent_category
        self._parallel_groups: Dict[str, List[str]] = {}  # group_id -> [agent_ids]
        self._unit_agent_id = "__unit__"
    
    def tensor(self, agent1_id: str, agent2_id: str) -> str:
        """
        Tensor product: parallel execution of two agents.
        
        Args:
            agent1_id: First agent ID
            agent2_id: Second agent ID
            
        Returns:
            Group ID representing parallel execution
        """
        group_id = f"parallel_{agent1_id}_{agent2_id}"
        self._parallel_groups[group_id] = [agent1_id, agent2_id]
        return group_id
    
    def unit(self) -> str:
        """
        Get the unit agent (empty/identity agent).
        
        Returns:
            Unit agent ID
        """
        return self._unit_agent_id
    
    def compose(self, f: str, g: str) -> Optional[str]:
        """
        Sequential composition of agent workflows.
        
        Args:
            f: First agent/workflow ID
            g: Second agent/workflow ID
            
        Returns:
            Composed workflow ID, or None if not composable
        """
        # Use agent category composition for sequential handoffs
        return self.agent_category.compose(f, g)
    
    def get_parallel_agents(self, group_id: str) -> List[str]:
        """
        Get agents in a parallel group.
        
        Args:
            group_id: Parallel group ID
            
        Returns:
            List of agent IDs in the group
        """
        return self._parallel_groups.get(group_id, [])
    
    def add_parallel_group(self, group_id: str, agent_ids: List[str]):
        """Add a parallel execution group."""
        self._parallel_groups[group_id] = agent_ids

