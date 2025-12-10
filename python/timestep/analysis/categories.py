"""Category definitions for agent systems."""

from typing import Generic, TypeVar, List, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class Category(ABC, Generic[T]):
    """Base category class for category theory modeling."""
    
    @abstractmethod
    def compose(self, f: T, g: T) -> Optional[T]:
        """
        Compose two morphisms.
        
        Args:
            f: First morphism
            g: Second morphism
            
        Returns:
            Composed morphism, or None if composition is not defined
        """
        pass
    
    @abstractmethod
    def identity(self, obj: T) -> T:
        """
        Get the identity morphism for an object.
        
        Args:
            obj: Object to get identity for
            
        Returns:
            Identity morphism
        """
        pass
    
    @abstractmethod
    def objects(self) -> List[T]:
        """Get all objects in the category."""
        pass
    
    @abstractmethod
    def morphisms(self, source: T, target: T) -> List[T]:
        """
        Get all morphisms from source to target.
        
        Args:
            source: Source object
            target: Target object
            
        Returns:
            List of morphisms from source to target
        """
        pass


class AgentCategory(Category):
    """
    Category of agents.
    
    Objects: Individual agents
    Morphisms: Handoffs between agents
    """
    
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._handoffs: Dict[str, List[str]] = {}  # agent_id -> [handoff_agent_ids]
    
    def add_agent(self, agent_id: str, agent: Any):
        """Add an agent to the category."""
        self._agents[agent_id] = agent
        if agent_id not in self._handoffs:
            self._handoffs[agent_id] = []
    
    def add_handoff(self, source_id: str, target_id: str):
        """Add a handoff morphism from source to target agent."""
        if source_id not in self._handoffs:
            self._handoffs[source_id] = []
        if target_id not in self._handoffs[source_id]:
            self._handoffs[source_id].append(target_id)
    
    def compose(self, f: str, g: str) -> Optional[str]:
        """
        Compose two handoff chains.
        
        If f is a handoff from A to B, and g is a handoff from B to C,
        then compose(f, g) is a handoff from A to C.
        
        Args:
            f: Source agent ID for first handoff
            g: Target agent ID for second handoff
            
        Returns:
            Composed handoff target ID, or None if composition not possible
        """
        # Check if f can handoff to g
        if f in self._handoffs and g in self._handoffs.get(f, []):
            return g
        # Check if there's a path f -> intermediate -> g
        if f in self._handoffs:
            for intermediate in self._handoffs[f]:
                if intermediate in self._handoffs and g in self._handoffs.get(intermediate, []):
                    return g
        return None
    
    def identity(self, obj: str) -> str:
        """
        Identity morphism is a self-handoff (trivial).
        
        Args:
            obj: Agent ID
            
        Returns:
            Same agent ID (self-handoff)
        """
        return obj
    
    def objects(self) -> List[str]:
        """Get all agent IDs in the category."""
        return list(self._agents.keys())
    
    def morphisms(self, source: str, target: str) -> List[str]:
        """
        Get all handoff paths from source to target.
        
        Args:
            source: Source agent ID
            target: Target agent ID
            
        Returns:
            List of intermediate agent IDs forming paths from source to target
        """
        paths = []
        
        def find_paths(current: str, target: str, visited: set, path: List[str]):
            if current == target:
                paths.append(path.copy())
                return
            if current in visited:
                return
            visited.add(current)
            for next_agent in self._handoffs.get(current, []):
                path.append(next_agent)
                find_paths(next_agent, target, visited, path)
                path.pop()
            visited.remove(current)
        
        find_paths(source, target, set(), [])
        return paths


class ToolCategory(Category):
    """
    Category of tools.
    
    Objects: Individual tools
    Morphisms: Tool invocations (Tool -> Result)
    """
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._invocations: Dict[str, List[str]] = {}  # tool_id -> [result_types]
    
    def add_tool(self, tool_id: str, tool: Any):
        """Add a tool to the category."""
        self._tools[tool_id] = tool
    
    def add_invocation(self, tool_id: str, result_type: str):
        """Add a tool invocation morphism."""
        if tool_id not in self._invocations:
            self._invocations[tool_id] = []
        if result_type not in self._invocations[tool_id]:
            self._invocations[tool_id].append(result_type)
    
    def compose(self, f: str, g: str) -> Optional[str]:
        """
        Compose tool invocations (tool chaining).
        
        If f produces output that feeds into g, return g's result type.
        
        Args:
            f: First tool ID
            g: Second tool ID
            
        Returns:
            Result type of composition, or None if not composable
        """
        # Simplified: if f's output type matches g's input type, they compose
        # This is a placeholder - actual implementation would need type checking
        if f in self._invocations and g in self._tools:
            # Check if f's output can feed into g
            # For now, assume they compose if both exist
            return self._invocations.get(f, [None])[0] if self._invocations.get(f) else None
        return None
    
    def identity(self, obj: str) -> str:
        """
        Identity morphism for a tool (trivial invocation).
        
        Args:
            obj: Tool ID
            
        Returns:
            Same tool ID
        """
        return obj
    
    def objects(self) -> List[str]:
        """Get all tool IDs in the category."""
        return list(self._tools.keys())
    
    def morphisms(self, source: str, target: str) -> List[str]:
        """
        Get tool invocation chains from source to target.
        
        Args:
            source: Source tool ID
            target: Target result type
            
        Returns:
            List of tool chains
        """
        # Simplified implementation
        if source in self._invocations and target in self._invocations.get(source, []):
            return [target]
        return []

