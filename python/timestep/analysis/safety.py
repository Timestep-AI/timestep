"""Runtime safety checks for agent systems."""

from typing import List, Set, Optional, Protocol, TypeVar
from ..stores.agent_store.store import load_agent
from ..stores.run_state_store.store import RunStateStore

T = TypeVar('T')


class CircularDependencyChecker:
    """Check for circular handoff dependencies at runtime."""
    
    async def check_circular_handoffs(self, agent_id: str) -> Optional[List[str]]:
        """
        Check if an agent has circular handoff dependencies.
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            None if no cycles found, otherwise list of agent IDs forming the cycle
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []
        
        async def has_cycle(aid: str) -> Optional[List[str]]:
            visited.add(aid)
            rec_stack.add(aid)
            path.append(aid)
            
            try:
                agent = await load_agent(aid)
            except Exception:
                # Can't load agent, skip cycle check
                rec_stack.remove(aid)
                path.pop()
                return None
            
            # Get handoff agent IDs from database
            from ..stores.shared.db_connection import DatabaseConnection
            
            connection_string = os.environ.get("PG_CONNECTION_URI")
            if not connection_string:
                rec_stack.remove(aid)
                path.pop()
                return None  # Can't check without DB
            
            db = DatabaseConnection(connection_string=connection_string)
            await db.connect()
            try:
                handoff_rows = await db.fetch("""
                    SELECT handoff_agent_id FROM agent_handoffs
                    WHERE agent_id = $1 AND handoff_agent_id IS NOT NULL
                """, aid)
                
                for row in handoff_rows:
                    handoff_agent_id = row['handoff_agent_id']
                    if handoff_agent_id not in visited:
                        cycle = await has_cycle(handoff_agent_id)
                        if cycle:
                            return cycle
                    elif handoff_agent_id in rec_stack:
                        # Found cycle - return the cycle path
                        cycle_start = path.index(handoff_agent_id)
                        return path[cycle_start:] + [handoff_agent_id]
            finally:
                await db.disconnect()
            
            rec_stack.remove(aid)
            path.pop()
            return None
        
        return await has_cycle(agent_id)


class ToolCompatibilityChecker:
    """Check tool compatibility with agent handoffs."""
    
    async def check_compatibility(self, agent_id: str) -> List[str]:
        """
        Check if tools used by an agent are compatible with its handoffs.
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            List of compatibility warnings
        """
        try:
            agent = await load_agent(agent_id)
        except Exception:
            return [f"Could not load agent {agent_id}"]
        
        warnings = []
        
        # Check if tool outputs match handoff input requirements
        # Note: This is a simplified check - actual type checking would be more complex
        for tool in agent.tools:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            
            for handoff in agent.handoffs:
                # Handoff can be Agent or Handoff object
                if hasattr(handoff, 'agent'):
                    handoff_agent = handoff.agent
                elif hasattr(handoff, 'name'):
                    handoff_agent = handoff
                else:
                    continue
                
                handoff_name = handoff_agent.name if hasattr(handoff_agent, 'name') else str(handoff_agent)
                # This is a placeholder - actual implementation would need type system
                warnings.append(
                    f"Tool {tool_name} may not be compatible with handoff to {handoff_name}"
                )
        
        return warnings


class StateInvariant(Protocol[T]):
    """Protocol for state invariants that must be preserved."""
    
    def check(self, state: T) -> bool:
        """Check if state satisfies invariant."""
        ...
    
    def description(self) -> str:
        """Human-readable description of the invariant."""
        ...


class StateVerifier:
    """Verify state transitions preserve invariants."""
    
    def __init__(self, invariants: List[StateInvariant]):
        """
        Initialize state verifier with invariants.
        
        Args:
            invariants: List of state invariants to check
        """
        self.invariants = invariants
    
    async def verify_transition(
        self, 
        state_store: RunStateStore,
        before_state_id: str,
        after_state_id: str
    ) -> List[str]:
        """
        Verify that state transition preserves all invariants.
        
        Args:
            state_store: RunStateStore instance
            before_state_id: State ID before transition
            after_state_id: State ID after transition
            
        Returns:
            List of violations (empty if all invariants satisfied)
        """
        try:
            before_state = await state_store.load_by_id(before_state_id)
            after_state = await state_store.load_by_id(after_state_id)
        except Exception as e:
            return [f"Could not load states: {e}"]
        
        violations = []
        for invariant in self.invariants:
            if not invariant.check(before_state):
                violations.append(f"Invariant violated before: {invariant.description()}")
            if not invariant.check(after_state):
                violations.append(f"Invariant violated after: {invariant.description()}")
        
        return violations

