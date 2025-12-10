"""Script to generate documentation from agent definitions."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import timestep
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from timestep.docs.generator import DocumentationGenerator
from timestep.stores.agent_store.store import load_agent
from timestep.stores.shared.db_connection import DatabaseConnection
from timestep.config.dbos_config import get_dbos_connection_string, configure_dbos


async def get_all_agent_ids() -> list[str]:
    """
    Load all agent IDs from the database.
    
    Returns:
        List of agent IDs
    """
    connection_string = get_dbos_connection_string()
    if not connection_string:
        await configure_dbos()
        connection_string = get_dbos_connection_string()
    
    if not connection_string:
        print("Warning: No database connection available. Cannot load agent IDs.")
        return []
    
    db = DatabaseConnection(connection_string=connection_string)
    await db.connect()
    try:
        rows = await db.fetch("SELECT id FROM agents")
        return [row['id'] for row in rows]
    finally:
        await db.disconnect()


async def main():
    """Main entry point for documentation generation."""
    # Load all agent IDs from database or config
    agent_ids = await get_all_agent_ids()
    
    if not agent_ids:
        print("No agents found in database. Skipping documentation generation.")
        return
    
    # Generate documentation
    docs_dir = Path(__file__).parent.parent / "docs" / "generated"
    generator = DocumentationGenerator(docs_dir)
    await generator.generate_all(agent_ids)
    
    print(f"Documentation generated successfully to {docs_dir}!")
    print(f"Generated documentation for {len(agent_ids)} agents.")


if __name__ == '__main__':
    asyncio.run(main())

