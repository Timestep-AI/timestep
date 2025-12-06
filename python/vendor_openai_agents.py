#!/usr/bin/env python3
"""Vendor the openai-agents-python package into timestep/_vendored/agents."""

import shutil
import sys
from pathlib import Path

def vendor_openai_agents():
    """Copy vendored openai-agents into the package."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    source = repo_root / "3rdparty" / "openai-agents-python" / "src" / "agents"
    dest = script_dir / "timestep" / "_vendored" / "agents"
    
    if not source.exists():
        print(f"Error: Source directory not found: {source}", file=sys.stderr)
        print("Make sure submodules are initialized: git submodule update --init", file=sys.stderr)
        sys.exit(1)
    
    # Remove existing vendored code
    if dest.exists():
        shutil.rmtree(dest)
    
    # Create parent directory
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the agents package
    shutil.copytree(source, dest)
    
    # Create __init__.py if it doesn't exist in the vendored location
    vendored_init = dest / "__init__.py"
    if not vendored_init.exists():
        vendored_init.touch()
    
    print(f"✓ Vendored {source} -> {dest}")
    return 0

if __name__ == "__main__":
    sys.exit(vendor_openai_agents())

