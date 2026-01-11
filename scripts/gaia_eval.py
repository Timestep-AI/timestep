#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets",
#   "openai>=1.0.0",
# ]
# ///

"""
GAIA Eval-Driven Development Script

This script demonstrates eval-driven development using GAIA level 1 tasks.

Workflow:
1. Fetch GAIA level 1 tasks (20 tasks, deterministically selected)
2. Create agent with configurable parameters
3. Run evaluation suite
4. View results with pass@k and pass^k metrics
5. Iterate: modify agent parameters, re-run, compare
"""

import sys
import os
from pathlib import Path

# Add local timestep package to path (for development)
# Assumes script is in scripts/ and timestep is in python/
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root / "python"))

from timestep import create_agent, run_suite, report, DEFAULT_TOOLS, FinalContains
from timestep.utils.jsonl import write_jsonl
from datasets import load_dataset


def convert_gaia_to_timestep(gaia_task):
    """Convert GAIA task format to Timestep task format."""
    task_id = gaia_task['task_id']
    question = gaia_task['Question']
    # GAIA uses "Final answer" (with space) according to the dataset format
    final_answer = gaia_task['Final answer']
    
    # Build messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions accurately."},
        {"role": "user", "content": question}
    ]
    
    # Build expected values (using FinalContains grader)
    expected = {
        "final_contains": final_answer
    }
    
    # Set limits
    limits = {
        "max_steps": 20,
        "time_limit_s": 120
    }
    
    return {
        "id": task_id,
        "messages": messages,
        "expected": expected,
        "limits": limits
    }


def main():
    """Main evaluation workflow."""
    print("GAIA Eval-Driven Development")
    print("=" * 60)
    
    # Set your OpenAI API key if not already set
    # os.environ['OPENAI_API_KEY'] = 'your-key-here'
    
    # Fetch GAIA level 1 tasks
    # According to https://huggingface.co/datasets/gaia-benchmark/GAIA
    # Use the "2023_level1" config with "validation" split (has answers for development)
    # Test split has private answers and is for final evaluation only
    print("\nFetching GAIA level 1 tasks...")
    dataset = load_dataset('gaia-benchmark/GAIA', '2023_level1', split='validation')
    
    # Convert to list and sort deterministically by task_id, take first 20
    level1_tasks = sorted(list(dataset), key=lambda x: x['task_id'])[:20]
    
    if not level1_tasks:
        raise SystemExit("No level 1 tasks found in dataset.")
    
    print(f"Loaded {len(level1_tasks)} GAIA level 1 tasks")
    print(f"First task ID: {level1_tasks[0]['task_id']}")
    print(f"Last task ID: {level1_tasks[-1]['task_id']}")
    
    # Convert to Timestep format
    print("\nConverting GAIA tasks to Timestep format...")
    tasks = [convert_gaia_to_timestep(t) for t in level1_tasks]
    
    # Save to JSONL file
    tasks_path = Path('tasks.jsonl')
    write_jsonl(tasks_path, tasks)
    print(f"Converted {len(tasks)} tasks and saved to {tasks_path}")
    
    # Create agent
    print("\nCreating agent...")
    agent = create_agent(
        model="gpt-4o-mini",
        temperature=0.0,
        # base_url="https://api.openai.com/v1",  # Optional: use different API
        # api_key="your-key",  # Optional: override env var
    )
    print("Agent created (streaming enabled)")
    
    # Run baseline evaluation
    print("\nRunning baseline evaluation...")
    run_suite(
        tasks_path=tasks_path,
        outdir=Path('runs/baseline'),
        agent=agent,
        tools=DEFAULT_TOOLS,
        graders=[FinalContains()],
        trials=3,  # Run 3 trials per task
        seed=0,    # Deterministic seed
        agent_timeout_s=120,
    )
    print("\nBaseline evaluation complete")
    
    # View results
    print("\n" + "=" * 60)
    print("Baseline Results:")
    print("=" + "=" * 60)
    report(Path('runs/baseline'))
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("=" * 60)
    print("- pass@k: Probability of at least one success in k trials (per task and overall)")
    print("- pass^k: Probability that all k trials succeed (per task and overall)")
    print("- Modify agent parameters (model, temperature, etc.) and re-run to iterate")
    print("- Gradually increase the number of GAIA tasks as agent improves")


if __name__ == "__main__":
    main()
