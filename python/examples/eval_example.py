#!/usr/bin/env python3
"""Example usage of Timestep eval framework."""

import json
from pathlib import Path

from timestep import run_suite, report, agent_builtin_echo, DEFAULT_TOOLS
from timestep.graders import FinalContains, ForbiddenTools, FinalRegex


def create_example_tasks():
    """Create example tasks for evaluation."""
    tasks = [
        {
            "id": "hello_01",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hello to Mike in one sentence."}
            ],
            "expected": {"final_contains": "Mike"},
            "limits": {"max_steps": 5, "time_limit_s": 30}
        },
        {
            "id": "calc_01",
            "messages": [
                {"role": "system", "content": "You must use the calc tool."},
                {"role": "user", "content": "Compute 19*7 using the calc tool, then answer with only the number."}
            ],
            "tools_allowed": ["calc"],
            "expected": {"final_regex": "^133$"},
            "limits": {"max_steps": 10, "time_limit_s": 30}
        }
    ]
    
    # Write to JSONL
    tasks_path = Path("tasks.jsonl")
    with tasks_path.open("w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    return tasks_path


def main():
    """Run example evaluation."""
    # Create tasks
    tasks_path = create_example_tasks()
    
    # Run eval suite
    run_suite(
        tasks_path=tasks_path,
        outdir=Path("runs/example"),
        agent=agent_builtin_echo,
        tools=DEFAULT_TOOLS,
        graders=[FinalContains(), ForbiddenTools(), FinalRegex()],
        trials=3,
        seed=0,
        agent_timeout_s=120,
    )
    
    # Generate report
    print("\n" + "="*60)
    print("Evaluation Report")
    print("="*60)
    report(Path("runs/example"))


if __name__ == "__main__":
    main()
