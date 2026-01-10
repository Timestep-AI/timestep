"""CLI interface for eval framework."""

import argparse
from pathlib import Path

from .agent import agent_builtin_echo, agent_cmd_factory
from .graders import parse_grader_spec
from .suite import run_suite, report
from .tools import DEFAULT_TOOLS


def main() -> None:
    """Main CLI entry point."""
    p = argparse.ArgumentParser(description="OpenAI-message-protocol eval MVP")
    sub = p.add_subparsers(dest="cmd", required=True)

    prun = sub.add_parser("run", help="Run eval suite")
    prun.add_argument("--tasks", required=True, type=Path, help="JSONL tasks file")
    prun.add_argument("--outdir", required=True, type=Path, help="Output directory")
    prun.add_argument("--trials", type=int, default=3)
    prun.add_argument("--seed", type=int, default=0)
    prun.add_argument("--agent", required=True,
                      help='Agent spec: "builtin:echo" or "cmd:python my_agent.py"')
    prun.add_argument("--agent-timeout-s", type=int, default=120)
    prun.add_argument("--graders", nargs="*", default=[
        "ForbiddenTools",
        "MaxToolCalls:50",
        "FinalRegex",   # uses task.expected.final_regex if present
        "FinalContains" # uses task.expected.final_contains if present
    ], help="List of graders (builtin). Example: FinalRegex:^133$ MaxToolCalls:5")

    preport = sub.add_parser("report", help="Summarize results")
    preport.add_argument("--outdir", required=True, type=Path)

    args = p.parse_args()

    if args.cmd == "report":
        report(args.outdir)
        return

    # Build agent
    if args.agent.startswith("builtin:"):
        name = args.agent.split(":", 1)[1]
        if name == "echo":
            agent = agent_builtin_echo
        else:
            raise SystemExit(f"Unknown builtin agent '{name}'. Available: echo")
    elif args.agent.startswith("cmd:"):
        cmd = args.agent.split(":", 1)[1].strip()
        agent = agent_cmd_factory(cmd, timeout_s=args.agent_timeout_s)
    else:
        raise SystemExit('Agent must be "builtin:echo" or "cmd:...".')

    # Build graders
    graders = [parse_grader_spec(s) for s in args.graders]

    # Tools (demo defaults). In real use you can customize these or load from a module.
    tools = dict(DEFAULT_TOOLS)

    run_suite(
        tasks_path=args.tasks,
        outdir=args.outdir,
        agent=agent,
        tools=tools,
        graders=graders,
        trials=args.trials,
        seed=args.seed,
        agent_timeout_s=args.agent_timeout_s,
    )


if __name__ == "__main__":
    main()
