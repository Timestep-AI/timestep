"""Suite runner for evaluation harness."""

import dataclasses
import random
from pathlib import Path
from typing import Any, Dict, List

from ..core.episode import run_episode, EpisodeInfo
from ..core.tools import ToolFn, index_tool_calls
from ..core.types import AgentFn, Message
from .graders import Grader, aggregate_grades
from ..utils.jsonl import read_jsonl, write_jsonl
from ..utils.io import write_json, now
from ..utils.messages import ensure_task_id

JSON = Dict[str, Any]


def run_suite(
    tasks_path: Path,
    outdir: Path,
    agent: AgentFn,
    tools: Dict[str, ToolFn],
    graders: List[Grader],
    trials: int,
    seed: int,
    agent_timeout_s: int,
) -> None:
    """Run evaluation suite on tasks from JSONL file."""
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    run_meta = {
        "version": "eval_mvp_v1",
        "tasks_path": str(tasks_path),
        "trials": trials,
        "seed": seed,
        "started_at": now(),
        "graders": [g.__class__.__name__ for g in graders],
        "agent_timeout_s": agent_timeout_s,
        "tools_available": sorted(list(tools.keys())),
    }
    write_json(outdir / "run_meta.json", run_meta)

    results_rows: List[JSON] = []

    for task in read_jsonl(tasks_path):
        task_id = ensure_task_id(task)
        task_messages = task.get("messages")
        if not isinstance(task_messages, list):
            raise SystemExit(f"Task {task_id} missing 'messages' list.")

        tools_allowed = task.get("tools_allowed")  # optional allowlist
        limits = task.get("limits", {}) or {}

        for trial in range(1, trials + 1):
            trial_seed = rng.randrange(0, 2**31 - 1)

            trial_dir = outdir / "trials" / task_id / f"trial_{trial:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Attach trial metadata (kept out of messages)
            task_meta = dict(task)
            task_meta["_trial"] = trial

            # Run episode (core agent-environment loop)
            messages, info = run_episode(
                initial_messages=task_messages,
                agent=agent,
                tools=tools,
                tools_allowed=tools_allowed,
                limits=limits,
                task_meta=task_meta,
                seed=trial_seed,
            )

            # Build tool index
            tool_idx = index_tool_calls(messages)

            # Grade
            grade_rows = [g.grade(messages, tool_idx, task, info) for g in graders]
            agg = aggregate_grades(grade_rows)

            # Persist artifacts
            write_json(trial_dir / "transcript.json", messages)
            write_json(trial_dir / "tool_index.json", [dataclasses.asdict(r) for r in tool_idx])
            write_json(trial_dir / "grades.json", {"grades": grade_rows, "aggregate": agg})
            write_json(trial_dir / "info.json", dataclasses.asdict(info))

            # Row for results.jsonl
            results_rows.append({
                "task_id": task_id,
                "trial": trial,
                "seed": trial_seed,
                "terminated_reason": info.terminated_reason,
                "steps": info.steps,
                "tool_calls": info.tool_calls,
                "duration_s": info.duration_s,
                "input_tokens": info.input_tokens,
                "output_tokens": info.output_tokens,
                "total_tokens": info.total_tokens,
                "cost_usd": info.cost_usd,
                "passed": agg["passed"],
                "score": agg["score"],
            })

    write_jsonl(outdir / "results.jsonl", results_rows)
    run_meta["ended_at"] = now()
    write_json(outdir / "run_meta.json", run_meta)


def report(outdir: Path) -> None:
    """Print summary report of evaluation results."""
    results_path = outdir / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"No results.jsonl in {outdir}")

    rows = list(read_jsonl(results_path))
    if not rows:
        print("No results.")
        return

    overall_pass = sum(1 for r in rows if r.get("passed")) / len(rows)
    overall_score = sum(float(r.get("score", 0.0)) for r in rows) / len(rows)
    avg_tokens = sum(float(r.get("total_tokens", 0)) for r in rows) / len(rows) if rows else 0

    by_task: Dict[str, List[JSON]] = {}
    for r in rows:
        by_task.setdefault(str(r["task_id"]), []).append(r)

    task_summaries = []
    for tid, rs in by_task.items():
        pr = sum(1 for x in rs if x.get("passed")) / len(rs)
        ms = sum(float(x.get("score", 0.0)) for x in rs) / len(rs)
        md = sum(float(x.get("duration_s", 0.0)) for x in rs) / len(rs)
        mt = sum(float(x.get("total_tokens", 0)) for x in rs) / len(rs)
        task_summaries.append((tid, pr, ms, md, mt, len(rs)))

    task_summaries.sort(key=lambda x: (x[1], x[2]))  # worst first

    print(f"Run: {outdir}")
    print(f"Trials: {len(rows)}")
    print(f"Overall pass rate: {overall_pass:.3f}")
    print(f"Overall mean score: {overall_score:.3f}")
    print(f"Average tokens per trial: {avg_tokens:.0f}")
    print()
    print("Worst tasks (task_id | pass_rate | mean_score | mean_duration_s | mean_tokens | trials):")
    for tid, pr, ms, md, mt, n in task_summaries[:20]:
        print(f"  {tid} | {pr:.3f} | {ms:.3f} | {md:.3f} | {mt:.0f} | {n}")
