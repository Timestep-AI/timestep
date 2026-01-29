# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "mlflow",
#   "datasets",
# ]
# ///

#!/usr/bin/env python3
"""Run GAIA benchmark evaluation against personal assistant agent using MLflow."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Fix package import: lib/python/ contains the timestep package
# but Python needs to import it as 'timestep', not 'python'
script_dir = Path(__file__).parent
lib_dir = script_dir.parent / "lib"
lib_python_dir = lib_dir / "python"

# Add lib/python to path
if str(lib_python_dir) not in sys.path:
    sys.path.insert(0, str(lib_python_dir))

# Create a 'timestep' module that points to the python directory
# This allows imports like 'from timestep.observability import tracing' to work
import types
timestep_module = types.ModuleType('timestep')
timestep_module.__path__ = [str(lib_python_dir)]
sys.modules['timestep'] = timestep_module

# Import MLflow
import mlflow
from mlflow.genai.scorers import Correctness


def load_tasks() -> List[Dict]:
    """Load evaluation tasks including GAIA tasks and test tasks.
    
    Returns:
        List of task dictionaries with 'question' and 'answer' fields
    """
    tasks = []
    
    # Add test task for "What's the weather in Oakland?"
    # This helps test the matching logic with a known question/answer pair
    tasks.append({
        "task_id": "test-weather-oakland",
        "question": "What's the weather in Oakland?",
        "answer": "The current weather in Oakland is 72°F and sunny, with a humidity level of 65%.",
    })
    
    # Load GAIA validation tasks from HuggingFace dataset
    from datasets import load_dataset
    
    # Load validation split from GAIA dataset, selecting only first 2 tasks
    ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    
    # Convert to list of dicts (only first 2 GAIA tasks)
    for item in ds.select(range(2)):
        task = {
            "task_id": item.get("task_id", ""),
            "question": item.get("Question", ""),
            "answer": item.get("Final answer", ""),
        }
        # Add file info if available
        if "file_name" in item and item["file_name"]:
            task["file_name"] = item["file_name"]
        if "file_path" in item and item["file_path"]:
            task["file_path"] = item["file_path"]
        tasks.append(task)
    
    return tasks


def generate_traces_for_evaluation(tasks: List[Dict], agent_url: str) -> None:
    """Generate traces by calling the agent for each task.
    
    This function calls the agent to generate traces that will be evaluated.
    The agent app handles all tracing automatically via OTEL and sends traces to MLflow.
    
    Args:
        tasks: List of GAIA tasks to process
        agent_url: URL of the agent to call
    """
    import httpx
    import time
    
    print(f"\nGenerating traces by calling agent for {len(tasks)} tasks...")
    
    for i, task in enumerate(tasks, 1):
        question = task.get("question", task.get("Question", ""))
        if not question:
            logger.warning(f"Task {i} has no question, skipping")
            continue
        
        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{agent_url}/v1/responses",
                    json={"input": question, "stream": False},
                )
                response.raise_for_status()
                print(f"  ✓ Generated trace for task {i}/{len(tasks)}")
        except Exception as e:
            logger.error(f"Error generating trace for task {i}: {e}")
        
        # Small delay to ensure traces are properly separated
        time.sleep(0.5)
    
    print(f"Generated {len(tasks)} traces. Waiting for traces to be ingested...")
    time.sleep(5)  # Give MLflow time to ingest the traces


async def main():
    # Configure MLflow tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Use default experiment (ID: 0) to match agent configuration
    # The agent app is configured with x-mlflow-experiment-id=0, so we must use the same experiment
    default_experiment = mlflow.get_experiment("0")
    experiment_id = default_experiment.experiment_id
    experiment_name = default_experiment.name
    
    print(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")
    print(f"\nIMPORTANT: Make sure your agent app is running with:")
    print(f"  OTEL_EXPORTER_OTLP_ENDPOINT={mlflow_tracking_uri}")
    print(f"  OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id={experiment_id}")
    print(f"\nNote: Using default experiment (ID: 0) to match agent configuration.")
    print(f"\nThe agent app handles all tracing automatically - this script just calls it.")
    print(f"Note: MLflow's OTLP exporter automatically appends /v1/traces to the endpoint URL.")
    
    # Create data directory if it doesn't exist (for saving failures)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load evaluation tasks (includes test tasks and GAIA tasks)
    print("\nLoading evaluation tasks...")
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks (1 test task + 2 GAIA tasks)")
    
    # Verify agent is reachable
    agent_url = os.getenv("PERSONAL_ASSISTANT_AGENT_URL", "http://localhost:9999")
    print(f"\nConnecting to personal assistant at {agent_url}")
    
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{agent_url}/", timeout=5.0)
            print(f"Agent is reachable (status: {response.status_code})")
    except Exception as e:
        print(f"Warning: Could not verify agent is running at {agent_url}: {e}")
        print("Make sure the personal assistant is running before starting evaluation.")
        print("Start it with: uv run scripts/personal_assistant_app.py")
        return
    
    # Step 1: Generate traces by calling the agent
    # The agent app will create traces automatically via OTEL and send them to MLflow
    generate_traces_for_evaluation(tasks, agent_url)
    
    # Step 2: Search and retrieve traces from MLflow
    # Get traces from the last hour (should include the traces we just generated)
    from datetime import datetime, timedelta
    one_hour_ago = datetime.now() - timedelta(hours=1)
    timestamp_ms = int(one_hour_ago.timestamp() * 1000)
    
    print(f"\nSearching for traces created after {one_hour_ago}...")
    # Search for traces - use timestamp filter only (locations parameter causes timeouts)
    traces = mlflow.search_traces(
        filter_string=f"timestamp > {timestamp_ms}"
    )
    
    if traces.empty:
        print("\nNo traces found. Make sure the agent app is running and generating traces.")
        print("Traces should be sent to MLflow with experiment ID:", experiment_id)
        print("You can also check the MLflow UI to see if traces are visible there.")
        return
    
    print(f"Found {len(traces)} traces")
    
    # Step 3: Annotate traces with expectations (ground truth)
    # Match traces to tasks by comparing request content
    print("\nAnnotating traces with expectations...")
    
    # Create a mapping of questions to expected answers
    question_to_answer = {
        task.get("question", task.get("Question", "")): task.get("answer", task.get("Final answer", ""))
        for task in tasks
    }
    
    annotated_count = 0
    
    for idx, trace_row in traces.iterrows():
        trace_id = trace_row.get('trace_id')
        if not trace_id:
            logger.warning(f"Trace row {idx} has no trace_id, skipping")
            continue
        
        # Try to extract the request/question from the trace
        # The trace DataFrame has a 'request' column that contains the input
        trace_request = None
        if 'request' in trace_row:
            trace_request = trace_row['request']
        elif 'trace' in trace_row and hasattr(trace_row['trace'], 'data'):
            # Try to get from trace data structure
            trace_data = trace_row['trace'].data if hasattr(trace_row['trace'], 'data') else None
            if trace_data and isinstance(trace_data, dict):
                trace_request = (
                    trace_data.get('request', {}).get('inputs', {}).get('question') or
                    trace_data.get('request', {}).get('input') or
                    trace_data.get('inputs', {}).get('question') or
                    trace_data.get('input')
                )
        
        # Match trace to task by comparing request/question content
        expected_answer = None
        if trace_request:
            # Try exact match first
            if trace_request in question_to_answer:
                expected_answer = question_to_answer[trace_request]
            else:
                # Try partial match (trace request might contain system prompt + question)
                for question, answer in question_to_answer.items():
                    if question in trace_request or trace_request in question:
                        expected_answer = answer
                        break
        
        # If we found a match, add the expectation
        if expected_answer:
            try:
                mlflow.log_expectation(
                    trace_id=trace_id,
                    name="expected_response",
                    value=expected_answer
                )
                annotated_count += 1
                logger.debug(f"Added expectation to trace {trace_id} for question: {trace_request[:50] if trace_request else 'N/A'}...")
            except Exception as e:
                logger.warning(f"Could not add expectation to trace {trace_id}: {e}")
        else:
            logger.warning(f"Could not match trace {trace_id} to any task. Request: {trace_request[:100] if trace_request else 'N/A'}...")
    
    print(f"Annotated {annotated_count} traces with expectations")
    
    # Step 4: Define scorers for MLflow evaluation
    # Correctness scorer compares against "expected_response" field in expectations
    scorers = [
        Correctness(),
    ]
    
    print("\nRunning MLflow evaluation on traces...")
    print("MLflow will automatically extract inputs, outputs, and expectations from traces.")
    
    # Step 5: Run MLflow evaluation on traces
    # No predict_fn needed - MLflow extracts everything from the trace objects
    results = mlflow.genai.evaluate(
        data=traces,  # Pass traces directly, not a dataset with predict_fn
        scorers=scorers,
    )
    
    print("\nEvaluation completed!")
    print(f"Results logged to MLflow experiment: {experiment_name}")
    print(f"View results at: {mlflow_tracking_uri}")
    
    print("\nEvaluation complete! Check MLflow UI for detailed results and traces.")
    print("Traces are created automatically by the agent app and sent to MLflow.")


if __name__ == "__main__":
    asyncio.run(main())
