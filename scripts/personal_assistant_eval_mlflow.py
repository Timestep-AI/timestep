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


def load_gaia_tasks() -> List[Dict]:
    """Load GAIA validation tasks from HuggingFace dataset."""
    from datasets import load_dataset
    
    # Load validation split from GAIA dataset
    ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    
    # Convert to list of dicts
    tasks = []
    for item in ds:
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


def agent_predict_fn(inputs: dict) -> str:
    """Prediction function that calls the personal assistant agent via HTTP.
    
    This function is used by MLflow's evaluation framework to get agent responses.
    The agent app handles all tracing automatically via OTEL.
    
    Args:
        inputs: Dict containing "question" key with the question to ask the agent
        
    Returns:
        The agent's response text
    """
    import httpx
    
    # Extract question from inputs dict (MLflow format)
    question = inputs.get("question", "")
    if not question:
        raise ValueError("Inputs dict must contain 'question' key")
    
    agent_url = os.getenv("PERSONAL_ASSISTANT_AGENT_URL", "http://localhost:9999")
    
    # Simple HTTP call - the agent app handles tracing automatically
    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            f"{agent_url}/v1/responses",
            json={"input": question, "stream": False},
        )
        response.raise_for_status()
        result_data = response.json()
        
        # Extract output from response
        output = ""
        output_items = result_data.get("output", [])
        for item in output_items:
            if item.get("type") == "message":
                content = item.get("content", [])
                for content_item in content:
                    if content_item.get("type") == "output_text":
                        output += content_item.get("text", "")
        
        return output


async def main():
    # Configure MLflow tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set MLflow experiment
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "GAIA Evaluation")
    mlflow.set_experiment(experiment_name)
    
    # Get or create experiment to get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    print(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")
    print(f"\nIMPORTANT: Make sure your agent app is running with:")
    print(f"  OTEL_EXPORTER_OTLP_ENDPOINT={mlflow_tracking_uri}/v1/traces")
    print(f"  OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id={experiment_id}")
    print(f"\nThe agent app handles all tracing automatically - this script just calls it.")
    
    # Create data directory if it doesn't exist (for saving failures)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load GAIA validation tasks
    print("\nLoading GAIA validation dataset...")
    tasks = load_gaia_tasks()
    print(f"Loaded {len(tasks)} GAIA tasks")
    
    # For initial testing, only use the first task
    tasks = tasks[:1]
    print(f"Using first task only for initial testing: {len(tasks)} task(s)")
    
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
    
    # Convert GAIA tasks to MLflow evaluation format
    # MLflow format: [{"inputs": {"question": "..."}, "expectations": {"expected_response": "..."}}]
    eval_dataset = []
    for task in tasks:
        eval_item = {
            "inputs": {"question": task.get("question", task.get("Question", ""))},
            "expectations": {"expected_response": task.get("answer", task.get("Final answer", ""))}
        }
        eval_dataset.append(eval_item)
    
    print(f"\nPrepared {len(eval_dataset)} evaluation cases")
    
    # Define scorers for MLflow evaluation
    # Correctness scorer compares against "expected_response" field in expectations
    scorers = [
        Correctness(),
    ]
    
    print("\nRunning MLflow evaluation...")
    print("The agent app will create traces automatically via OTEL.")
    
    # Run MLflow evaluation
    # MLflow will call agent_predict_fn for each case, which makes HTTP calls to the agent
    # The agent app handles tracing automatically and sends traces to MLflow
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=agent_predict_fn,
        scorers=scorers,
    )
    
    print("\nEvaluation completed!")
    print(f"Results logged to MLflow experiment: {experiment_name}")
    print(f"View results at: {mlflow_tracking_uri}")
    
    print("\nEvaluation complete! Check MLflow UI for detailed results and traces.")
    print("Traces are created automatically by the agent app and sent to MLflow.")


if __name__ == "__main__":
    asyncio.run(main())
