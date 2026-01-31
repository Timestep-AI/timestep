# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "mlflow",
#   "datasets",
#   "openai",
# ]
# ///

#!/usr/bin/env python3
"""Run GAIA benchmark evaluation against personal assistant agent using MLflow.

All agents should export to the same OTLP/MLflow endpoint (distributed tracing).
With trace context propagation, one request to /v1/responses = one trace (3 tasks = 3 traces).
The OTel log file (from the personal assistant) is used to select which traces to evaluate
and to supply request/response for the Correctness scorer.
"""

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


def _otel_trace_id_to_mlflow(trace_id: str) -> str:
    """Convert OTel trace_id (0x...) to MLflow format (tr-...)."""
    if not trace_id:
        return ""
    hex_part = trace_id.replace("0x", "").strip()
    return f"tr-{hex_part}" if hex_part else ""


def _parse_otel_log_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse OTel log JSONL file and build trace_id (MLflow format) -> {input, output, inputs_list}.

    - input: content from the first gen_ai.user.message from service personal-assistant (original user question).
    - inputs_list: all gen_ai.user.message contents from personal-assistant for this trace (in file order).
      Used to match trace to task when the first message is a handoff sub-query (e.g. research assistant's query).
    - output: message.content from the last gen_ai.choice with finish_reason=stop from
      service.name=personal-assistant (PA's final response). If no PA stop is found,
      uses the last gen_ai.choice with finish_reason=stop from any service.
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not file_path or not Path(file_path).exists():
        return result
    try:
        by_trace: Dict[str, List[Dict]] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                trace_id_raw = record.get("trace_id") or ""
                trace_id_mlflow = _otel_trace_id_to_mlflow(trace_id_raw)
                if not trace_id_mlflow:
                    continue
                event_name = record.get("event_name") or ""
                body = record.get("body")
                resource = record.get("resource") or {}
                attrs = resource.get("attributes") or {}
                service = (attrs.get("service.name") or "").strip('"')
                if trace_id_mlflow not in by_trace:
                    by_trace[trace_id_mlflow] = []
                by_trace[trace_id_mlflow].append({
                    "event_name": event_name,
                    "body": body or {},
                    "service": service,
                })
        for tid, events in by_trace.items():
            input_str = None
            inputs_list: List[str] = []
            output_str = None
            output_str_pa = None
            for ev in events:
                name = ev["event_name"]
                body = ev["body"]
                service = ev.get("service") or ""
                if name == "gen_ai.user.message" and service == "personal-assistant":
                    content = None
                    if isinstance(body, dict) and body.get("content"):
                        content = body["content"]
                    elif isinstance(body, str):
                        content = body
                    if content:
                        inputs_list.append(content)
                        if input_str is None:
                            input_str = content
                if name == "gen_ai.choice" and isinstance(body, dict) and body.get("finish_reason") == "stop":
                    msg = body.get("message") or {}
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if content:
                        output_str = content
                        if service == "personal-assistant":
                            output_str_pa = content
            out = output_str_pa if output_str_pa is not None else output_str
            if input_str is not None or out or inputs_list:
                result[tid] = {
                    "input": input_str or "",
                    "output": out or "",
                    "inputs_list": inputs_list,
                }
    except Exception as e:
        logger.warning("Could not parse OTel log file %s: %s", file_path, e)
    return result


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
    
    Each task gets a unique context_id to ensure conversations don't mix.
    
    Args:
        tasks: List of GAIA tasks to process
        agent_url: URL of the agent to call
    """
    import httpx
    import time
    from uuid import uuid4
    
    print(f"\nGenerating traces by calling agent for {len(tasks)} tasks...")
    
    for i, task in enumerate(tasks, 1):
        question = task.get("question", task.get("Question", ""))
        if not question:
            logger.warning(f"Task {i} has no question, skipping")
            continue
        
        # Log the exact question being sent for debugging
        print(f"  Calling agent for task {i}/{len(tasks)}: {question[:60]}...")
        
        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{agent_url}/v1/responses",
                    json={
                        "input": question,
                        "stream": False,
                    },
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
    print(f"\nIMPORTANT: Use distributed tracing — all agents should export to the same OTLP/MLflow endpoint.")
    print(f"  Start personal assistant, weather assistant, and research assistant with the same env:")
    print(f"    OTEL_EXPORTER_OTLP_ENDPOINT={mlflow_tracking_uri}")
    print(f"    OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id={experiment_id}")
    print(f"  Personal assistant only (for eval): OTEL_LOGS_EXPORTER=file OTEL_LOGS_FILE_PATH=data/otel_logs.jsonl")
    print(f"  With trace context propagation, one request = one trace (spans from PA, handoffs, tools); 3 tasks = 3 traces.")
    print(f"\nNote: MLflow's OTLP exporter automatically appends /v1/traces to the endpoint URL.")
    
    # Create data directory if it doesn't exist (for saving failures)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load evaluation tasks (includes test tasks and GAIA tasks)
    print("\nLoading evaluation tasks...")
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks (1 test task + 2 GAIA tasks)")
    
    agent_url = os.getenv("PERSONAL_ASSISTANT_AGENT_URL", "http://localhost:9999")
    print(f"\nPersonal assistant URL: {agent_url}")
    
    # Step 1: Generate traces by calling the agent
    # Record timestamp BEFORE generating traces so we only search for traces from this run
    from datetime import datetime
    trace_generation_start = datetime.now()
    print(f"\nTrace generation started at: {trace_generation_start}")
    print(f"This timestamp will be used to filter traces (only traces after this time)")
    
    # The agent app will create traces automatically via OTEL and send them to MLflow
    generate_traces_for_evaluation(tasks, agent_url)
    
    trace_generation_end = datetime.now()
    print(f"\nTrace generation completed at: {trace_generation_end}")
    print(f"Duration: {(trace_generation_end - trace_generation_start).total_seconds():.2f} seconds")
    
    # Step 2: Search and retrieve traces from MLflow
    # Get traces created after we started generating (to avoid old traces from previous runs)
    timestamp_ms = int(trace_generation_start.timestamp() * 1000)
    
    print(f"\nSearching for traces created after {trace_generation_start}...")
    print(f"(This should only include traces from this evaluation run)")
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

    # Require OTel log file: it is the canonical source for which traces to evaluate.
    # The personal assistant must be run with OTEL_LOGS_EXPORTER=file and OTEL_LOGS_FILE_PATH.
    log_path = os.getenv("OTEL_LOGS_FILE_PATH", "data/otel_logs.jsonl")
    if not Path(log_path).exists():
        print(f"\nERROR: OTel log file not found: {log_path}")
        print("Run the personal assistant with: OTEL_LOGS_EXPORTER=file OTEL_LOGS_FILE_PATH=data/otel_logs.jsonl")
        return
    log_io = _parse_otel_log_file(log_path)
    if not log_io:
        print(f"\nERROR: OTel log file is empty or has no parseable trace entries: {log_path}")
        return
    
    # Use log file as source of truth: evaluate only traces that appear there (one per task).
    wanted_ids = set(log_io.keys())
    before = len(traces)
    traces = traces[traces["trace_id"].isin(wanted_ids)]
    print(f"\nEvaluating {len(traces)} trace(s) from log (log has {len(log_io)} task(s)); {before - len(traces)} trace(s) from MLflow excluded (not in log).")
    if traces.empty:
        print("No traces from this run found in MLflow. Ensure the agent exported traces to the same experiment and after the script's start time.")
        return
    
    # Check if any traces were created before we started (shouldn't happen with proper timestamp filter)
    traces_before_start = 0
    for idx, trace_row in traces.iterrows():
        trace_timestamp = trace_row.get('request_time', 0)
        if isinstance(trace_timestamp, (int, float)):
            trace_timestamp_ms = int(trace_timestamp) if trace_timestamp > 1e12 else int(trace_timestamp * 1000)
            if trace_timestamp_ms < timestamp_ms:
                traces_before_start += 1
    
    if traces_before_start > 0:
        print(f"⚠️  WARNING: {traces_before_start} traces were created BEFORE trace generation started!")
        print(f"   This suggests old traces are being included. Timestamp filter: {timestamp_ms}")
    
    print(f"\n{len(traces)} traces ready for evaluation")
    
    # Populate request/response from log file; set inputs/outputs for Correctness scorer
    # Correctness scorer requires DataFrame columns named "inputs" and "outputs"
    if "response" not in traces.columns:
        traces["response"] = ""
    for idx, trace_row in traces.iterrows():
        tid = trace_row.get("trace_id")
        if tid and tid in log_io:
            traces.at[idx, "request"] = log_io[tid]["input"]
            traces.at[idx, "response"] = log_io[tid]["output"]
    if "request" in traces.columns:
        traces["inputs"] = traces["request"].fillna("")
    else:
        traces["inputs"] = ""
    if "response" in traces.columns:
        traces["outputs"] = traces["response"].fillna("")
    else:
        traces["outputs"] = ""
    
    # Step 3: Annotate traces with expectations (ground truth)
    # Match traces to tasks by comparing request content (prefer log-derived input when available)
    print("\nAnnotating traces with expectations...")
    
    # Create a mapping of questions to expected answers
    question_to_answer = {
        task.get("question", task.get("Question", "")): task.get("answer", task.get("Final answer", ""))
        for task in tasks
    }
    
    annotated_count = 0
    # Store trace_id -> expected_answer mapping for later use in DataFrame
    trace_expectations = {}
    
    for idx, trace_row in traces.iterrows():
        trace_id = trace_row.get('trace_id')
        if not trace_id:
            logger.warning(f"Trace row {idx} has no trace_id, skipping")
            continue
        
        # Extract the actual user question from the trace for matching to a task.
        # Prefer log-derived input; use inputs_list to find the first message that matches a task
        # (handoff traces may have research/weather sub-query as first PA message in file order).
        user_question = None
        inputs_list: List[str] = []
        if trace_id in log_io:
            rec = log_io[trace_id]
            if rec.get("input"):
                user_question = rec["input"]
            inputs_list = rec.get("inputs_list") or []
        # Fallback: span attributes gen_ai.prompt.*.content (where role="user") or request column
        if not user_question and "spans" in trace_row and trace_row["spans"]:
            try:
                spans = trace_row['spans']
                if isinstance(spans, list) and len(spans) > 0:
                    first_span = spans[0]
                    if isinstance(first_span, dict) and 'attributes' in first_span:
                        attrs = first_span['attributes']
                        for i in range(6):
                            prompt_role_key = f'gen_ai.prompt.{i}.role'
                            prompt_content_key = f'gen_ai.prompt.{i}.content'
                            if prompt_role_key in attrs and attrs[prompt_role_key] == '"user"':
                                if prompt_content_key in attrs:
                                    user_question = attrs[prompt_content_key].strip('"')
                                    break
            except Exception as e:
                logger.debug(f"Could not extract user question from spans: {e}")
        if not user_question and 'request' in trace_row:
            trace_request = trace_row['request']
            for question in question_to_answer.keys():
                if question in trace_request:
                    user_question = question
                    break

        # Match trace to task: try primary user_question first, then any message in inputs_list
        # that matches a task (so we get expected_response even when first PA message is a handoff sub-query).
        expected_answer = None
        def _match_question(q: str) -> str | None:
            if not q:
                return None
            if q in question_to_answer:
                return question_to_answer[q]
            for question, answer in question_to_answer.items():
                if question in q or q in question:
                    return answer
            return None

        if user_question:
            expected_answer = _match_question(user_question)
        if expected_answer is None and inputs_list:
            for msg in inputs_list:
                expected_answer = _match_question(msg)
                if expected_answer is not None:
                    user_question = msg
                    break
        
        # If we found a match, add the expectation
        if expected_answer:
            try:
                mlflow.log_expectation(
                    trace_id=trace_id,
                    name="expected_response",
                    value=expected_answer
                )
                # Store for DataFrame column
                trace_expectations[trace_id] = expected_answer
                annotated_count += 1
                logger.debug(f"Added expectation to trace {trace_id} for question: {user_question[:50] if user_question else 'N/A'}...")
            except Exception as e:
                logger.warning(f"Could not add expectation to trace {trace_id}: {e}")
        else:
            logger.warning(f"Could not match trace {trace_id} to any task. User question: {user_question[:100] if user_question else 'N/A'}...")
    
    print(f"Annotated {annotated_count} traces with expectations")
    
    # Add expectations column to DataFrame for MLflow evaluation
    # The Correctness scorer looks for "expected_response" or "expected_facts" in the expectations dict
    expectations_list = []
    for idx, trace_row in traces.iterrows():
        trace_id = trace_row.get('trace_id')
        if trace_id and trace_id in trace_expectations:
            expectations_list.append({"expected_response": trace_expectations[trace_id]})
        else:
            expectations_list.append({})
    
    traces["expectations"] = expectations_list
    
    if traces.empty:
        print("\nNo traces to evaluate. Exiting.")
        return

    # Step 4: Define scorers for MLflow evaluation
    # Correctness scorer compares against "expected_response" field in expectations
    scorers = [
        Correctness(),
    ]
    
    print("\nRunning MLflow evaluation on traces...")
    
    # Run MLflow evaluation on traces
    # MLflow automatically extracts inputs, outputs, and expectations from trace objects
    results = mlflow.genai.evaluate(
        data=traces,
        scorers=scorers,
    )
    
    print("\n✨ Evaluation completed!")
    print(f"Results logged to MLflow experiment: {experiment_name}")
    print(f"View results at: {mlflow_tracking_uri}")


if __name__ == "__main__":
    asyncio.run(main())
