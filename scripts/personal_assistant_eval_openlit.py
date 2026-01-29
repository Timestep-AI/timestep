# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "openlit",
#   "datasets",
# ]
# ///

#!/usr/bin/env python3
"""Run GAIA benchmark evaluation against personal assistant agent using OpenLIT."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Initialize OpenLIT (following Grafana AI Observability guide)
# OpenLIT automatically picks up standard OTEL env vars:
# - OTEL_EXPORTER_OTLP_ENDPOINT
# - OTEL_EXPORTER_OTLP_HEADERS
# - OTEL_RESOURCE_ATTRIBUTES (for service.name, deployment.environment, etc.)
import openlit

# Set OTEL_RESOURCE_ATTRIBUTES if not already set (standard OTEL way)
if "OTEL_RESOURCE_ATTRIBUTES" not in os.environ:
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "service.name=timestep,deployment.environment=evaluation"

# Initialize OpenLIT - it will read configuration from standard OTEL env vars
openlit.init(disable_metrics=False)

otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not set")
otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "not set")
resource_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "not set")
print(f"OpenLIT initialized with standard OTEL env vars")
print(f"OTLP endpoint: {otlp_endpoint}")
print(f"OTLP headers: {'set' if otlp_headers != 'not set' else 'not set'}")
print(f"Resource attributes: {resource_attrs}")

# Initialize OpenLIT evaluator
evaluator = openlit.evals.All(
    provider="openai",
    collect_metrics=True  # Automatically sends to Grafana
)


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


def check_answer(got: str, expected: str) -> bool:
    """Simple answer checker."""
    return got.strip().lower() == expected.strip().lower()


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert Pydantic model or other object to dict."""
    if isinstance(obj, dict):
        return obj
    # Try Pydantic v2 model_dump() first
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Try Pydantic v1 dict() method
    if hasattr(obj, "dict"):
        return obj.dict()
    # Try to access as attributes if it's a Pydantic model
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    # Fallback: return as-is (will be converted to string in error cases)
    return obj


def print_summary(results: List[Dict[str, Any]]):
    """Print evaluation summary."""
    if not results:
        print("No results to summarize")
        return
    
    passed = sum(1 for r in results if r.get("correct") is True)
    total = sum(1 for r in results if r.get("correct") is not None)
    
    if total > 0:
        print(f"\nResults: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    # Show OpenLIT evaluation summary
    # Each evaluator.measure() returns a single JsonOutput object with:
    # - score: float
    # - evaluation: "hallucination", "bias_detection", or "toxicity_detection"
    # - classification: category or "none"
    # - explanation: string
    # - verdict: "yes" or "no"
    openlit_results = [r.get("openlit_results") for r in results if r.get("openlit_results")]
    if openlit_results:
        # Convert Pydantic models to dicts for safe access
        openlit_dicts = [to_dict(r) for r in openlit_results]
        # Check evaluation field values as per OpenLIT documentation
        hallucinations = sum(1 for r in openlit_dicts if r.get("evaluation") == "hallucination" and r.get("verdict") == "yes")
        bias = sum(1 for r in openlit_dicts if r.get("evaluation") == "bias_detection" and r.get("verdict") == "yes")
        toxicity = sum(1 for r in openlit_dicts if r.get("evaluation") == "toxicity_detection" and r.get("verdict") == "yes")
        print(f"\nOpenLIT Evaluations:")
        print(f"  Hallucinations detected: {hallucinations}")
        print(f"  Bias detected: {bias}")
        print(f"  Toxicity detected: {toxicity}")
    
    # Show failures
    failures = [r for r in results if r.get("correct") is False]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:10]:
            print(f"\n  Case ID: {f.get('case_id', 'unknown')}")
            print(f"  Input: {f['input'][:100]}...")
            print(f"  Expected: {f['expected']}")
            print(f"  Got: {f['output'][:100]}...")
            if f.get("openlit_results"):
                openlit_dict = to_dict(f["openlit_results"])
                print(f"  OpenLIT: {openlit_dict}")


async def main():
    # Create data directory if it doesn't exist (for saving failures)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load GAIA validation tasks directly from HuggingFace dataset
    print("Loading GAIA validation dataset...")
    tasks = load_gaia_tasks()
    print(f"Loaded {len(tasks)} GAIA tasks")
    
    # For initial testing, only use the first task
    tasks = tasks[:1]
    print(f"Using first task only for initial testing: {len(tasks)} task(s)")
    
    # Connect to personal assistant agent (assumes it's already running)
    agent_url = os.getenv("PERSONAL_ASSISTANT_AGENT_URL", "http://localhost:9999")
    print(f"Connecting to personal assistant at {agent_url}")
    
    # Verify agent is reachable
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            # Try to reach the agent's health endpoint or root
            response = await client.get(f"{agent_url}/", timeout=5.0)
            print(f"Agent is reachable (status: {response.status_code})")
    except Exception as e:
        print(f"Warning: Could not verify agent is running at {agent_url}: {e}")
        print("Make sure the personal assistant is running before starting evaluation.")
        print("Start it with: uv run scripts/personal_assistant_app.py")
        return
    
    # Convert GAIA tasks to test cases
    test_cases = []
    for task in tasks:
        test_case = {
            "id": task.get("task_id", task.get("id", f"task-{len(test_cases)}")),
            "input": task.get("question", task.get("input", "")),
            "expected": task.get("answer", task.get("expected_output"))
        }
        # Note: GAIA file attachments would need to be downloaded separately
        # For now, we skip file context (can be added later if needed)
        test_cases.append(test_case)
    
    print(f"Running {len(test_cases)} test cases...")
    
    # Run evaluation using agent's HTTP endpoint
    results = []
    
    async with httpx.AsyncClient() as client:
        for case in test_cases:
            case_id = case.get("id", f"case-{len(results)}")
            input_text = case.get("input", "")
            expected_output = case.get("expected")
            contexts = case.get("contexts")
            
            # Make HTTP request to /v1/responses endpoint
            try:
                response = await client.post(
                    f"{agent_url}/v1/responses",
                    json={"input": input_text, "stream": False},
                    timeout=300.0  # 5 minute timeout for complex tasks
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
                
                # Basic correctness check
                correct = None
                if expected_output:
                    correct = check_answer(output, expected_output)
                
                # OpenLIT programmatic evaluation
                openlit_results = None
                try:
                    # Run comprehensive evaluation (hallucination, bias, toxicity)
                    # Metrics automatically sent to Grafana via collect_metrics=True
                    # The evaluator makes OpenAI API calls internally which OpenLIT instruments
                    print(f"Running OpenLIT evaluation for case {case_id}...")
                    openlit_results = evaluator.measure(
                        prompt=input_text,
                        contexts=contexts or [],
                        text=output
                    )
                    print(f"OpenLIT evaluation completed for case {case_id}")
                except Exception as e:
                    logger.warning(f"OpenLIT evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                result = {
                    "case_id": case_id,
                    "input": input_text,
                    "output": output,
                    "expected": expected_output,
                    "correct": correct,
                    "openlit_results": openlit_results
                }
                
                results.append(result)
                print(f"Completed case {case_id}: {'✓' if correct else '✗' if correct is False else '?'}")
                
            except Exception as e:
                logger.error(f"Error running case {case_id}: {e}")
                results.append({
                    "case_id": case_id,
                    "input": input_text,
                    "output": "",
                    "expected": expected_output,
                    "correct": None,
                    "error": str(e)
                })
    
    # Print results
    print_summary(results)
    
    # Save failures
    failures = [r for r in results if r.get("correct") is False]
    if failures:
        # Convert JsonOutput objects to dicts for JSON serialization
        failures_serializable = []
        for f in failures:
            failure_dict = dict(f)
            if "openlit_results" in failure_dict and failure_dict["openlit_results"] is not None:
                failure_dict["openlit_results"] = to_dict(failure_dict["openlit_results"])
            failures_serializable.append(failure_dict)
        
        failures_file = data_dir / "gaia_failures.json"
        with open(failures_file, "w") as f:
            json.dump(failures_serializable, f, indent=2)
        print(f"\nFailures saved to {failures_file}")
    
    # Flush OpenLIT data to ensure all metrics/traces are sent before script exits
    try:
        from opentelemetry import trace
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, 'force_flush'):
            tracer_provider.force_flush()
            print("OpenLIT traces flushed")
    except Exception as e:
        logger.debug(f"Could not flush traces: {e}")
    
    try:
        from opentelemetry import metrics
        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, 'force_flush'):
            meter_provider.force_flush()
            print("OpenLIT metrics flushed")
    except Exception as e:
        logger.debug(f"Could not flush metrics: {e}")
    
    # Give OpenLIT time to send all data before script exits
    import time
    print("\nWaiting for OpenLIT to send data to Grafana Cloud...")
    time.sleep(2)  # Wait 2 seconds for data to be sent
    
    print(f"\nEvaluation metrics sent to Grafana Cloud")
    print(f"View in Grafana AI Observability dashboards")
    print(f"Resource attributes: {resource_attrs}")
    print(f"Note: It may take a few minutes for data to appear in the dashboard")


if __name__ == "__main__":
    asyncio.run(main())
