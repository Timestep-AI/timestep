# Timestep Test Harness

A Rust-based behavior test harness that runs the same tests against both Python and TypeScript implementations of the Timestep library.

## Overview

This harness ensures that both implementations behave identically by:
1. Defining test cases in a language-agnostic format (JSON/TOML)
2. Executing the same tests against both implementations
3. Comparing results and reporting discrepancies

## Setup

### Prerequisites

- Rust (latest stable)
- Python 3.11+ with `timestep` package installed
- Node.js 20+ with TypeScript dependencies installed

### Building

```bash
cd rust
cargo build --release
```

## Usage

### Run all tests

```bash
cargo run -- test
```

### Run specific test

```bash
cargo run -- test --filter "ollama_provider"
```

### Run against specific implementations

```bash
# Python only
cargo run -- test --python --no-typescript

# TypeScript only
cargo run -- test --typescript --no-python
```

### Generate HTML report

```bash
cargo run -- test --format html --output report.html
```

## Test Case Format

Test cases are defined in JSON format. See `tests/behavior/` for examples.

### Structure

```json
{
  "name": "test_name",
  "description": "Test description",
  "setup": {
    "provider_type": "ollama|multi",
    "provider_config": {
      "api_key": "...",
      "base_url": "..."
    }
  },
  "input": {
    "model_name": "llama3",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  },
  "expected": {
    "provider_type": "ollama",
    "model_name": "llama3",
    "should_succeed": true
  }
}
```

## Architecture

- **Rust Harness**: Orchestrates test execution
- **Python Bridge**: Executes tests against Python implementation
- **TypeScript Bridge**: Executes tests against TypeScript implementation
- **Test Runners**: Python/TypeScript scripts that execute individual tests

## Future Enhancements

- [ ] HTTP mock server for external dependencies
- [ ] Performance benchmarking
- [ ] Coverage analysis
- [ ] CI/CD integration
- [ ] Rust implementation (third implementation for validation)

