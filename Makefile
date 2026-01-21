# Makefile for Timestep

.PHONY: help test-example-python test-example-typescript

default: help

help:
	@echo "Available targets:"
	@echo "  make help                    - Show this help message"
	@echo "  make test-example-python      - Run the Python example (A2A + MCP servers + test client)"
	@echo "  make test-example-typescript - Run the TypeScript example (A2A + MCP servers + test client) [PENDING v2 SDK]"
	@echo ""
	@echo "Environment variables:"
	@echo "  TEST_MESSAGE                 - Custom message to send to the test client (default: 'What's the weather in Oakland?')"
	@echo "  OPENAI_API_KEY               - OpenAI API key (required)"
	@echo ""
	@echo "Note: TypeScript example is pending MCP SDK v2 release (expected Q1 2026)"

test-example-python:
	@echo "Starting servers..."
	@OPENAI_API_KEY=$$OPENAI_API_KEY docker compose -f examples/python/compose.yml up -d
	@echo "Waiting for servers to be ready..."
	@timeout 30 bash -c 'until curl -s http://localhost:8000/agents/00000000-0000-0000-0000-000000000000/.well-known/agent-card.json > /dev/null 2>&1 && curl -s http://localhost:8080/mcp > /dev/null 2>&1; do sleep 0.5; done' || (echo "Servers failed to start"; docker compose -f examples/python/compose.yml logs; exit 1)
	@echo "Following logs and running test client..."
	@(docker compose -f examples/python/compose.yml logs -f & echo $$! > /tmp/compose-logs.pid) && \
	uv run examples/python/test_client.py "$${TEST_MESSAGE:-What's the weather in Oakland?}"; \
	CLIENT_EXIT=$$?; \
	kill $$(cat /tmp/compose-logs.pid 2>/dev/null) 2>/dev/null || true; \
	docker compose -f examples/python/compose.yml down; \
	exit $$CLIENT_EXIT

test-example-typescript:
	@echo "=========================================="
	@echo "TypeScript Example - PENDING v2 SDK RELEASE"
	@echo "=========================================="
	@echo ""
	@echo "The TypeScript implementation is incomplete and pending the release of"
	@echo "@modelcontextprotocol/sdk v2 (expected Q1 2026)."
	@echo ""
	@echo "Current status:"
	@echo "  - v1.x SDK doesn't export the HTTP transport classes we need"
	@echo "  - v2 APIs (McpServer, StreamableHTTPServerTransport, etc.) are not yet published"
	@echo "  - We explored git dependencies but they require package.json (conflicts with inline deps)"
	@echo ""
	@echo "See examples/typescript/*.ts files for details and TODOs."
	@echo ""
	@exit 1
