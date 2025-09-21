default: publish

deno-server-dev:
	@echo "🦕 Starting Deno/Oak A2A Server with auto-reload..."
	cd typescript/timestep && deno task dev

deno-server:
	@echo "🦕 Starting Deno/Oak A2A Server..."
	cd typescript/timestep && deno task start

timestep-cli-server:
	@echo "🚀 Starting Timestep cli server..."
	cd typescript/timestep && npm run build
	cd typescript/timestep && node dist/cli.js stop
	cd typescript/timestep && node dist/cli.js server

timestep-cli-chat:
	@echo "🚀 Starting Timestep cli chat..."
	cd typescript/timestep && npx tsx src/cli.tsx chat

timestep-cli-get-version:
	@echo "🚀 Starting Timestep cli get-version..."
	cd typescript/timestep && npx tsx src/cli.tsx get-version

timestep-cli-list-agents:
	@echo "🚀 Starting Timestep cli list-agents..."
	cd typescript/timestep && npx tsx src/cli.tsx list-agents

timestep-cli-list-chats:
	@echo "🚀 Starting Timestep cli list-chats..."
	cd typescript/timestep && npx tsx src/cli.tsx list-chats

timestep-cli-list-models:
	@echo "🚀 Starting Timestep cli list-models..."
	cd typescript/timestep && npx tsx src/cli.tsx list-models

timestep-cli-list-tools:
	@echo "🚀 Starting Timestep cli list-tools..."
	cd typescript/timestep && npx tsx src/cli.tsx list-tools

timestep-cli-list-traces:
	@echo "🚀 Starting Timestep cli list-traces..."
	cd typescript/timestep && npx tsx src/cli.tsx list-traces

timestep-cli-list-mcp-servers:
	@echo "🚀 Starting Timestep cli list-mcp-servers..."
	cd typescript/timestep && npx tsx src/cli.tsx list-mcp-servers

timestep-cli-list-model-providers:
	@echo "🚀 Starting Timestep cli list-model-providers..."
	cd typescript/timestep && npx tsx src/cli.tsx list-model-providers

timestep-cli-list-all: timestep-cli-list-agents timestep-cli-list-chats timestep-cli-list-models timestep-cli-list-tools timestep-cli-list-traces timestep-cli-list-mcp-servers timestep-cli-list-model-providers

run-a2a-inspector:
	@echo "🔍 Running A2A Inspector..."
	cd bash && ./run-a2a-inspector.sh

test-built-in-weather:
	@echo "📘 Running TypeScript A2A Client tests..."
	cd typescript/timestep && npx tsx src/a2aClient.ts --agentId 00000000-0000-0000-0000-000000000000 --auto-approve --user-input "What's the weather in Oakland and San Francisco?"

test-built-in-weather-cli:
	@echo "📘 Running TypeScript A2A Client tests..."
	cd typescript/timestep && npx tsx src/cli.tsx chat --agentId 00000000-0000-0000-0000-000000000000 --auto-approve --user-input "What's the weather in Oakland and San Francisco?"

test-built-in-weather-cli-ag-ui:
	@echo "📘 Running TypeScript AG-UI Client tests..."
	cd typescript/timestep && npx tsx src/cli.tsx ag-ui-chat --agentId 00000000-0000-0000-0000-000000000000 --user-input "What's the weather in Oakland and San Francisco?"

test-e2e: timestep-cli-server timestep-cli-list-all test-built-in-weather-cli

publish:
	@echo "📘 Publishing Timestep..."
	./bash/bump-version.sh
	cd typescript/timestep && npx prettier --write .
	cd typescript/timestep/examples && deno run --allow-read --allow-write --allow-run check-examples.ts
	make test-e2e
	git add .
	@VERSION=$$(grep '"version"' typescript/timestep/package.json | cut -d'"' -f4); \
	git commit -m "Bump version to $$VERSION"
	git push
	cd typescript/timestep && npm publish
