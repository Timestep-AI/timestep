AGENT_ID := 00000000-0000-0000-0000-000000000000
BASE_SERVER_URL := https://ohzbghitbjryfpmucgju.supabase.co/functions/v1/server 
USER_INPUT := "What's the weather in Oakland and San Francisco?"

default: publish

test-built-in-weather-cli: timestep-cli-server
	@echo "📘 Running TypeScript A2A Client tests..."
	cd typescript/timestep && npx tsx src/cli.tsx chat \
		--agentId $(AGENT_ID) \
		--auto-approve \
		--user-input $(USER_INPUT)

test-built-in-weather-cli-supabase-edge-function:
	@echo "📘 Running TypeScript A2A Client tests..."
	cd typescript/timestep && npx tsx src/cli.tsx chat \
		--agentId $(AGENT_ID) \
		--auto-approve \
		--auth-token $(AUTH_TOKEN) \
		--baseServerUrl $(BASE_SERVER_URL) \
		--user-input $(USER_INPUT)

timestep-cli-chat: timestep-cli-server
	@echo "🚀 Starting Timestep cli chat..."
	cd typescript/timestep && npx tsx src/cli.tsx chat

timestep-cli-chat-resume: timestep-cli-server
	@echo "🚀 Starting Timestep cli chat resume..."
	cd typescript/timestep/ && npx tsx src/cli.tsx chat \
		--agentId $(AGENT_ID) \
		--contextId $(CONTEXT_ID)

timestep-cli-chat-remote:
	@echo "🚀 Starting Timestep cli chat remote..."
	cd typescript/timestep && npx tsx src/cli.tsx chat \
		--agentId $(AGENT_ID) \
		--auth-token $(AUTH_TOKEN) \
		--baseServerUrl $(BASE_SERVER_URL)

timestep-cli-get-full-conversation-history:
	@echo "🚀 Starting Timestep cli get full conversation history..."
	cd typescript/timestep && npx tsx src/cli.tsx get-full-conversation-history \
		--agentId $(AGENT_ID) \
		--contextId $(CONTEXT_ID)

timestep-cli-server:
	@echo "🚀 Starting Timestep cli server..."
	cd typescript/timestep && npm run build
	cd typescript/timestep && node dist/cli.js stop
	cd typescript/timestep && node dist/cli.js server

publish: test-built-in-weather-cli
	@echo "📘 Publishing Timestep..."
	./bash/bump-version.sh
	cd typescript/timestep && npx prettier --write .
	cd typescript/timestep/examples && deno run --allow-read --allow-write --allow-run check-examples.ts
	git add .
	@VERSION=$$(grep '"version"' typescript/timestep/package.json | cut -d'"' -f4); \
	git commit -m "Bump version to $$VERSION"
	git push
	cd typescript/timestep && npm publish

run-a2a-inspector:
	@echo "🔍 Running A2A Inspector..."
	cd bash && ./run-a2a-inspector.sh
