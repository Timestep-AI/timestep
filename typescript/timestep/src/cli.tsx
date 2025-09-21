#!/usr/bin/env node
import React from 'react';
import {render} from 'ink';
import meow from 'meow';
import App from './app.js';

const cli = meow(
	`
	Usage
	  $ timestep [command]

	Commands
	  server                   Start the agents server
	  stop                     Stop the agents server
	  chat                     Start interactive chat with an agent
	  get-version              Show version information
	  list-agents              List all agents from the server
	  list-chats               List all chats from the server
	  list-models              List all models from the server
	  list-tools               List all tools from the server
	  list-traces              List all traces from the server
	  list-settings-api-keys   List all configured API keys
	  list-settings-mcp-servers List all configured MCP servers
	  list-settings-model-providers List all configured model providers

	Options
	  --name                   Your name (for greeting)
	  --agentId                Agent ID to use for chat command
	  --auto-approve           Auto-approve tool calls for chat command
	  --user-input             User input to send automatically for chat command

	Examples
	  $ timestep --name=Jane
	  Hello, Jane

	  $ timestep server
	  Start the agents server

	  $ timestep stop
	  Stop the agents server

	  $ timestep chat
	  Start interactive chat with an agent

	  $ timestep chat --agentId 00000000-0000-0000-0000-000000000000 --auto-approve --user-input "What's the weather?"
	  Start chat with specific agent, auto-approve tools, and send message

	  $ timestep list-agents
	  List all agents from the server

	  $ timestep list-chats
	  List all chats from the server

	  $ timestep list-models
	  List all models from the server

	  $ timestep list-tools
	  List all tools from the server

	  $ timestep list-traces
	  List all traces from the server

	  $ timestep list-settings-api-keys
	  List all configured API keys

	  $ timestep list-settings-mcp-servers
	  List all configured MCP servers

	  $ timestep list-settings-model-providers
	  List all configured model providers
`,
	{
		importMeta: import.meta,
		flags: {
			name: {
				type: 'string',
			},
			agentId: {
				type: 'string',
			},
			autoApprove: {
				type: 'boolean',
				alias: 'auto-approve',
			},
			userInput: {
				type: 'string',
				alias: 'user-input',
			},
			baseServerUrl: {
				type: 'string',
				alias: 'base-server-url',
			},
			authToken: {
				type: 'string',
				alias: 'auth-token',
			},
		},
	},
);

const command = cli.input[0];

render(<App name={cli.flags.name} command={command} flags={cli.flags} />);
