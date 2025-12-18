/**
 * ChatKit server that streams responses from a single assistant.
 */

import { Agent, run } from "@openai/agents";
import type {
  ThreadMetadata,
  ThreadStreamEvent,
  UserMessageItem,
} from "chatkit-server";
import {
  AgentContext,
  ChatKitServer,
  simpleToAgentInput,
  streamAgentResponse,
} from "chatkit-server";
import { MemoryStore } from "./memory_store.ts";

const MAX_RECENT_ITEMS = 30;
const MODEL = "gpt-4.1-mini";

// Define the context type
type RequestContext = Record<string, unknown>;

// Create the assistant agent
const assistantAgent = new Agent({
  model: MODEL,
  name: "Starter Assistant",
  instructions:
    "You are a concise, helpful assistant. " +
    "Keep replies short and focus on directly answering " +
    "the user's request.",
});

export class StarterChatServer extends ChatKitServer<RequestContext> {
  declare store: MemoryStore;

  constructor() {
    const store = new MemoryStore();
    super(store);
    this.store = store;
  }

  async *respond(
    thread: ThreadMetadata,
    _item: UserMessageItem | null,
    context: RequestContext
  ): AsyncGenerator<ThreadStreamEvent, void, unknown> {
    const itemsPage = await this.store.loadThreadItems(
      thread.id,
      null,
      MAX_RECENT_ITEMS,
      "desc",
      context
    );
    const items = [...itemsPage.data].reverse();

    const agentInput = await simpleToAgentInput(items);

    const agentContext = new AgentContext({
      thread,
      store: this.store,
      requestContext: context,
    });

    const result = await run(assistantAgent, agentInput, {
      stream: true,
      context: agentContext,
    });

    for await (const event of streamAgentResponse(agentContext, result)) {
      yield event;
    }
  }
}

