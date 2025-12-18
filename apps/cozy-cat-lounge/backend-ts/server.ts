/**
 * CatAssistantServer implements the ChatKitServer interface.
 */

import { run } from "@openai/agents";
import type {
  Action,
  Attachment,
  StreamOptions,
  ThreadMetadata,
  ThreadStreamEvent,
  UserMessageItem,
  WidgetItem,
  HiddenContextItem,
  AssistantMessageItem,
  ThreadItemDoneEvent,
  ThreadItemReplacedEvent,
} from "chatkit-server";
import {
  AgentContext,
  ChatKitServer,
  streamAgentResponse,
  ThreadItemConverter,
} from "chatkit-server";

import { catAgent, type CatAgentContext } from "./cat_agent.ts";
import { CatStore } from "./cat_store.ts";
import { MemoryStore } from "./memory_store.ts";
import { BasicThreadItemConverter } from "./thread_item_converter.ts";
import { buildNameSuggestionsWidget, type CatNameSuggestion } from "./widgets/name_suggestions_widget.ts";

const MAX_RECENT_ITEMS = 20;

type RequestContext = Record<string, unknown>;

export class CatAssistantServer extends ChatKitServer<RequestContext> {
  declare store: MemoryStore;
  catStore: CatStore;
  threadItemConverter: BasicThreadItemConverter;

  constructor() {
    const store = new MemoryStore();
    super(store);
    this.store = store;
    this.catStore = new CatStore();
    this.threadItemConverter = new BasicThreadItemConverter();
  }

  // -- Required overrides --------------------------------------------------
  async *action(
    thread: ThreadMetadata,
    action: Action<string, unknown>,
    sender: WidgetItem | null,
    context: RequestContext
  ): AsyncGenerator<ThreadStreamEvent, void, unknown> {
    if (action.type === "cats.select_name") {
      for await (const event of this._handleSelectNameAction(
        thread,
        action.payload as Record<string, unknown>,
        sender,
        context
      )) {
        yield event;
      }
      return;
    }
    return;
  }

  async *respond(
    thread: ThreadMetadata,
    _item: UserMessageItem | null,
    context: RequestContext
  ): AsyncGenerator<ThreadStreamEvent, void, unknown> {
    // Create the agent context with the cat store
    const agentContext = new AgentContext({
      thread,
      store: this.store,
      requestContext: context,
    }) as CatAgentContext;
    
    // Add cat store to the context
    (agentContext as CatAgentContext).cats = this.catStore;

    // Load all items in the thread to send to the agent as input
    const itemsPage = await this.store.loadThreadItems(
      thread.id,
      null,
      MAX_RECENT_ITEMS,
      "desc",
      context
    );

    // Runner expects the most recent message to be last.
    const items = [...itemsPage.data].reverse();

    // Translate ChatKit thread items into agent input.
    const agentInput = await this.threadItemConverter.toAgentInput(items);

    const result = await run(catAgent, agentInput, {
      stream: true,
      context: agentContext,
    });

    for await (const event of streamAgentResponse(agentContext, result)) {
      yield event;
    }
  }

  getStreamOptions(_thread: ThreadMetadata, _context: RequestContext): StreamOptions {
    // Don't allow stream cancellation because most cat-lounge interactions update the cat's state.
    return { allowCancel: false };
  }

  async toMessageContent(_input: Attachment): Promise<unknown> {
    throw new Error("File attachments are not supported in this demo.");
  }

  // -- Helpers -------------------------------------------------------------
  private async *_handleSelectNameAction(
    thread: ThreadMetadata,
    payload: Record<string, unknown>,
    sender: WidgetItem | null,
    context: RequestContext
  ): AsyncGenerator<ThreadStreamEvent, void, unknown> {
    const name = ((payload.name as string) || "").trim();
    if (!name || !sender) {
      return;
    }

    const rawOptions = (payload.options as Array<{ name: string; reason?: string }>) || [];
    const options: CatNameSuggestion[] = rawOptions.map((opt) => ({
      name: opt.name,
      reason: opt.reason,
    }));

    const currentState = await this.catStore.load(thread.id);
    const isAlreadyNamed = currentState.name !== "Unnamed Cat";
    const selection = isAlreadyNamed ? currentState.name : name;
    const widget = buildNameSuggestionsWidget(options, selection);

    yield {
      type: "thread.item.replaced",
      item: { ...sender, widget },
    } as ThreadItemReplacedEvent;

    if (isAlreadyNamed) {
      const messageItem: AssistantMessageItem = {
        type: "assistant_message",
        id: this.store.generateItemId("message", thread, context),
        thread_id: thread.id,
        createdAt: new Date(),
        content: [
          { type: "output_text", text: `${currentState.name} already has a name, so we can't rename them.` },
        ],
      };
      yield { type: "thread.item.done", item: messageItem } as ThreadItemDoneEvent;
      return;
    }

    // Save the name in the cat store and update the thread title in the chatkit store.
    const state = await this.catStore.mutate(thread.id, (s) => s.rename(name));
    const title = `${state.name}'s Lounge`;
    thread.title = title;
    await this.store.saveThread(thread, context);

    // Add a hidden context item so that future agent input will know that the user
    // has selected a name from the suggestions list.
    await this.store.addThreadItem(
      thread.id,
      {
        type: "hidden_context",
        id: this.store.generateItemId("message", thread, context),
        thread_id: thread.id,
        createdAt: new Date(),
        content: `<CAT_NAME_SELECTED>${state.name}</CAT_NAME_SELECTED>`,
      } as HiddenContextItem,
      context
    );

    const messageItem: AssistantMessageItem = {
      type: "assistant_message",
      id: this.store.generateItemId("message", thread, context),
      thread_id: thread.id,
      createdAt: new Date(),
      content: [
        { type: "output_text", text: `Love that choice. ${state.name}'s profile card is now ready. Would you like to check it out?` },
      ],
    };
    yield { type: "thread.item.done", item: messageItem } as ThreadItemDoneEvent;
  }
}

export function createChatkitServer(): CatAssistantServer {
  return new CatAssistantServer();
}

