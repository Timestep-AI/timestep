/**
 * NewsAssistantServer implements the ChatKitServer interface for the News Guide demo.
 */

import { run } from "@openai/agents";
import {
  ChatKitServer,
  AgentContext,
  streamAgentResponse,
} from "chatkit-server";
import type {
  ThreadMetadata,
  ThreadStreamEvent,
  UserMessageItem,
  Action,
  WidgetItem,
  ThreadItemDoneEvent,
} from "chatkit-server";
import * as path from "node:path";
import { MemoryStore } from "./memory_store.ts";
import { ArticleStore } from "./data/article_store.ts";
import { EventStore, type EventRecord } from "./data/event_store.ts";
import { NewsGuideThreadItemConverter } from "./thread_item_converter.ts";
import { newsAgent, type NewsAgentContext } from "./agents/news_agent.ts";
import { eventFinderAgent, type EventFinderContext } from "./agents/event_finder_agent.ts";
import { puzzleAgent, type PuzzleAgentContext } from "./agents/puzzle_agent.ts";
import { titleAgent } from "./agents/title_agent.ts";
import { buildEventListWidget } from "./widgets/event_list_widget.ts";

export class NewsAssistantServer extends ChatKitServer<Record<string, unknown>> {
  articleStore: ArticleStore;
  eventStore: EventStore;
  private threadItemConverter = new NewsGuideThreadItemConverter();
  private titleAgentInstance = titleAgent;

  constructor() {
    const store = new MemoryStore();
    super(store);

    const dataDir = path.resolve(Deno.cwd(), "apps/news-guide/backend-ts/data");
    this.articleStore = new ArticleStore(dataDir);
    this.eventStore = new EventStore(dataDir);
  }

  async *respond(
    thread: ThreadMetadata,
    item: UserMessageItem | null,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    const updatingThreadTitle = this.maybeUpdateThreadTitle(thread, item);

    const itemsPage = await this.store.loadThreadItems(thread.id, null, 20, "desc", context);
    const items = itemsPage.data.reverse();
    const inputItems = await this.threadItemConverter.toAgentInput(items);

    const { agent, agentContext } = this.selectAgent(thread, item, context);

    const result = await run(agent, inputItems, {
      stream: true,
      context: agentContext,
    });

    for await (const event of streamAgentResponse(agentContext, result)) {
      yield event;
    }

    await updatingThreadTitle;
  }

  async *action(
    thread: ThreadMetadata,
    action: Action<string, unknown>,
    sender: WidgetItem | null,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    if (action.type === "open_article") {
      for await (const event of this.handleOpenArticleAction(thread, action, context)) {
        yield event;
      }
      return;
    }
    if (action.type === "view_event_details") {
      for await (const event of this.handleViewEventDetailsAction(action, sender)) {
        yield event;
      }
      return;
    }
  }

  private async *handleOpenArticleAction(
    thread: ThreadMetadata,
    action: Action<string, unknown>,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    const payload = action.payload as Record<string, unknown> | null;
    const articleId = payload?.id as string | undefined;
    if (!articleId) return;

    const metadata = this.articleStore.getMetadata(articleId);
    const title = metadata?.title as string | undefined;
    const message = title
      ? `Want a quick summary of _${title}_ or have any questions about it?`
      : "Want a quick summary or have any questions about this article?";

    yield {
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: thread.id,
        id: this.store.generateItemId("message", thread, context),
        created_at: new Date(),
        content: [{ type: "output_text", text: message }],
      },
    } as ThreadItemDoneEvent;
  }

  private async *handleViewEventDetailsAction(
    action: Action<string, unknown>,
    sender: WidgetItem | null
  ): AsyncIterator<ThreadStreamEvent> {
    const payload = action.payload as Record<string, unknown> | null;
    const selectedEventId = payload?.id as string | undefined;
    const eventIds = ((payload?.event_ids || []) as unknown[])
      .filter((e): e is string => typeof e === "string" && e.length > 0);
    const isSelected = Boolean(payload?.is_selected);

    if (isSelected || !selectedEventId || eventIds.length === 0 || !sender) {
      return;
    }

    const records: EventRecord[] = [];
    for (const eventId of eventIds) {
      const record = this.eventStore.getEvent(eventId);
      if (record) records.push(record);
    }

    const updatedWidget = buildEventListWidget(records, selectedEventId);
    if (!updatedWidget) return;

    yield {
      type: "thread.item.updated",
      item_id: sender.id,
      update: { type: "widget.root", widget: updatedWidget },
    };
  }

  private selectAgent(
    thread: ThreadMetadata,
    item: UserMessageItem | null,
    context: Record<string, unknown>
  ): { agent: unknown; agentContext: AgentContext } {
    const toolChoice = this.resolveToolChoice(item);

    if (toolChoice === "event_finder") {
      const eventContext: EventFinderContext = new AgentContext({
        thread,
        store: this.store,
        requestContext: context,
      }) as EventFinderContext;
      eventContext.events = this.eventStore;
      return { agent: eventFinderAgent, agentContext: eventContext };
    }

    if (toolChoice === "puzzle") {
      const puzzleContext: PuzzleAgentContext = new AgentContext({
        thread,
        store: this.store,
        requestContext: context,
      }) as PuzzleAgentContext;
      return { agent: puzzleAgent, agentContext: puzzleContext };
    }

    const newsContext: NewsAgentContext = new AgentContext({
      thread,
      store: this.store,
      requestContext: context,
    }) as NewsAgentContext;
    newsContext.articles = this.articleStore;
    newsContext.articleId = (context as { article_id?: string }).article_id;
    return { agent: newsAgent, agentContext: newsContext };
  }

  private resolveToolChoice(item: UserMessageItem | null): string | null {
    if (!item?.inference_options) return null;
    const toolChoice = item.inference_options.tool_choice;
    if (toolChoice && typeof (toolChoice as { id?: string }).id === "string") {
      return (toolChoice as { id: string }).id;
    }
    return null;
  }

  private async maybeUpdateThreadTitle(
    thread: ThreadMetadata,
    userMessage: UserMessageItem | null
  ): Promise<void> {
    if (userMessage === null || thread.title !== null) return;

    const runResult = await run(
      this.titleAgentInstance,
      await this.threadItemConverter.toAgentInput(userMessage)
    );
    let modelResult: string = runResult.finalOutput;
    modelResult = modelResult.charAt(0).toUpperCase() + modelResult.slice(1);
    thread.title = modelResult.replace(/\.$/, "");
  }
}

