/**
 * MetroMapServer implements the ChatKitServer interface for the metro-map demo.
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
  HiddenContextItem,
} from "chatkit-server";
import * as path from "node:path";
import { MemoryStore } from "./memory_store.ts";
import { MetroMapStore } from "./data/metro_map_store.ts";
import { MetroMapThreadItemConverter } from "./thread_item_converter.ts";
import { metroMapAgent, type MetroAgentContext } from "./agents/metro_map_agent.ts";
import { titleAgent } from "./agents/title_agent.ts";
import { buildLineSelectWidget } from "./widgets/line_select_widget.ts";

export class MetroMapServer extends ChatKitServer<Record<string, unknown>> {
  metroMapStore: MetroMapStore;
  private threadItemConverter: MetroMapThreadItemConverter;
  private titleAgentInstance = titleAgent;

  constructor() {
    const store = new MemoryStore();
    super(store);

    const dataDir = path.resolve(Deno.cwd(), "apps/metro-map/backend-ts/data");
    this.metroMapStore = new MetroMapStore(dataDir);
    this.threadItemConverter = new MetroMapThreadItemConverter(this.metroMapStore);
  }

  async *respond(
    thread: ThreadMetadata,
    _item: UserMessageItem | null,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    const updatingThreadTitle = this.maybeUpdateThreadTitle(thread, _item, context);

    const itemsPage = await this.store.loadThreadItems(thread.id, null, 20, "desc", context);
    const items = itemsPage.data.reverse();
    const inputItems = await this.threadItemConverter.toAgentInput(items);

    const agentContext: MetroAgentContext = new AgentContext({
      thread,
      store: this.store,
      requestContext: context,
    }) as MetroAgentContext;
    agentContext.metro = this.metroMapStore;

    const result = await run(metroMapAgent, inputItems, {
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
    if (action.type === "line.select") {
      if (action.payload === null) return;
      for await (const event of this.handleLineSelectAction(thread, action.payload as Record<string, unknown>, sender, context)) {
        yield event;
      }
      return;
    }
  }

  private async *handleLineSelectAction(
    thread: ThreadMetadata,
    payload: Record<string, unknown>,
    sender: WidgetItem | null,
    context: Record<string, unknown>
  ): AsyncIterator<ThreadStreamEvent> {
    const lineId = payload.id as string;

    // Update widget to show selected line
    const updatedWidget = buildLineSelectWidget(this.metroMapStore.listLines(), lineId);

    if (sender) {
      const updatedWidgetItem = { ...sender, widget: updatedWidget };
      yield {
        type: "thread.item.replaced",
        item: updatedWidgetItem,
      };
    }

    // Add hidden context
    const hidden: HiddenContextItem = {
      type: "hidden_context",
      id: this.store.generateItemId("message", thread, context),
      thread_id: thread.id,
      created_at: new Date(),
      content: `<LINE_SELECTED>${lineId}</LINE_SELECTED>`,
    };
    await this.store.addThreadItem(thread.id, hidden, context);

    yield {
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: thread.id,
        id: this.store.generateItemId("message", thread, context),
        created_at: new Date(),
        content: [{
          type: "output_text",
          text: "Would you like to add the station to the beginning or end of the line?",
        }],
      },
    } as ThreadItemDoneEvent;

    yield {
      type: "client.effect",
      name: "location_select_mode",
      data: { lineId },
    };
  }

  private async maybeUpdateThreadTitle(
    thread: ThreadMetadata,
    userMessage: UserMessageItem | null,
    context: Record<string, unknown>
  ): Promise<void> {
    if (userMessage === null || thread.title !== null) return;

    const runResult = await run(
      this.titleAgentInstance,
      await this.threadItemConverter.toAgentInput(userMessage)
    );
    let modelResult: string = runResult.finalOutput;
    modelResult = modelResult.charAt(0).toUpperCase() + modelResult.slice(1);
    thread.title = modelResult.replace(/\.$/, "");
    await this.store.saveThread(thread, context);
  }
}

