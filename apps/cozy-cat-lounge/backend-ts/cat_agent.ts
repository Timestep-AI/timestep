/**
 * Cat agent implementation with tools for the Cozy Cat Lounge.
 */

import { Agent, tool } from "@openai/agents";
import type { AgentContext, HiddenContextItem, ThreadItemDoneEvent, ClientEffectEvent } from "chatkit-server";
import { z } from "zod";

import { CatState } from "./cat_state.ts";
import { CatStore } from "./cat_store.ts";
import { buildNameSuggestionsWidget, type CatNameSuggestion } from "./widgets/name_suggestions_widget.ts";
import { buildProfileCardWidget, profileWidgetCopyText } from "./widgets/profile_card_widget.ts";
import type { MemoryStore } from "./memory_store.ts";

const INSTRUCTIONS = `
    You are Cozy Cat Companion, a playful caretaker helping the user look after a virtual cat.
    Keep interactions light, imaginative, and focused on the cat's wellbeing. Provide concise
    status updates and narrate what happens after each action.

    Let the user know that the cat's color pattern will stay a mystery until they officially name it.

    Always keep the per-thread cat stats (energy, happiness, cleanliness, name, age)
    in sync with the tools provided. When you need the latest numbers, call \`get_cat_status\` before making a plan.

    Tools:
    - When the user asks you to feed, play with, or clean the cat, immediately call the respective tool
      (\`feed_cat\`, \`play_with_cat\`, or \`clean_cat\`). Describe the outcome afterwards using the updated stats.
      - When feeding, mention specific snacks or cat treats that was used to feed that cat if the user did not specify any food.
      - When playing, mention specific toys or objects that cats usually like that was used to play with the cat if the user did not specify any item.
      - When cleaning, mention specific items or methods that were used to clean the cat if the user did not specify any method.
      - If the user asks to "freshen up" the cat, call the \`clean_cat\` tool.
      - Once an action has been performed, it will be reflected as a <FED_CAT>, <PLAYED_WITH_CAT>, or <CLEANED_CAT> tag in the thread content.
      - Do not fire off multiple tool calls for the same action unless the user explicitly asks for it.
      - When the user interacts with an unnamed cat, prompt the user to name the cat.
    - When you call \`suggest_cat_names\`, pass a \`suggestions\` array containing at least three creative options.
      Each suggestion must include \`name\` (short and unique) and \`reason\` (why it fits the cat's current personality or stats).
      Prefer single word names, but if the suggested name is multiple words, use a space to separate them. For example: "Mr. Whiskers" or "Fluffy Paws".
      The user's choice will be reflected as a <CAT_NAME_SELECTED> tag in the thread content. Use that name in all future
      responses.
    - When the user explicitly asks for a profile card, call \`show_cat_profile\` with the age of the cat (for example: 1, 2, 3, etc.) and the name of a favorite toy (for example: "Ball of yarn", "Stuffed mouse", be creative but keep it two words or less!)
    - When the user's message is addressed directly to the cat, call \`speak_as_cat\` with the desired line so the dashboard bubbles it.
      When speaking as the cat, use "meow" or "purr" with a parenthesis at the end to translate it into English. For example: meow (I'm low on energy)
    - Never call \`set_cat_name\` if the cat already has a name that is not "Unnamed Cat".
    - If the cat currently does not have a name and the user explicitly names the cat, call \`set_cat_name\` with the exact name.
      Use that name in all future responses.

    Stay in character: talk about caring for the cat, suggest next steps if the stats look unbalanced, and avoid unrelated topics.

    Notes:
    - If the user has not yet named the cat, ask if they'd like to name it.
    - The cat's color pattern is only revealed once it has been named; encourage the user to name the cat to discover it.
    - Once a cat is named, it cannot be renamed. Do not invoke the \`set_cat_name\` tool if the cat has already been named.
    - If a user addresses an unnamed cat by a name for the first time, ask the user whether they'd like to name the cat.
    - If a user indicates they want to name the cat but does not provide a name, call the \`suggest_cat_names\` tool to give some options.
    - After naming the cat, let the user know that the cat's profile card has been issued and ask them whether they'd like to see it.
`;

const MODEL = "gpt-4.1-mini";

// Extended context type for the cat agent
export interface CatAgentContext extends AgentContext {
  cats: CatStore;
}

// Helper to extract context from SDK wrapper
// The @openai/agents SDK wraps context in a RunContextWrapper
function getContext(ctx: unknown): CatAgentContext {
  const wrapper = ctx as { context: CatAgentContext };
  return wrapper.context || ctx as CatAgentContext;
}

// Helper to get cat state
async function getState(ctx: CatAgentContext): Promise<CatState> {
  const threadId = ctx.thread.id;
  return await ctx.cats.load(threadId);
}

// Helper to update cat state
async function updateState(
  ctx: CatAgentContext,
  mutator: (state: CatState) => void
): Promise<CatState> {
  const threadId = ctx.thread.id;
  return await ctx.cats.mutate(threadId, mutator);
}

// Helper to sync status to client
async function syncStatus(
  ctx: CatAgentContext,
  state: CatState,
  flash?: string | null
): Promise<void> {
  await ctx.stream({
    type: "client_effect",
    name: "update_cat_status",
    data: {
      state: state.toPayload(ctx.thread.id),
      flash: flash ?? null,
    },
  } as ClientEffectEvent);
}

// Helper to add hidden context
async function addHiddenContext(ctx: CatAgentContext, content: string): Promise<void> {
  const thread = ctx.thread;
  const requestContext = ctx.requestContext;
  const store = ctx.store as MemoryStore;
  
  await store.addThreadItem(
    thread.id,
    {
      type: "hidden_context",
      id: store.generateItemId("message", thread, requestContext),
      thread_id: thread.id,
      createdAt: new Date(),
      content: content,
    } as HiddenContextItem,
    requestContext
  );
}

// Tool: Get cat status
const getCatStatusTool = tool({
  name: "get_cat_status",
  description: "Read the cat's current stats before deciding what to do next. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const catCtx = getContext(ctx);
    const state = await getState(catCtx);
    // Must return payload so that the assistant can use it to generate a natural language response.
    return state.toPayload(catCtx.thread.id);
  },
});

// Tool: Feed cat
const feedCatTool = tool({
  name: "feed_cat",
  description: "Feed the cat to replenish energy and keep moods stable.\n- `meal`: Meal or snack description to include in the update (can be null).",
  parameters: z.object({
    meal: z.string().nullable().describe("Meal or snack description, or null if not specified"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);
    const state = await updateState(catCtx, (s) => s.feed());
    const flash = args.meal ? `Fed ${state.name} ${args.meal}` : `${state.name} enjoyed a snack`;
    await addHiddenContext(catCtx, `<FED_CAT>${flash}</FED_CAT>`);
    await syncStatus(catCtx, state, flash);
    return { success: true };
  },
});

// Tool: Play with cat
const playWithCatTool = tool({
  name: "play_with_cat",
  description: "Play with the cat to boost happiness and create fun moments.\n- `activity`: Toy or activity used during playtime (can be null).",
  parameters: z.object({
    activity: z.string().nullable().describe("Toy or activity used during playtime, or null if not specified"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);
    const state = await updateState(catCtx, (s) => s.play());
    const flash = args.activity || "Playtime";
    await addHiddenContext(catCtx, `<PLAYED_WITH_CAT>${flash}</PLAYED_WITH_CAT>`);
    await syncStatus(catCtx, state, `${state.name} played: ${flash}`);
    return { success: true };
  },
});

// Tool: Clean cat
const cleanCatTool = tool({
  name: "clean_cat",
  description: "Clean the cat to tidy up and improve cleanliness.\n- `method`: Cleaning method or item used (can be null).",
  parameters: z.object({
    method: z.string().nullable().describe("Cleaning method or item used, or null if not specified"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);
    const state = await updateState(catCtx, (s) => s.clean());
    const flash = args.method || "Bath time";
    await addHiddenContext(catCtx, `<CLEANED_CAT>${flash}</CLEANED_CAT>`);
    await syncStatus(catCtx, state, `${state.name} is fresh: ${flash}`);
    return { success: true };
  },
});

// Tool: Set cat name
const setCatNameTool = tool({
  name: "set_cat_name",
  description: "Give the cat a permanent name and update the thread title to match.\n- `name`: Desired name for the cat.",
  parameters: z.object({
    name: z.string().describe("Desired name for the cat"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);
    const store = catCtx.store as MemoryStore;

    let state = await getState(catCtx);
    if (state.name !== "Unnamed Cat") {
      await catCtx.stream({
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: catCtx.thread.id,
          id: catCtx.generateId("message"),
          createdAt: new Date(),
          content: [{ type: "output_text", text: `${state.name} is ready to play!` }],
        },
      } as ThreadItemDoneEvent);
      return { success: true, message: "Cat already has a name" };
    }

    const cleaned = args.name.trim().split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(" ");
    if (!cleaned) {
      throw new Error("A name is required to rename the cat.");
    }

    state = await updateState(catCtx, (s) => s.rename(cleaned));
    catCtx.thread.title = `${state.name}'s Lounge`;
    await store.saveThread(catCtx.thread, catCtx.requestContext);

    await addHiddenContext(catCtx, `<CAT_NAME_SELECTED>${state.name}</CAT_NAME_SELECTED>`);
    await syncStatus(catCtx, state, `Now called ${state.name}`);
    return { success: true };
  },
});

// Tool: Show cat profile
const showCatProfileTool = tool({
  name: "show_cat_profile",
  description: "Show the cat's profile card with avatar and age.\n- `age`: Cat age (in years) to display and persist (can be null).\n- `favorite_toy`: Favorite toy label to include (can be null).",
  parameters: z.object({
    age: z.number().nullable().describe("Cat age in years, or null if not specified"),
    favorite_toy: z.string().nullable().describe("Favorite toy label, or null if not specified"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);

    const state = await updateState(catCtx, (s) => {
      if (args.age) s.setAge(args.age);
    });
    const widget = buildProfileCardWidget(state, args.favorite_toy);
    await catCtx.streamWidget(widget, profileWidgetCopyText(state));

    if (state.name === "Unnamed Cat") {
      await catCtx.stream({
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: catCtx.thread.id,
          id: catCtx.generateId("message"),
          createdAt: new Date(),
          content: [{ type: "output_text", text: "Would you like to give your cat a name?" }],
        },
      } as ThreadItemDoneEvent);
    } else {
      await catCtx.stream({
        type: "thread.item.done",
        item: {
          type: "assistant_message",
          thread_id: catCtx.thread.id,
          id: catCtx.generateId("message"),
          createdAt: new Date(),
          content: [{ type: "output_text", text: `License checked! Would you like to feed, play with, or clean ${state.name}?` }],
        },
      } as ThreadItemDoneEvent);
    }
    return { success: true };
  },
});

// Tool: Speak as cat
const speakAsCatTool = tool({
  name: "speak_as_cat",
  description: "Speak as the cat so a bubble appears in the dashboard.\n- `line`: The text the cat should say.",
  parameters: z.object({
    line: z.string().describe("The text the cat should say"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);
    const message = args.line.trim();
    if (!message) {
      throw new Error("A line is required for the cat to speak.");
    }
    const state = await getState(catCtx);
    await catCtx.stream({
      type: "client_effect",
      name: "cat_say",
      data: {
        state: state.toPayload(catCtx.thread.id),
        message: message,
      },
    } as ClientEffectEvent);
    return { success: true };
  },
});

// Tool: Suggest cat names
const suggestCatNamesTool = tool({
  name: "suggest_cat_names",
  description: "Render up to three creative cat name options provided in the `suggestions` argument.\n- `suggestions`: List of name suggestions with a `name` and `reason` for each.",
  parameters: z.object({
    suggestions: z.array(z.object({
      name: z.string().describe("The suggested name"),
      reason: z.string().nullable().describe("Why this name fits the cat, or null if not specified"),
    })).describe("List of name suggestions"),
  }),
  execute: async (args, ctx) => {
    const catCtx = getContext(ctx);

    const normalized: CatNameSuggestion[] = args.suggestions.map((s) => ({
      name: s.name,
      reason: s.reason,
    }));

    if (normalized.length === 0) {
      throw new Error("Provide at least one valid name suggestion before calling the tool.");
    }

    await catCtx.stream({
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: catCtx.thread.id,
        id: catCtx.generateId("message"),
        createdAt: new Date(),
        content: [{ type: "output_text", text: "Here are some name suggestions for your cat." }],
      },
    } as ThreadItemDoneEvent);

    const widget = buildNameSuggestionsWidget(normalized);
    await catCtx.streamWidget(widget, normalized.map((s) => s.name).join(", "));
    return { success: true };
  },
});

// Create the cat agent
export const catAgent = new Agent({
  model: MODEL,
  name: "Cozy Cat Companion",
  instructions: INSTRUCTIONS,
  tools: [
    // Fetches data used by the agent to run inference.
    getCatStatusTool,
    // Produces a simple widget output.
    showCatProfileTool,
    // Invokes a client effect to make the cat speak.
    speakAsCatTool,
    // Mutates state then invokes a client effect to sync client state.
    feedCatTool,
    playWithCatTool,
    cleanCatTool,
    // Mutates both cat state and thread state then invokes a client effect
    // to sync client state.
    setCatNameTool,
    // Outputs interactive widget output with partially agent-generated content.
    suggestCatNamesTool,
  ],
  // Stop inference after tool calls with widget outputs to prevent repetition.
  toolUseBehavior: {
    stopAtToolNames: [
      "suggest_cat_names",
      "show_cat_profile",
    ],
  },
});

