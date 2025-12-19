/**
 * News Guide agent for article discovery and recommendations.
 */

import { Agent, tool } from "@openai/agents";
import { z } from "zod";
import type { AgentContext, ThreadItemDoneEvent } from "chatkit-server";
import type { ArticleStore, ArticleMetadata } from "../data/article_store.ts";
import { buildArticleListWidget } from "../widgets/article_list_widget.ts";
import { eventFinderAgent } from "./event_finder_agent.ts";
import { puzzleAgent } from "./puzzle_agent.ts";

const INSTRUCTIONS = `
    You are News Guide, a service-forward assistant focused on helping readers quickly
    discover the most relevant news for their needs. Prioritize clarity, highlight how
    each story serves the reader, and keep answers concise with skimmable structure.

    Before recommending or summarizing, always consult the latest article metadata via
    the available tools.

    If the reader provides desired topics, locations, or tags, filter results before responding
    and call out any notable gaps.

    Unless the reader explicitly asks for a set number of articles, default to suggesting 2 articles.

    When the reader references "this article," "this story," or "this page," treat that as a request
    about the currently open article. Load it with \`get_current_page\`, review the content, and answer
    their question directly using specific details instead of asking them to copy anything over.

    When summarizing:
      - Cite the article title.
      - The summary should be 2-4 sentences long.
      - Do NOT explicitly mention the word "summary" in your response.
      - After summarizing, ask the reader if they have any questions about the article.

    Formatting output:
      - Default to italicizing article titles when you mention them, and wrap any direct excerpts from the article content in
        Markdown blockquotes so they stand out.
      - Add generous paragraph breaks for readability.

    Use the tools deliberately:
      - Call \`list_available_tags_and_keywords\` to get a list of all unique tags and keywords available to search by. Fuzzy match
        the reader's phrasing to these tags/keywords (case-insensitive, partial matches are ok) and pick the closest ones—instead
        of relying on any hard-coded synonym map—before running a search.
      - Use \`get_current_page\` to fetch the full article the reader currently has open whenever they need deeper details
        or ask questions about "this page".
      - Use \`search_articles_by_tags\` only when the reader explicitly references tags/sections (e.g., "show me everything tagged parks"); otherwise skip it.
      - Default to \`search_articles_by_keywords\` to match metadata (titles, subtitles, tags, keywords) to the reader's asks.
      - Use \`search_articles_by_exact_text\` when the reader quote a phrase or wants an exact content match.
      - After fetching story candidates, prefer \`show_article_list_widget\` with the returned articles fetched using
        \`search_articles_by_tags\` or \`search_articles_by_keywords\` or \`search_articles_by_exact_text\` and a message
        that explains why these articles were selected for the reader right now.
      - If the reader explicitly asks about events, happenings, or things to do, call \`delegate_to_event_finder\`
        with their request so the Foxhollow Event Finder agent can take over.
      - If the reader wants a Foxhollow-flavored puzzle, coffee-break brain teaser, or mentions the puzzle tool,
        call \`delegate_to_puzzle_keeper\` so the Foxhollow Puzzle Keeper can lead with Two Truths and the mini crossword.

    Custom tags:
     - When you see an <ARTICLE_REFERENCE>{article_id}</ARTICLE_REFERENCE> tag in the context, call \`get_article_by_id\`
       with that id before citing details so your answer can reference the tagged article accurately.
     - When you see an <AUTHOR_REFERENCE>{author}</AUTHOR_REFERENCE> tag in the context, or the reader names an author,
       call \`search_articles_by_author\` with that author before recommending pieces so you feature their work first.

    Suggest a next step—such as related articles or follow-up angles—whenever it adds value.
`.trim();

const MODEL = "gpt-4.1-mini";
const FEATURED_PAGE_ID = "featured";

export interface NewsAgentContext extends AgentContext {
  articles: ArticleStore;
  articleId?: string;
}

// Helper to extract context from SDK wrapper
function getContext(ctx: unknown): NewsAgentContext {
  const wrapper = ctx as { context: NewsAgentContext };
  return wrapper.context || (ctx as NewsAgentContext);
}

const ArticleMetadataSchema = z.object({
  id: z.string(),
  title: z.string(),
  heroImage: z.string(),
  heroImageUrl: z.string(),
  url: z.string(),
  filename: z.string(),
  date: z.string(),
  author: z.string(),
  tags: z.array(z.string()),
  keywords: z.array(z.string()),
});

// Tool: Search articles by tags
const searchArticlesByTagsTool = tool({
  name: "search_articles_by_tags",
  description: "List newsroom articles, optionally filtered by tags.\n- `tags`: One or more tags to filter by.",
  parameters: z.object({
    tags: z.array(z.string()).describe("Tags to filter by"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const tags = args.tags.filter((t) => t?.trim()).map((t) => t.trim().toLowerCase());
    if (tags.length === 0) throw new Error("Please provide at least one tag to search for.");
    const tagLabel = tags.join(", ");
    await agentCtx.stream({ type: "progress.update", text: `Searching for tags: ${tagLabel}` });
    const records = agentCtx.articles.listMetadataForTags(tags);
    return { articles: records };
  },
});

// Tool: Search articles by author
const searchArticlesByAuthorTool = tool({
  name: "search_articles_by_author",
  description: "Find articles written by a specific author.\n- `author`: Author name to search for (case-insensitive).",
  parameters: z.object({
    author: z.string().describe("Author name"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const author = args.author.trim();
    if (!author) throw new Error("Please provide an author name to search for.");
    const displayName = author.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
    await agentCtx.stream({ type: "progress.update", text: `Looking for articles by ${displayName}...` });
    const records = agentCtx.articles.searchMetadataByAuthor(author);
    return { author, articles: records };
  },
});

// Tool: List available tags and keywords
const listAvailableTagsAndKeywordsTool = tool({
  name: "list_available_tags_and_keywords",
  description: "List all unique tags and keywords available across the newsroom archive. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    await agentCtx.stream({ type: "progress.update", text: "Referencing available tags and keywords..." });
    return agentCtx.articles.listAvailableTagsAndKeywords();
  },
});

// Tool: Search articles by keywords
const searchArticlesByKeywordsTool = tool({
  name: "search_articles_by_keywords",
  description: "Search newsroom articles by keywords within their metadata (title, tags, keywords, etc.).\n- `keywords`: List of keywords to match against metadata.",
  parameters: z.object({
    keywords: z.array(z.string()).describe("Keywords to search for"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const cleaned = args.keywords.filter((k) => k?.trim()).map((k) => k.trim().toLowerCase());
    if (cleaned.length === 0) throw new Error("Please provide at least one non-empty keyword to search for.");
    const formatted = cleaned.join(", ");
    await agentCtx.stream({ type: "progress.update", text: `Searching for keywords: ${formatted}` });
    const records = agentCtx.articles.searchMetadataByKeywords(cleaned);
    return { articles: records };
  },
});

// Tool: Search articles by exact text
const searchArticlesByExactTextTool = tool({
  name: "search_articles_by_exact_text",
  description: "Search newsroom articles for an exact text match within their content.\n- `text`: Exact string to find inside article bodies.",
  parameters: z.object({
    text: z.string().describe("Exact text to search for"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    const trimmed = args.text.trim();
    if (!trimmed) throw new Error("Please provide a non-empty text string to search for.");
    await agentCtx.stream({ type: "progress.update", text: `Scanning articles for: ${trimmed}` });
    const records = agentCtx.articles.searchContentByExactText(trimmed);
    return { articles: records };
  },
});

// Tool: Get article by ID
const getArticleByIdTool = tool({
  name: "get_article_by_id",
  description: "Fetch the markdown content for a specific article.\n- `article_id`: Identifier of the article to load.",
  parameters: z.object({
    article_id: z.string().describe("Article identifier"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    await agentCtx.stream({ type: "progress.update", text: "Loading article..." });
    const record = agentCtx.articles.getArticle(args.article_id);
    if (!record) throw new Error(`Article '${args.article_id}' does not exist.`);
    return { article: record };
  },
});

// Tool: Get current page
const getCurrentPageTool = tool({
  name: "get_current_page",
  description: "Load the full content for the page the reader currently has open. No parameters.",
  parameters: z.object({}),
  execute: async (_args, ctx) => {
    const agentCtx = getContext(ctx);
    const articleId = agentCtx.articleId;
    if (!articleId) throw new Error("No article id is available in the current request context.");

    if (articleId === FEATURED_PAGE_ID) {
      const metadata = agentCtx.articles.listMetadataForTags([FEATURED_PAGE_ID]);
      const articles = metadata.map((m) => agentCtx.articles.getArticle(m.id)).filter(Boolean);
      await agentCtx.stream({ type: "progress.update", text: "Page contents retrieved" });
      return { page: FEATURED_PAGE_ID, articles };
    }

    const record = agentCtx.articles.getArticle(articleId);
    if (!record) throw new Error(`Article '${articleId}' does not exist.`);
    await agentCtx.stream({ type: "progress.update", text: "Full article retrieved" });
    return { page: "article", articles: [record], article_id: articleId };
  },
});

// Tool: Show article list widget
const showArticleListWidgetTool = tool({
  name: "show_article_list_widget",
  description: "Show a Newsroom-style article list widget for a provided set of articles.\n- `articles`: Article metadata entries to display.\n- `message`: Introductory text explaining why these were selected.",
  parameters: z.object({
    articles: z.array(ArticleMetadataSchema).describe("Articles to display"),
    message: z.string().describe("Introductory message"),
  }),
  execute: async (args, ctx) => {
    const agentCtx = getContext(ctx);
    if (args.articles.length === 0) {
      throw new Error("Provide at least one article metadata entry before calling this tool.");
    }

    await agentCtx.stream({
      type: "thread.item.done",
      item: {
        type: "assistant_message",
        thread_id: agentCtx.thread.id,
        id: agentCtx.generateId("message"),
        created_at: new Date(),
        content: [{ type: "output_text", text: args.message }],
      },
    } as ThreadItemDoneEvent);

    const widget = buildArticleListWidget(args.articles as ArticleMetadata[]);
    const titles = args.articles.map((a) => a.title).join(", ");
    await agentCtx.streamWidget(widget, titles);

    return { success: true };
  },
});

export const newsAgent = new Agent({
  model: MODEL,
  name: "Foxhollow Dispatch News Guide",
  instructions: INSTRUCTIONS,
  tools: [
    listAvailableTagsAndKeywordsTool,
    getArticleByIdTool,
    getCurrentPageTool,
    searchArticlesByAuthorTool,
    searchArticlesByTagsTool,
    searchArticlesByKeywordsTool,
    searchArticlesByExactTextTool,
    showArticleListWidgetTool,
    eventFinderAgent.asTool({
      toolName: "delegate_to_event_finder",
      toolDescription: "Delegate event-specific requests to the Foxhollow Event Finder agent.",
    }),
    puzzleAgent.asTool({
      toolName: "delegate_to_puzzle_keeper",
      toolDescription: "Delegate coffee break puzzle requests to the Foxhollow Puzzle Keeper agent.",
    }),
  ],
  toolUseBehavior: {
    stopAtToolNames: [
      showArticleListWidgetTool.name,
      "delegate_to_event_finder",
      "delegate_to_puzzle_keeper",
    ],
  },
});

