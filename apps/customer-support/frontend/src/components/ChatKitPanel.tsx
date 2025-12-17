import { ChatKit, useChatKit } from "@openai/chatkit-react";
import type { ColorScheme } from "../hooks/useColorScheme";
import {
  SUPPORT_CHATKIT_API_DOMAIN_KEY,
  SUPPORT_CHATKIT_API_URL,
  SUPPORT_GREETING,
  SUPPORT_STARTER_PROMPTS,
} from "../lib/config";

type ChatKitPanelProps = {
  theme: ColorScheme;
  onThreadChange: (threadId: string | null) => void;
  onResponseCompleted: () => void;
};

export function ChatKitPanel({
  theme,
  onThreadChange,
  onResponseCompleted,
}: ChatKitPanelProps) {

  const chatkit = useChatKit({
    api: {
      url: SUPPORT_CHATKIT_API_URL,
      domainKey: SUPPORT_CHATKIT_API_DOMAIN_KEY,
    },
    theme: {
      colorScheme: theme,
      color: {
        grayscale: {
          hue: 220,
          tint: 6,
          shade: theme === "dark" ? -1 : -4,
        },
        accent: {
          primary: theme === "dark" ? "#f8fafc" : "#0f172a",
          level: 1,
        },
      },
      radius: "round",
    },
    startScreen: {
      greeting: SUPPORT_GREETING,
      prompts: SUPPORT_STARTER_PROMPTS,
    },
    composer: {
      placeholder: "Ask the concierge a question",
    },
    threadItemActions: {
      feedback: false,
    },
    onResponseEnd: () => {
      onResponseCompleted();
    },
    onThreadChange: ({ threadId }) => {
      onThreadChange(threadId ?? null);
    },
    onError: ({ error }) => {
      // ChatKit displays surfaced errors; we keep logging for debugging.
      console.error("ChatKit error", error);
    },
  });

  return (
    <div className="relative h-full w-full overflow-hidden border border-slate-200/60 bg-white shadow-card dark:border-slate-800/70 dark:bg-slate-900">
      <ChatKit control={chatkit.control} className="block h-full w-full" />
    </div>
  );
}
