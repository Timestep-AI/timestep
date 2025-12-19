/**
 * Custom thread item converter for customer support.
 */

import { ThreadItemConverter } from "chatkit-server";
import type { HiddenContextItem } from "chatkit-server";

export class CustomerSupportThreadItemConverter extends ThreadItemConverter {
  async hiddenContextToInput(item: HiddenContextItem): Promise<unknown> {
    return {
      type: "message",
      role: "user",
      content: [
        {
          type: "input_text",
          text: typeof item.content === "string" ? item.content : JSON.stringify(item.content),
        },
      ],
    };
  }
}

