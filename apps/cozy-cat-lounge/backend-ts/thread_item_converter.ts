/**
 * Helpers that convert ChatKit thread items into model-friendly inputs.
 */

import { ThreadItemConverter } from "chatkit-server";
import type { HiddenContextItem } from "chatkit-server";

/**
 * Adds HiddenContextItem support for the boilerplate demo.
 */
export class BasicThreadItemConverter extends ThreadItemConverter {
  async hiddenContextToInput(item: HiddenContextItem): Promise<unknown> {
    return {
      type: "message",
      content: [
        {
          type: "input_text",
          text: item.content as string,
        },
      ],
      role: "user",
    };
  }
}

