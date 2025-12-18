/**
 * This is an example of a store for non-chatkit application data.
 * Thread-safe in-memory store for cat state keyed by thread id.
 */

import { CatState } from "./cat_state.ts";

export class CatStore {
  private _states: Map<string, CatState> = new Map();
  private _lockPromise: Promise<void> = Promise.resolve();

  private async _withLock<T>(fn: () => T | Promise<T>): Promise<T> {
    // Simple mutex pattern
    let resolve: () => void;
    const currentLock = this._lockPromise;
    this._lockPromise = new Promise((r) => {
      resolve = r;
    });

    await currentLock;
    try {
      return await fn();
    } finally {
      resolve!();
    }
  }

  private _ensure(threadId: string): CatState {
    let state = this._states.get(threadId);
    if (state === undefined) {
      state = new CatState();
      this._states.set(threadId, state);
    }
    return state;
  }

  async load(threadId: string): Promise<CatState> {
    return this._withLock(() => {
      return this._ensure(threadId).clone();
    });
  }

  async mutate(
    threadId: string,
    mutator: (state: CatState) => void
  ): Promise<CatState> {
    return this._withLock(() => {
      const state = this._ensure(threadId);
      mutator(state);
      return state.clone();
    });
  }
}

