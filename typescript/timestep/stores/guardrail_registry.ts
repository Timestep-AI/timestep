/** Guardrail registry for mapping guardrail identifiers to Guardrail objects. */

import type { InputGuardrail, OutputGuardrail } from '@openai/agents';

// Global registries mapping guardrail identifiers to Guardrail objects
const inputGuardrailRegistry = new Map<string, InputGuardrail>();
const outputGuardrailRegistry = new Map<string, OutputGuardrail>();

/**
 * Register an input guardrail in the registry.
 *
 * @param identifier - Unique identifier for the guardrail (typically guardrail.name)
 * @param guardrail - The InputGuardrail object to register
 */
export function registerInputGuardrail(identifier: string, guardrail: InputGuardrail): void {
  inputGuardrailRegistry.set(identifier, guardrail);
}

/**
 * Register an output guardrail in the registry.
 *
 * @param identifier - Unique identifier for the guardrail (typically guardrail.name)
 * @param guardrail - The OutputGuardrail object to register
 */
export function registerOutputGuardrail(identifier: string, guardrail: OutputGuardrail): void {
  outputGuardrailRegistry.set(identifier, guardrail);
}

/**
 * Get an input guardrail from the registry by identifier.
 *
 * @param identifier - The guardrail identifier
 * @returns The InputGuardrail object if found, undefined otherwise
 */
export function getInputGuardrail(identifier: string): InputGuardrail | undefined {
  return inputGuardrailRegistry.get(identifier);
}

/**
 * Get an output guardrail from the registry by identifier.
 *
 * @param identifier - The guardrail identifier
 * @returns The OutputGuardrail object if found, undefined otherwise
 */
export function getOutputGuardrail(identifier: string): OutputGuardrail | undefined {
  return outputGuardrailRegistry.get(identifier);
}

/**
 * Clear the guardrail registries (useful for testing).
 */
export function clearRegistry(): void {
  inputGuardrailRegistry.clear();
  outputGuardrailRegistry.clear();
}


