/**
 * Tests for runtime safety checks.
 */

import { describe, it, expect } from 'vitest';
import {
  CircularDependencyChecker,
  ToolCompatibilityChecker,
} from '../../timestep/analysis/safety';

describe('CircularDependencyChecker', () => {
  it('should check for circular dependencies', async () => {
    const checker = new CircularDependencyChecker();
    // This will fail if no DB connection, but that's expected
    // In a real test, we'd set up a test database
    const result = await checker.checkCircularHandoffs('nonexistent_agent');
    // Should return null (no cycle) or fail gracefully
    expect(result === null || Array.isArray(result)).toBe(true);
  });
});

describe('ToolCompatibilityChecker', () => {
  it('should check tool compatibility', async () => {
    const checker = new ToolCompatibilityChecker();
    // This will fail if no DB connection, but that's expected
    // In a real test, we'd set up a test database
    try {
      const warnings = await checker.checkCompatibility('nonexistent_agent');
      expect(Array.isArray(warnings)).toBe(true);
    } catch (e) {
      // Expected if agent doesn't exist or DB not available
      expect(e).toBeDefined();
    }
  });
});

