/**
 * Tests for olog builder.
 */

import { describe, it, expect } from 'vitest';
import {
  Olog,
  OlogType,
  OlogAspect,
  OlogRelationType,
  OlogValidator,
} from '../../timestep/analysis/olog';

describe('Olog', () => {
  it('should add types to olog', () => {
    const olog = new Olog();
    const ologType: OlogType = {
      name: 'Agent',
      description: 'An agent',
      examples: ['Agent1', 'Agent2'],
    };
    olog.addType(ologType);

    const ologAny = olog as any;
    const types = ologAny.types as Map<string, OlogType>;
    expect(types.has('Agent')).toBe(true);
    expect(types.get('Agent')?.name).toBe('Agent');
  });

  it('should add aspects to olog', () => {
    const olog = new Olog();
    olog.addType({ name: 'Agent', description: 'An agent' });
    olog.addType({ name: 'Tool', description: 'A tool' });

    const aspect: OlogAspect = {
      source: 'Agent',
      target: 'Tool',
      relation: OlogRelationType.USES,
      description: 'Agent uses tool',
    };
    olog.addAspect(aspect);

    const ologAny = olog as any;
    const aspects = ologAny.aspects as OlogAspect[];
    expect(aspects.length).toBe(1);
    expect(aspects[0].source).toBe('Agent');
    expect(aspects[0].target).toBe('Tool');
  });

  it('should convert olog to markdown', () => {
    const olog = new Olog();
    olog.addType({
      name: 'Agent',
      description: 'An agent',
      examples: ['Agent1'],
    });
    olog.addType({ name: 'Tool', description: 'A tool' });
    olog.addAspect({
      source: 'Agent',
      target: 'Tool',
      relation: OlogRelationType.USES,
    });

    const markdown = olog.toMarkdown();
    expect(markdown).toContain('# Agent System Ontology');
    expect(markdown).toContain('Agent');
    expect(markdown).toContain('Tool');
  });

  it('should convert olog to mermaid', () => {
    const olog = new Olog();
    olog.addType({ name: 'Agent', description: 'An agent' });
    olog.addType({ name: 'Tool', description: 'A tool' });
    olog.addAspect({
      source: 'Agent',
      target: 'Tool',
      relation: OlogRelationType.USES,
    });

    const mermaid = olog.toMermaid();
    expect(mermaid).toContain('graph TD');
    expect(mermaid).toContain('Agent');
    expect(mermaid).toContain('Tool');
  });
});

describe('OlogValidator', () => {
  it('should validate olog', () => {
    const olog = new Olog();
    olog.addType({ name: 'Agent', description: 'An agent' });
    olog.addAspect({
      source: 'Agent',
      target: 'Tool', // Tool type doesn't exist
      relation: OlogRelationType.USES,
    });

    const validator = new OlogValidator();
    const issues = validator.validate(olog);

    // Should find that Tool type is missing
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((issue) => issue.includes('Tool'))).toBe(true);
  });
});

