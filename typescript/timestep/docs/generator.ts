/**
 * Documentation generator for agent systems.
 */

import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { OlogBuilder } from '../analysis/olog';
import { DiagramBuilder } from '../visualizations/stringDiagrams';
import { DiagramRenderer } from '../visualizations/renderer';

export class DocumentationGenerator {
  private outputDir: string;

  constructor(outputDir: string) {
    this.outputDir = outputDir;
  }

  async generateAll(agentIds: string[]): Promise<void> {
    // Ensure output directory exists
    await mkdir(this.outputDir, { recursive: true });

    // Generate olog documentation
    const olog = await OlogBuilder.fromAgentSystem(agentIds);
    await this.writeOlogDocs(olog);

    // Generate string diagram visualizations
    for (const agentId of agentIds) {
      await this.writeAgentDiagram(agentId);
    }
  }

  private async writeOlogDocs(olog: any): Promise<void> {
    // Markdown documentation
    const mdPath = join(this.outputDir, 'ontology.md');
    await writeFile(mdPath, olog.toMarkdown());

    // Mermaid diagram
    const mermaidPath = join(this.outputDir, 'ontology.mmd');
    await writeFile(mermaidPath, olog.toMermaid());
  }

  private async writeAgentDiagram(agentId: string): Promise<void> {
    const diagram = await DiagramBuilder.fromAgent(agentId);
    const renderer = new DiagramRenderer();

    // Write multiple formats
    const formats = ['mermaid', 'svg', 'json'];
    for (const fmt of formats) {
      const output = renderer.render(diagram, fmt);
      const ext = fmt === 'mermaid' ? 'mmd' : fmt;
      const path = join(this.outputDir, `agent_${agentId}.${ext}`);
      await writeFile(path, output);
    }
  }
}

