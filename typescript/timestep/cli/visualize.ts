/**
 * CLI tool for visualizing agent systems.
 */

import { DiagramBuilder } from '../visualizations/stringDiagrams';
import { DiagramRenderer } from '../visualizations/renderer';
import { writeFile } from 'fs/promises';
import { parseArgs } from 'util';

async function main() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
      format: {
        type: 'string',
        default: 'mermaid',
      },
      output: {
        type: 'string',
      },
    },
    allowPositionals: true,
  });

  const agentId = positionals[0];
  if (!agentId) {
    console.error('Error: agent_id is required');
    process.exit(1);
  }

  const format = (values.format as string) || 'mermaid';
  const validFormats = ['mermaid', 'svg', 'dot', 'json'];
  if (!validFormats.includes(format)) {
    console.error(`Error: format must be one of: ${validFormats.join(', ')}`);
    process.exit(1);
  }

  const builder = DiagramBuilder;
  const diagram = await builder.fromAgent(agentId);

  const renderer = new DiagramRenderer();
  const output = renderer.render(diagram, format);

  if (values.output) {
    await writeFile(values.output as string, output);
  } else {
    console.log(output);
  }
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});

