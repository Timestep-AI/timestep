#!/usr/bin/env tsx
/** Example usage of Timestep eval framework. */

import { writeFileSync } from 'fs';
import { runSuite, report, agentBuiltinEcho, DEFAULT_TOOLS } from '../timestep/eval/index.js';
import { FinalContains, ForbiddenTools, FinalRegex } from '../timestep/eval/graders.js';

function createExampleTasks(): string {
  /** Create example tasks for evaluation. */
  const tasks = [
    {
      id: 'hello_01',
      messages: [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Say hello to Mike in one sentence.' }
      ],
      expected: { final_contains: 'Mike' },
      limits: { max_steps: 5, time_limit_s: 30 }
    },
    {
      id: 'calc_01',
      messages: [
        { role: 'system', content: 'You must use the calc tool.' },
        { role: 'user', content: 'Compute 19*7 using the calc tool, then answer with only the number.' }
      ],
      tools_allowed: ['calc'],
      expected: { final_regex: '^133$' },
      limits: { max_steps: 10, time_limit_s: 30 }
    }
  ];
  
  // Write to JSONL
  const tasksPath = 'tasks.jsonl';
  writeFileSync(tasksPath, tasks.map(t => JSON.stringify(t)).join('\n') + '\n');
  
  return tasksPath;
}

function main(): void {
  /** Run example evaluation. */
  // Create tasks
  const tasksPath = createExampleTasks();
  
  // Run eval suite
  runSuite(
    tasksPath,
    'runs/example',
    agentBuiltinEcho,
    DEFAULT_TOOLS,
    [new FinalContains(), new ForbiddenTools(), new FinalRegex()],
    3,
    0,
    120
  );
  
  // Generate report
  console.log('\n' + '='.repeat(60));
  console.log('Evaluation Report');
  console.log('='.repeat(60));
  report('runs/example');
}

main();
