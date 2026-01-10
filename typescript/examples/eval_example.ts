/** Example usage of Timestep evaluation harness with various graders. */

import { runSuite, report, agentBuiltinEcho, DEFAULT_TOOLS } from '@timestep-ai/timestep';
import {
  FinalContains,
  ForbiddenTools,
  FinalRegex,
  TranscriptContains,
  MinToolCalls,
  ToolCallSequence,
  // LLMJudge,  // Uncomment if you have OpenAI API key
} from '@timestep-ai/timestep';
import { writeFileSync } from 'fs';

function createExampleTasks(): string {
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
      expected: {
        final_regex: '^133$',
        must_call_tool: 'calc'
      },
      limits: { max_steps: 10, time_limit_s: 30, min_tool_calls: 1 }
    },
    {
      id: 'transcript_01',
      messages: [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'First say "Hello", then say "World".' }
      ],
      expected: {
        transcript_contains: 'Hello',
      },
      limits: { max_steps: 5, time_limit_s: 30 }
    }
  ];
  
  // Write to JSONL
  const tasksPath = 'tasks.jsonl';
  writeFileSync(tasksPath, tasks.map(t => JSON.stringify(t)).join('\n') + '\n');
  
  return tasksPath;
}

async function main() {
  // Create tasks
  const tasksPath = createExampleTasks();
  
  // Define graders
  const graders = [
    new FinalContains(),  // Code-based: checks final message contains substring
    new ForbiddenTools(),  // Code-based: checks tool usage
    new FinalRegex(),  // Code-based: regex on final message
    new TranscriptContains(),  // Code-based: checks any message in transcript
    new MinToolCalls(),  // Code-based: ensures minimum tool calls
    new ToolCallSequence(),  // Code-based: checks tool was called
    // new LLMJudge(  // LLM-as-judge: uses OpenAI to grade
    //   undefined,
    //   'gpt-4o-mini',
    //   0.0,
    //   false
    // ),
  ];
  
  // Run eval suite
  await runSuite(
    tasksPath,
    'runs/example',
    agentBuiltinEcho,
    DEFAULT_TOOLS,
    graders,
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

main().catch(console.error);
