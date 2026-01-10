/** CLI interface for eval framework. */

import { agentBuiltinEcho, agentCmdFactory } from './agent.js';
import { parseGraderSpec } from './graders.js';
import { runSuite, report } from './suite.js';
import { DEFAULT_TOOLS } from './tools.js';

export async function main(args: string[]): Promise<void> {
  /** Main CLI entry point. */
  if (args.length < 2) {
    console.error('Usage: timestep <run|report> [options]');
    process.exit(1);
  }

  const cmd = args[0];
  
  if (cmd === 'report') {
    const outdirIndex = args.indexOf('--outdir');
    if (outdirIndex === -1 || !args[outdirIndex + 1]) {
      console.error('--outdir required for report command');
      process.exit(1);
    }
    const outdir = args[outdirIndex + 1];
    report(outdir);
    return;
  }

  if (cmd === 'run') {
    // Parse arguments
    const tasksIndex = args.indexOf('--tasks');
    const outdirIndex = args.indexOf('--outdir');
    const agentIndex = args.indexOf('--agent');
    const trialsIndex = args.indexOf('--trials');
    const seedIndex = args.indexOf('--seed');
    const agentTimeoutIndex = args.indexOf('--agent-timeout-s');
    const gradersIndex = args.indexOf('--graders');

    if (tasksIndex === -1 || !args[tasksIndex + 1]) {
      console.error('--tasks required for run command');
      process.exit(1);
    }
    if (outdirIndex === -1 || !args[outdirIndex + 1]) {
      console.error('--outdir required for run command');
      process.exit(1);
    }
    if (agentIndex === -1 || !args[agentIndex + 1]) {
      console.error('--agent required for run command');
      process.exit(1);
    }

    const tasks = args[tasksIndex + 1];
    const outdir = args[outdirIndex + 1];
    const agentSpec = args[agentIndex + 1];
    const trials = trialsIndex !== -1 && args[trialsIndex + 1] ? Number(args[trialsIndex + 1]) : 3;
    const seed = seedIndex !== -1 && args[seedIndex + 1] ? Number(args[seedIndex + 1]) : 0;
    const agentTimeoutS = agentTimeoutIndex !== -1 && args[agentTimeoutIndex + 1] ? Number(args[agentTimeoutIndex + 1]) : 120;
    
    let graders = [
      'ForbiddenTools',
      'MaxToolCalls:50',
      'FinalRegex',
      'FinalContains',
    ];
    if (gradersIndex !== -1) {
      const graderArgs: string[] = [];
      for (let i = gradersIndex + 1; i < args.length && !args[i].startsWith('--'); i++) {
        graderArgs.push(args[i]);
      }
      if (graderArgs.length > 0) {
        graders = graderArgs;
      }
    }

    // Build agent
    let agent;
    if (agentSpec.startsWith('builtin:')) {
      const name = agentSpec.split(':')[1];
      if (name === 'echo') {
        agent = agentBuiltinEcho;
      } else {
        console.error(`Unknown builtin agent '${name}'. Available: echo`);
        process.exit(1);
      }
    } else if (agentSpec.startsWith('cmd:')) {
      const cmd = agentSpec.split(':').slice(1).join(':');
      agent = agentCmdFactory(cmd, agentTimeoutS);
    } else {
      console.error('Agent must be "builtin:echo" or "cmd:...".');
      process.exit(1);
    }

    // Build graders
    const graderInstances = graders.map(s => parseGraderSpec(s));

    // Tools (demo defaults)
    const tools = { ...DEFAULT_TOOLS };

    await runSuite(
      tasks,
      outdir,
      agent,
      tools,
      graderInstances,
      trials,
      seed,
      agentTimeoutS,
    );
    return;
  }

  console.error(`Unknown command: ${cmd}`);
  console.error('Usage: timestep <run|report> [options]');
  process.exit(1);
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2)).catch(console.error);
}
