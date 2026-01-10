/** Suite runner for evaluation harness. */

import { readJsonl, writeJsonl } from '../utils/jsonl.js';
import { writeJson, now } from '../utils/io.js';
import { ensureTaskId } from '../utils/messages.js';
import type { AgentFn, ToolFn, EpisodeInfo } from '../core/index.js';
import { runEpisode, indexToolCalls } from '../core/index.js';
import type { Grader } from './graders.js';
import { aggregateGrades } from './graders.js';
import { mkdirSync } from 'fs';
import { join } from 'path';

export type JSON = Record<string, any>;

export async function runSuite(
  tasksPath: string,
  outdir: string,
  agent: AgentFn,
  tools: Record<string, ToolFn>,
  graders: Grader[],
  trials: number,
  seed: number,
  agentTimeoutS: number,
): Promise<void> {
  /** Run evaluation suite on tasks from JSONL file. */
  mkdirSync(outdir, { recursive: true });
  
  // Simple seeded RNG
  let rngState = seed;
  function random(): number {
    rngState = (rngState * 1103515245 + 12345) & 0x7fffffff;
    return rngState / 0x7fffffff;
  }
  function randrange(min: number, max: number): number {
    return Math.floor(random() * (max - min)) + min;
  }

  const runMeta = {
    version: 'eval_mvp_v1',
    tasks_path: tasksPath,
    trials,
    seed,
    started_at: now(),
    graders: graders.map(g => g.name),
    agent_timeout_s: agentTimeoutS,
    tools_available: Object.keys(tools).sort(),
  };
  writeJson(join(outdir, 'run_meta.json'), runMeta);

  const resultsRows: JSON[] = [];

  for (const task of readJsonl(tasksPath)) {
    const taskId = ensureTaskId(task);
    const taskMessages = task.messages;
    if (!Array.isArray(taskMessages)) {
      throw new Error(`Task ${taskId} missing 'messages' list.`);
    }

    const toolsAllowed = task.tools_allowed; // optional allowlist
    const limits = task.limits || {};

    for (let trial = 1; trial <= trials; trial++) {
      const trialSeed = randrange(0, 2**31 - 1);

      const trialDir = join(outdir, 'trials', taskId, `trial_${String(trial).padStart(2, '0')}`);
      mkdirSync(trialDir, { recursive: true });

      // Attach trial metadata (kept out of messages)
      const taskMeta = { ...task, _trial: trial };

      // Run episode (core agent-environment loop)
      const [messages, info] = await runEpisode(
        taskMessages,
        agent,
        tools,
        toolsAllowed,
        limits,
        taskMeta,
        trialSeed,
      );

      // Build tool index
      const toolIdx = indexToolCalls(messages);

      // Grade (handle async graders)
      const gradeRows = await Promise.all(
        graders.map(async g => {
          const result = g.grade(messages, toolIdx, task, info);
          // Handle both sync and async graders
          return result instanceof Promise ? await result : result;
        })
      );
      const agg = aggregateGrades(gradeRows);

      // Persist artifacts
      writeJson(join(trialDir, 'transcript.json'), messages);
      writeJson(join(trialDir, 'tool_index.json'), toolIdx.map(r => ({
        tool_call_id: r.tool_call_id,
        name: r.name,
        arguments_raw: r.arguments_raw,
        arguments: r.arguments,
        result_raw: r.result_raw,
        result: r.result,
        error: r.error,
      })));
      writeJson(join(trialDir, 'grades.json'), { grades: gradeRows, aggregate: agg });
      writeJson(join(trialDir, 'info.json'), {
        task_id: info.task_id,
        trial: info.trial,
        seed: info.seed,
        steps: info.steps,
        tool_calls: info.tool_calls,
        duration_s: info.duration_s,
        terminated_reason: info.terminated_reason,
        error: info.error,
        input_tokens: info.input_tokens,
        output_tokens: info.output_tokens,
        total_tokens: info.total_tokens,
        cost_usd: info.cost_usd,
      });

      // Row for results.jsonl
      resultsRows.push({
        task_id: taskId,
        trial,
        seed: trialSeed,
        terminated_reason: info.terminated_reason,
        steps: info.steps,
        tool_calls: info.tool_calls,
        duration_s: info.duration_s,
        input_tokens: info.input_tokens,
        output_tokens: info.output_tokens,
        total_tokens: info.total_tokens,
        cost_usd: info.cost_usd,
        passed: agg.passed,
        score: agg.score,
      });
    }
  }

  writeJsonl(join(outdir, 'results.jsonl'), resultsRows);
  runMeta.ended_at = now();
  writeJson(join(outdir, 'run_meta.json'), runMeta);
}

export function report(outdir: string): void {
  /** Print summary report of evaluation results. */
  const resultsPath = join(outdir, 'results.jsonl');
  
  let rows: JSON[] = [];
  try {
    rows = Array.from(readJsonl(resultsPath));
  } catch (e: any) {
    throw new Error(`No results.jsonl in ${outdir}: ${e.message}`);
  }
  
  if (rows.length === 0) {
    console.log('No results.');
    return;
  }

  const overallPass = rows.filter(r => r.passed).length / rows.length;
  const overallScore = rows.reduce((sum, r) => sum + Number(r.score || 0), 0) / rows.length;
  const avgTokens = rows.reduce((sum, r) => sum + Number(r.total_tokens || 0), 0) / rows.length;

  const byTask: Record<string, JSON[]> = {};
  for (const r of rows) {
    const tid = String(r.task_id);
    if (!byTask[tid]) byTask[tid] = [];
    byTask[tid].push(r);
  }

  const taskSummaries: Array<[string, number, number, number, number, number]> = [];
  for (const [tid, rs] of Object.entries(byTask)) {
    const pr = rs.filter(x => x.passed).length / rs.length;
    const ms = rs.reduce((sum, x) => sum + Number(x.score || 0), 0) / rs.length;
    const md = rs.reduce((sum, x) => sum + Number(x.duration_s || 0), 0) / rs.length;
    const mt = rs.reduce((sum, x) => sum + Number(x.total_tokens || 0), 0) / rs.length;
    taskSummaries.push([tid, pr, ms, md, mt, rs.length]);
  }

  taskSummaries.sort((a, b) => a[1] - b[1] || a[2] - b[2]); // worst first

  console.log(`Run: ${outdir}`);
  console.log(`Trials: ${rows.length}`);
  console.log(`Overall pass rate: ${overallPass.toFixed(3)}`);
  console.log(`Overall mean score: ${overallScore.toFixed(3)}`);
  console.log(`Average tokens per trial: ${avgTokens.toFixed(0)}`);
  console.log();
  console.log('Worst tasks (task_id | pass_rate | mean_score | mean_duration_s | mean_tokens | trials):');
  for (const [tid, pr, ms, md, mt, n] of taskSummaries.slice(0, 20)) {
    console.log(`  ${tid} | ${pr.toFixed(3)} | ${ms.toFixed(3)} | ${md.toFixed(3)} | ${mt.toFixed(0)} | ${n}`);
  }
}
