/** Suite runner for evaluation harness. */

import { readJsonl, writeJsonl } from '../utils/jsonl';
import { writeJson, now } from '../utils/io';
import { ensureTaskId } from '../utils/messages';
import type { AgentFn, ToolFn, EpisodeInfo, JSON } from '../core/index';
import { runEpisode, indexToolCalls } from '../core/index';
import type { Grader } from './graders';
import { aggregateGrades } from './graders';
import { mkdirSync } from 'fs';
import { join } from 'path';

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

  const runMeta: JSON = {
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

  const taskSummaries: Array<[string, number, number, number, number, number, number, number]> = [];
  for (const [tid, rs] of Object.entries(byTask)) {
    const k = rs.length;
    const passedCount = rs.filter(x => x.passed).length;
    const pr = passedCount / k;
    const ms = rs.reduce((sum, x) => sum + Number(x.score || 0), 0) / k;
    const md = rs.reduce((sum, x) => sum + Number(x.duration_s || 0), 0) / k;
    const mt = rs.reduce((sum, x) => sum + Number(x.total_tokens || 0), 0) / k;
    const passAtK = passedCount > 0 ? 1.0 : 0.0;
    const passPowerK = passedCount === k ? 1.0 : 0.0;
    taskSummaries.push([tid, pr, ms, md, mt, k, passAtK, passPowerK]);
  }

  taskSummaries.sort((a, b) => a[1] - b[1] || a[2] - b[2]); // worst first

  // Overall pass@k and pass^k
  const totalTasks = Object.keys(byTask).length;
  const tasksWithAnyPass = Object.values(byTask).filter(rs => rs.some(r => r.passed)).length;
  const tasksWithAllPass = Object.values(byTask).filter(rs => rs.every(r => r.passed)).length;
  const overallPassAtK = totalTasks > 0 ? tasksWithAnyPass / totalTasks : 0.0;
  const overallPassPowerK = totalTasks > 0 ? tasksWithAllPass / totalTasks : 0.0;

  console.log(`Run: ${outdir}`);
  console.log(`Trials: ${rows.length}`);
  console.log(`Overall pass rate: ${overallPass.toFixed(3)}`);
  console.log(`Overall mean score: ${overallScore.toFixed(3)}`);
  console.log(`Overall pass@k: ${overallPassAtK.toFixed(3)}`);
  console.log(`Overall pass^k: ${overallPassPowerK.toFixed(3)}`);
  console.log(`Average tokens per trial: ${avgTokens.toFixed(0)}`);
  console.log();
  
  // Format table with aligned columns
  const header = `${'task_id'.padEnd(40)} ${'pass_rate'.padStart(10)} ${'mean_score'.padStart(11)} ${'pass@k'.padStart(8)} ${'pass^k'.padStart(8)} ${'duration_s'.padStart(11)} ${'tokens'.padStart(8)} ${'trials'.padStart(7)}`;
  console.log('Worst tasks:');
  console.log(`  ${header}`);
  console.log(`  ${'-'.repeat(40)} ${'-'.repeat(10)} ${'-'.repeat(11)} ${'-'.repeat(8)} ${'-'.repeat(8)} ${'-'.repeat(11)} ${'-'.repeat(8)} ${'-'.repeat(7)}`);
  for (const [tid, pr, ms, md, mt, k, passAtK, passPowerK] of taskSummaries.slice(0, 20)) {
    // Truncate task_id if too long
    const tidDisplay = tid.length > 40 ? tid.slice(0, 37) + '...' : tid;
    console.log(`  ${tidDisplay.padEnd(40)} ${pr.toFixed(3).padStart(10)} ${ms.toFixed(3).padStart(11)} ${passAtK.toFixed(3).padStart(8)} ${passPowerK.toFixed(3).padStart(8)} ${md.toFixed(3).padStart(11)} ${mt.toFixed(0).padStart(8)} ${String(k).padStart(7)}`);
  }
}
