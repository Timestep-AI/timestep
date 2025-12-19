import { spawn, ChildProcess, execSync } from 'child_process';
import * as path from 'path';
import * as http from 'http';

const APP_ROOT = path.resolve(__dirname, '../..');
const FRONTEND_DIR = path.join(APP_ROOT, 'frontend');
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 3000;

// Determine which backend to use based on BACKEND_TYPE env var
function getBackendArg(): string {
  const backendType = process.env.BACKEND_TYPE;
  if (backendType === 'ts' || backendType === 'typescript') {
    return 'backend-ts';
  }
  return 'backend';
}

let backendProcess: ChildProcess | null = null;
let frontendProcess: ChildProcess | null = null;
let isStarting = false;
let isStopping = false;

async function waitForServer(url: string, timeout: number = 60000): Promise<void> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    try {
      await new Promise<void>((resolve, reject) => {
        const req = http.get(url, (res) => {
          res.on('data', () => {});
          res.on('end', () => {
            if (res.statusCode && res.statusCode >= 200 && res.statusCode < 500) resolve();
            else reject(new Error(`Status ${res.statusCode}`));
          });
        });
        req.on('error', reject);
        req.setTimeout(2000, () => { req.destroy(); reject(new Error('Timeout')); });
      });
      return;
    } catch { await new Promise(resolve => setTimeout(resolve, 500)); }
  }
  throw new Error(`Server at ${url} did not become ready within ${timeout}ms`);
}

export async function startBackend(skipLock = false): Promise<void> {
  if (backendProcess) return;
  if (!skipLock) { while (isStarting) await new Promise(r => setTimeout(r, 100)); if (backendProcess) return; }

  const backendArg = getBackendArg();
  console.log(`Starting backend server (${backendArg})...`);
  try {
    execSync('fuser -k 8000/tcp 2>/dev/null || lsof -ti:8000 | xargs kill -9 2>/dev/null || true', { stdio: 'inherit' });
    await new Promise(r => setTimeout(r, 1000));
  } catch {}

  const scriptPath = path.join(APP_ROOT, 'scripts', 'run-backend.sh');
  // For TypeScript backend, run from project root
  const cwd = backendArg === 'backend-ts' ? path.resolve(APP_ROOT, '../..') : APP_ROOT;
  backendProcess = spawn('bash', [scriptPath, backendArg], { cwd, env: { ...process.env, PORT: BACKEND_PORT.toString() }, stdio: 'pipe' });
  backendProcess.stdout?.on('data', (d) => console.log(`[Backend] ${d.toString().trim()}`));
  backendProcess.stderr?.on('data', (d) => console.error(`[Backend Error] ${d.toString().trim()}`));
  backendProcess.on('exit', (code) => { console.log(`Backend exited with code ${code}`); backendProcess = null; });

  await waitForServer(`http://localhost:${BACKEND_PORT}`, 60000);
  console.log('Backend server is ready');
}

export async function startFrontend(skipLock = false): Promise<void> {
  if (frontendProcess) return;
  if (!skipLock) { while (isStarting) await new Promise(r => setTimeout(r, 100)); if (frontendProcess) return; }

  console.log('Starting frontend server...');
  try {
    execSync('fuser -k 3000/tcp 2>/dev/null || lsof -ti:3000 | xargs kill -9 2>/dev/null || true', { stdio: 'inherit' });
    await new Promise(r => setTimeout(r, 2000));
  } catch {}

  frontendProcess = spawn('npm', ['run', 'dev', '--', '--port', '3000', '--strictPort', '--host', '0.0.0.0'], {
    cwd: FRONTEND_DIR, env: { ...process.env, PORT: '3000' }, stdio: 'pipe', shell: true
  });
  frontendProcess.stdout?.on('data', (d) => console.log(`[Frontend] ${d.toString().trim()}`));
  frontendProcess.stderr?.on('data', (d) => console.log(`[Frontend] ${d.toString().trim()}`));
  frontendProcess.on('exit', (code) => { console.log(`Frontend exited with code ${code}`); frontendProcess = null; });

  await waitForServer(`http://localhost:${FRONTEND_PORT}`, 60000);
  console.log('Frontend server is ready');
}

export async function stopBackend(): Promise<void> {
  if (!backendProcess) return;
  console.log('Stopping backend server...');
  backendProcess.kill('SIGTERM');
  await new Promise<void>(resolve => {
    if (!backendProcess) { resolve(); return; }
    backendProcess.on('exit', resolve);
    setTimeout(() => { backendProcess?.kill('SIGKILL'); resolve(); }, 5000);
  });
  backendProcess = null;
  console.log('Backend server stopped');
}

export async function stopFrontend(): Promise<void> {
  if (!frontendProcess) return;
  console.log('Stopping frontend server...');
  frontendProcess.kill('SIGTERM');
  await new Promise<void>(resolve => {
    if (!frontendProcess) { resolve(); return; }
    frontendProcess.on('exit', resolve);
    setTimeout(() => { frontendProcess?.kill('SIGKILL'); resolve(); }, 5000);
  });
  frontendProcess = null;
  console.log('Frontend server stopped');
}

export async function startServers(): Promise<void> {
  if (isStarting) { while (isStarting) await new Promise(r => setTimeout(r, 100)); return; }
  if (backendProcess && frontendProcess) return;
  isStarting = true;
  try { await Promise.all([startBackend(true), startFrontend(true)]); }
  finally { isStarting = false; }
}

export async function stopServers(): Promise<void> {
  if (isStopping) return;
  isStopping = true;
  try { await Promise.all([stopBackend(), stopFrontend()]); }
  finally { isStopping = false; }
}
