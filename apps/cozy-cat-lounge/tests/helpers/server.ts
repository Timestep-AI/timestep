import { spawn, ChildProcess, execSync } from 'child_process';
import * as path from 'path';
import * as http from 'http';

const APP_ROOT = path.resolve(__dirname, '../..');
const FRONTEND_DIR = path.join(APP_ROOT, 'frontend');
const BACKEND_PORT = process.env.PORT ? parseInt(process.env.PORT) : 8000;
const FRONTEND_PORT = 3000;

// Backend name from env var (defaults to 'python' for Python)
const BACKEND_NAME = process.env.BACKEND || 'python';

let backendProcess: ChildProcess | null = null;
let frontendProcess: ChildProcess | null = null;
let isStarting = false;
let isStopping = false;

/**
 * Wait for a server to be ready by checking HTTP endpoint
 */
async function waitForServer(
  url: string,
  timeout: number = 60000,
  interval: number = 500
): Promise<void> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    try {
      await new Promise<void>((resolve, reject) => {
        const req = http.get(url, (res) => {
          res.on('data', () => {});
          res.on('end', () => {
            if (res.statusCode && res.statusCode >= 200 && res.statusCode < 500) {
              resolve();
            } else {
              reject(new Error(`Server returned status ${res.statusCode}`));
            }
          });
        });
        
        req.on('error', (err) => {
          reject(err);
        });
        
        req.setTimeout(2000, () => {
          req.destroy();
          reject(new Error('Request timeout'));
        });
      });
      
      // Server is ready
      return;
    } catch (error) {
      // Server not ready yet, wait and retry
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  }
  
  throw new Error(`Server at ${url} did not become ready within ${timeout}ms`);
}

/**
 * Start the backend server
 * Uses BACKEND env var to determine which backend to start (default: 'python')
 * @param skipLock - If true, skip the lock check (used when called from startServers)
 */
export async function startBackend(skipLock: boolean = false): Promise<void> {
  if (backendProcess) {
    console.log('Backend already running');
    return;
  }
  
  // Wait if another process is starting (unless we're the one starting)
  if (!skipLock) {
    while (isStarting) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (backendProcess) {
      return;
    }
  }

  console.log(`Starting backend server (${BACKEND_NAME})...`);
  
  // Kill any process using port 8000 first
  try {
    console.log('Killing any processes on port 8000...');
    execSync('fuser -k 8000/tcp 2>/dev/null || lsof -ti:8000 | xargs kill -9 2>/dev/null || true', { stdio: 'inherit' });
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for port to be released
  } catch (error) {
    console.log('Port 8000 cleanup completed (or was already free)');
  }
  
  const scriptPath = path.join(APP_ROOT, 'scripts', 'run-backend.sh');
  
  // Set environment variables
  const env = {
    ...process.env,
    PORT: BACKEND_PORT.toString(),
  };

  // Pass the backend name as argument to the script
  backendProcess = spawn('bash', [scriptPath, BACKEND_NAME], {
    cwd: APP_ROOT,
    env,
    stdio: 'pipe',
    shell: false,
  });

  backendProcess.stdout?.on('data', (data) => {
    console.log(`[Backend] ${data.toString().trim()}`);
  });

  backendProcess.stderr?.on('data', (data) => {
    console.error(`[Backend Error] ${data.toString().trim()}`);
  });

  backendProcess.on('error', (error) => {
    console.error(`[Backend] Failed to start: ${error.message}`);
    backendProcess = null;
  });

  backendProcess.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.error(`Backend process exited with code ${code}`);
    } else {
      console.log(`Backend process exited with code ${code}`);
    }
    backendProcess = null;
  });

  // Wait for backend to be ready
  await waitForServer(`http://localhost:${BACKEND_PORT}`, 60000);
  console.log('Backend server is ready');
}

/**
 * Start the frontend dev server
 * @param skipLock - If true, skip the lock check (used when called from startServers)
 */
export async function startFrontend(skipLock: boolean = false): Promise<void> {
  if (frontendProcess) {
    console.log('Frontend already running');
    return;
  }
  
  // Wait if another process is starting (unless we're the one starting)
  if (!skipLock) {
    while (isStarting) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (frontendProcess) {
      return;
    }
  }

  console.log('Starting frontend server...');

  // Kill any process using port 3000 first (and nearby ports just in case)
  try {
    console.log('Killing any processes on port 3000...');
    execSync('fuser -k 3000/tcp 2>/dev/null || lsof -ti:3000 | xargs kill -9 2>/dev/null || true', { stdio: 'inherit' });
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for port to be released
  } catch (error) {
    console.log('Port 3000 cleanup completed (or was already free)');
  }

  // Force port 3000 using Vite's --port and --strictPort flags (fail if port in use)
  frontendProcess = spawn('npm', ['run', 'dev', '--', '--port', '3000', '--strictPort', '--host', '0.0.0.0'], {
    cwd: FRONTEND_DIR,
    env: {
      ...process.env,
      PORT: '3000',
    },
    stdio: 'pipe',
    shell: true,
  });

  frontendProcess.stdout?.on('data', (data) => {
    const output = data.toString().trim();
    console.log(`[Frontend] ${output}`);
  });

  frontendProcess.stderr?.on('data', (data) => {
    const output = data.toString().trim();
    // Vite outputs to stderr, so we log it as info
    console.log(`[Frontend] ${output}`);
  });

  frontendProcess.on('error', (error) => {
    console.error(`[Frontend] Failed to start: ${error.message}`);
    frontendProcess = null;
  });

  frontendProcess.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.error(`Frontend process exited with code ${code}`);
    } else {
      console.log(`Frontend process exited with code ${code}`);
    }
    frontendProcess = null;
  });

  // Wait for frontend to be ready
  await waitForServer(`http://localhost:${FRONTEND_PORT}`, 60000);
  console.log('Frontend server is ready');
}

/**
 * Stop the backend server
 */
export async function stopBackend(): Promise<void> {
  if (backendProcess) {
    console.log('Stopping backend server...');
    backendProcess.kill('SIGTERM');
    
    // Wait for process to exit
    await new Promise<void>((resolve) => {
      if (!backendProcess) {
        resolve();
        return;
      }
      
      backendProcess.on('exit', () => {
        resolve();
      });
      
      // Force kill after 5 seconds
      setTimeout(() => {
        if (backendProcess) {
          backendProcess.kill('SIGKILL');
          resolve();
        }
      }, 5000);
    });
    
    backendProcess = null;
    console.log('Backend server stopped');
  }
}

/**
 * Stop the frontend server
 */
export async function stopFrontend(): Promise<void> {
  if (frontendProcess) {
    console.log('Stopping frontend server...');
    frontendProcess.kill('SIGTERM');
    
    // Wait for process to exit
    await new Promise<void>((resolve) => {
      if (!frontendProcess) {
        resolve();
        return;
      }
      
      frontendProcess.on('exit', () => {
        resolve();
      });
      
      // Force kill after 5 seconds
      setTimeout(() => {
        if (frontendProcess) {
          frontendProcess.kill('SIGKILL');
          resolve();
        }
      }, 5000);
    });
    
    frontendProcess = null;
    console.log('Frontend server stopped');
  }
}

/**
 * Start both servers (with lock to prevent concurrent starts)
 */
export async function startServers(): Promise<void> {
  if (isStarting) {
    // Wait for current start to complete
    while (isStarting) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return;
  }
  
  if (backendProcess && frontendProcess) {
    console.log('Servers already running');
    return;
  }
  
  isStarting = true;
  try {
    // Pass skipLock=true to avoid deadlock
    await Promise.all([startBackend(true), startFrontend(true)]);
  } finally {
    isStarting = false;
  }
}

/**
 * Stop both servers
 */
export async function stopServers(): Promise<void> {
  if (isStopping) {
    return;
  }
  
  isStopping = true;
  try {
    await Promise.all([stopBackend(), stopFrontend()]);
  } finally {
    isStopping = false;
  }
}
