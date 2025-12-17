import { startServers } from './server';

async function globalSetup() {
  console.log('Global setup: Starting servers for all tests...');
  
  // Set a timeout for server startup (50 seconds, leaving 10 seconds for tests)
  const startupTimeout = 50000;
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('Server startup timed out after 50 seconds')), startupTimeout);
  });
  
  await Promise.race([
    startServers(),
    timeoutPromise,
  ]);
  
  console.log('Global setup: Servers are ready');
}

export default globalSetup;

