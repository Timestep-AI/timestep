import { startServers } from './server';

async function globalSetup() {
  console.log('Global setup: Starting servers...');
  await Promise.race([
    startServers(),
    new Promise((_, reject) => setTimeout(() => reject(new Error('Server startup timed out')), 50000)),
  ]);
  console.log('Global setup: Servers are ready');
}

export default globalSetup;

