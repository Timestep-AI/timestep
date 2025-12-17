import { stopServers } from './server';

async function globalTeardown() {
  console.log('Global teardown: Stopping servers...');
  await stopServers();
  console.log('Global teardown: Servers stopped');
}

export default globalTeardown;

