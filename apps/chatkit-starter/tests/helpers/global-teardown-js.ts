import { stopServersJS } from './server';

async function globalTeardown() {
  console.log('Global teardown (JS): Stopping servers...');
  await stopServersJS();
  console.log('Global teardown (JS): Servers stopped');
}

export default globalTeardown;

