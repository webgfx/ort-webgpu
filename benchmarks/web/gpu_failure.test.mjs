import { test } from 'node:test';
import assert from 'node:assert/strict';
import { monitorGpuDevice } from './gpu_failure.mjs';

function deviceFixture() {
  let lose, onError;
  return { device: { lost: new Promise(resolve => { lose = resolve; }),
    addEventListener(_name, handler) { onError = handler; }, removeEventListener() { onError = undefined; } },
    lose: info => lose(info), error: message => onError?.({ error: { message } }) };
}
test('a device loss rejects an inference that never completes', async () => {
  const fixture = deviceFixture();
  const monitor = monitorGpuDevice(fixture.device);
  const result = monitor.race(() => new Promise(() => {}));
  fixture.lose({ reason: 'unknown', message: 'DXGI_ERROR_DEVICE_HUNG' });
  await assert.rejects(result, /DEVICE_HUNG/);
  assert.equal(monitor.failed, true);
  await assert.rejects(monitor.race(() => 1), /DEVICE_HUNG/);
});
test('uncaptured GPU errors fail promptly, but deliberate shutdown does not', async () => {
  const fixture = deviceFixture();
  const monitor = monitorGpuDevice(fixture.device);
  assert.equal(await monitor.race(() => 42), 42);
  fixture.error('invalid dispatch');
  assert.throws(() => monitor.throwIfFailed(), /invalid dispatch/);
  const second = deviceFixture();
  const closed = monitorGpuDevice(second.device);
  closed.close();
  second.lose({ reason: 'destroyed', message: '' });
  await Promise.resolve();
  assert.equal(closed.failed, false);
});
