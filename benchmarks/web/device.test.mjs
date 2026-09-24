import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ORT_WEB_LIMITS, ortDeviceDescriptor } from './device.mjs';

test('enables supported ORT compute features and full compute limits', () => {
  const limits = Object.fromEntries(ORT_WEB_LIMITS.map(key => [key, 1024]));
  limits.maxComputeWorkgroupStorageSize = 32768;
  const descriptor = ortDeviceDescriptor({ features: new Set(['shader-f16', 'subgroups', 'subgroup-size-control']), limits });
  assert.deepEqual(descriptor.requiredFeatures, ['shader-f16', 'subgroups', 'subgroup-size-control']);
  assert.deepEqual(descriptor.requiredLimits, limits);
});

test('never requests features that the browser does not expose', () => {
  const limits = Object.fromEntries(ORT_WEB_LIMITS.map(key => [key, 256]));
  assert.deepEqual(ortDeviceDescriptor({ features: new Set(['shader-f16']), limits }).requiredFeatures, ['shader-f16']);
  assert.throws(() => ortDeviceDescriptor({ features: new Set(), limits }), /shader-f16/);
  assert.throws(() => ortDeviceDescriptor({ features: new Set(['shader-f16']), limits: {} }), /Invalid adapter limit/);
});
