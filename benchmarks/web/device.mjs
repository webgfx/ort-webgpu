// Match ORT WebGpuContext::GetAvailableRequiredFeatures/GetRequiredLimits for
// browser-visible capabilities. Do not enable hidden APIs or disable security.
export const ORT_WEB_FEATURES = [
  'shader-f16', 'subgroups', 'subgroup-size-control', 'timestamp-query',
  'chromium-experimental-subgroup-matrix',
];
export const ORT_WEB_LIMITS = [
  'maxBindGroups', 'maxComputeWorkgroupStorageSize', 'maxComputeWorkgroupsPerDimension',
  'maxStorageBuffersPerShaderStage', 'maxStorageBufferBindingSize', 'maxBufferSize',
  'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ',
];

export function ortDeviceDescriptor(adapter) {
  if (!adapter?.features.has('shader-f16')) throw new Error('The current models require WebGPU shader-f16');
  const requiredLimits = {};
  for (const key of ORT_WEB_LIMITS) {
    const value = adapter.limits[key];
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Invalid adapter limit ${key}`);
    requiredLimits[key] = value;
  }
  return { requiredFeatures: ORT_WEB_FEATURES.filter(feature => adapter.features.has(feature)), requiredLimits };
}
