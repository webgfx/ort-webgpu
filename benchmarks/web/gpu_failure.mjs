/** Reject outstanding waits promptly when WebGPU loses the device. */
export function monitorGpuDevice(device) {
  let failure;
  let closed = false;
  let rejectFailure;
  const failurePromise = new Promise((_, reject) => { rejectFailure = reject; });
  // A device can fail between operations; retain the error without an unhandled rejection.
  failurePromise.catch(() => {});
  const fail = message => {
    if (closed || failure) return;
    failure = new Error(message);
    rejectFailure(failure);
  };
  const onError = event => fail(`WebGPU error: ${event.error?.message || event.error}`);
  device.addEventListener('uncapturederror', onError);
  device.lost.then(info => fail(`GPU device lost: ${info.reason}: ${info.message}`));
  return {
    get failed() { return !!failure; },
    throwIfFailed() { if (failure) throw failure; },
    race(operation) {
      if (failure) return Promise.reject(failure);
      return Promise.race([Promise.resolve().then(operation), failurePromise]);
    },
    close() { closed = true; device.removeEventListener('uncapturederror', onError); },
  };
}
