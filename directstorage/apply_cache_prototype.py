"""Apply the optional persistent Dawn-cache prototype to an isolated PR checkout."""

import argparse
import difflib
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    args = parser.parse_args()
    relative = Path("onnxruntime/core/providers/webgpu/webgpu_context.cc")
    path = args.ort / relative
    original = path.read_text(encoding="utf-8")
    marker = '#include "core/providers/webgpu/webgpu_context.h"'
    if "BenchmarkDawnDiskCache" in original:
        raise RuntimeError("Cache prototype is already applied")
    updated = original.replace(
        marker,
        marker
        + """
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
#include "core/platform/env_var.h"
#include "core/providers/webgpu/benchmark_dawn_disk_cache.h"
#endif""",
        1,
    )
    old_class = """class DawnPlatform final : public dawn::platform::Platform {
 public:
  std::unique_ptr<dawn::platform::WorkerTaskPool> CreateWorkerTaskPool() override {
    return dawn::platform::WorkerTaskPool::CreateDawnDefault(GetDawnWorkerThreadCount());
  }
};"""
    new_class = """class DawnPlatform final : public dawn::platform::Platform {
 public:
  DawnPlatform() {
    const auto directory = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_BENCH_CACHE_DIR");
    if (!directory.empty()) {
      cache_ = std::make_unique<BenchmarkDawnDiskCache>(directory);
    }
  }

  dawn::platform::CachingInterface* GetCachingInterface() override {
    return cache_.get();
  }

  void WriteCacheStatistics() noexcept {
    if (cache_) cache_->WriteStatistics();
  }

  std::unique_ptr<dawn::platform::WorkerTaskPool> CreateWorkerTaskPool() override {
    return dawn::platform::WorkerTaskPool::CreateDawnDefault(GetDawnWorkerThreadCount());
  }

 private:
  std::unique_ptr<BenchmarkDawnDiskCache> cache_;
};"""
    if old_class not in updated:
        raise RuntimeError("Unexpected DawnPlatform implementation")
    updated = updated.replace(old_class, new_class, 1)
    old_destructor = """WebGpuContext::~WebGpuContext() {
  ContinueInitialize();
  if (initialize_future_.valid()) {
    initialize_future_.wait();
  }
}"""
    new_destructor = (
        old_destructor[:-1]
        + """#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  GetDawnPlatform().WriteCacheStatistics();
#endif
}"""
    )
    if old_destructor not in updated:
        raise RuntimeError("Unexpected WebGpuContext destructor")
    updated = updated.replace(old_destructor, new_destructor, 1)
    header_relative = Path(
        "onnxruntime/core/providers/webgpu/benchmark_dawn_disk_cache.h"
    )
    header_source = Path(__file__).with_name("dawn_disk_cache.h")
    header = header_source.read_text(encoding="utf-8")
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            updated.splitlines(True),
            fromfile="a/" + relative.as_posix(),
            tofile="b/" + relative.as_posix(),
        )
    )
    patch += "".join(
        difflib.unified_diff(
            [],
            header.splitlines(True),
            fromfile="/dev/null",
            tofile="b/" + header_relative.as_posix(),
        )
    )
    args.patch_output.write_text(patch, encoding="utf-8", newline="\n")
    path.write_text(updated, encoding="utf-8", newline="\n")
    shutil.copyfile(header_source, args.ort / header_relative)
    print(args.patch_output)


if __name__ == "__main__":
    main()
