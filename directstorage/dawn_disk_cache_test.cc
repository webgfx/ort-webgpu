// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dawn_disk_cache.h"

#include <cstdlib>
#include <iostream>
#include <thread>

// This isolated test exercises the cache without constructing a GPU device.
namespace dawn::platform {
CachingInterface::CachingInterface() = default;
CachingInterface::~CachingInterface() = default;
}  // namespace dawn::platform

void Check(bool condition, const char* expression) {
  if (!condition) {
    std::cerr << "FAIL: " << expression << std::endl;
    std::_Exit(1);
  }
}
#define CHECK(expression) Check((expression), #expression)

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const auto directory = std::filesystem::path{argv[1]} /
      ("run-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  onnxruntime::webgpu::BenchmarkDawnDiskCache cache{directory};
  const std::string key_string = "Dawn-version/device/compiled-shader-A";
  const std::string other_string = "Dawn-version/device/compiled-shader-B";
  const std::string payload_string = "compiled shader bytes for a deterministic test";
  const auto key = std::as_bytes(std::span(key_string));
  const auto other = std::as_bytes(std::span(other_string));
  const auto payload = std::as_bytes(std::span(payload_string));
  std::array<std::byte, 46> destination{};
  CHECK(destination.size() == payload.size());
  CHECK(cache.FindKey(key) == 0);
  cache.StoreData(key, payload);
  CHECK(cache.FindKey(key) == payload.size());
  CHECK(cache.LoadData(key, destination) == payload.size());
  CHECK(std::equal(payload.begin(), payload.end(), destination.begin()));
  CHECK(cache.FindKey(other) == 0);
  CHECK(cache.LoadData(key, std::span(destination).first(2)) == 0);

  std::filesystem::path entry;
  for (const auto& file : std::filesystem::directory_iterator(directory)) {
    if (file.path().extension() == ".bin") entry = file.path();
  }
  CHECK(!entry.empty());
  {
    std::fstream corrupt(entry, std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekp(-1, std::ios::end);
    corrupt.put('!');
  }
  CHECK(cache.LoadData(key, destination) == 0);
  cache.StoreData(key, payload);
  CHECK(cache.LoadData(key, destination) == payload.size());
  std::filesystem::resize_file(entry, 8);
  CHECK(cache.FindKey(key) == 0);
  cache.StoreData(key, payload);
  std::filesystem::resize_file(entry, std::filesystem::file_size(entry) - 1);
  CHECK(cache.FindKey(key) == 0);

  std::array<std::thread, 8> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      const std::string thread_key = key_string + std::to_string(i);
      const auto bytes = std::as_bytes(std::span(thread_key));
      std::array<std::byte, 46> output{};
      cache.StoreData(bytes, payload);
      CHECK(cache.LoadData(bytes, output) == payload.size());
      CHECK(std::equal(payload.begin(), payload.end(), output.begin()));
    });
  }
  for (auto& thread : threads) thread.join();
  cache.WriteStatistics();
  std::cout << "PASS: miss, roundtrip, wrong key/size, corruption, truncation, repair, parallel keys\n";
}
