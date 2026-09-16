// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include "onnxruntime_cxx_api.h"

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  try {
    const auto start = std::chrono::steady_clock::now();
    const auto level = argc > 3 ? ORT_LOGGING_LEVEL_INFO : ORT_LOGGING_LEVEL_WARNING;
    Ort::Env env(level, "phi4-session-probe");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    options.AppendExecutionProvider("WebGPU", {
        {"weightLoadAcceleration", argv[2]},
        {"powerPreference", "high-performance"},
        {"dawnBackendType", "D3D12"},
        {"validationMode", "basic"},
        {"enableGraphCapture", "0"},
        {"multiRotaryCacheConcatOffset", "4096"}});
    const auto before_session = std::chrono::steady_clock::now();
    const std::filesystem::path model{argv[1]};
    Ort::Session session(env, model.c_str(), options);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "RESULT {\"mode\":\"" << argv[2] << "\",\"session_ms\":"
              << std::chrono::duration<double, std::milli>(end - before_session).count()
              << ",\"setup_and_session_ms\":"
              << std::chrono::duration<double, std::milli>(end - start).count()
              << ",\"inputs\":" << session.GetInputCount()
              << ",\"outputs\":" << session.GetOutputCount() << "}" << std::endl;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
