// Standalone ORT C++ WebGPU generator. No GenAI CPU-search dependency.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#ifdef _WIN32
#include <dxgi1_4.h>
#include <wrl/client.h>
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Tensor = std::shared_ptr<Ort::Value>;
using Tensors = std::map<std::string, Tensor>;
using Shape = std::vector<int64_t>;
double Milliseconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
void Require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}
ONNXTensorElementDataType Type(const std::string& name) {
  if (name == "float16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (name == "float32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (name == "int64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  if (name == "int32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  throw std::runtime_error("Unsupported tensor type: " + name);
}
size_t Bytes(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2
         : type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8
                                                       : 4;
}
int GreedyBits(const uint8_t* values, bool half, int vocabulary,
               const std::set<int>& eos) {
  // Monotonic IEEE rank, identical to the GPU shader. No scalar FP16-to-float
  // conversion; both signed zeros tie and every NaN/Infinity is still rejected.
  const uint32_t center = half ? 0x8000u : 0x80000000u;
  const uint32_t magnitudeMask = half ? 0x7fffu : 0x7fffffffu;
  const uint32_t exponentMask = half ? 0x7c00u : 0x7f800000u;
  uint32_t best = 0;
  int token = -1;
  for (int i = 0; i < vocabulary; ++i) {
    const uint32_t bits = half ? reinterpret_cast<const uint16_t*>(values)[i]
                               : reinterpret_cast<const uint32_t*>(values)[i];
    Require((bits & exponentMask) != exponentMask, "Nonfinite logits");
    const auto magnitude = bits & magnitudeMask;
    const auto rank = (bits & center) ? center - magnitude : center + magnitude;
    if (rank > best && !eos.contains(i)) {
      best = rank;
      token = i;
    }
  }
  Require(token >= 0, "No valid next token");
  return token;
}
json AdapterUsage() {
  json result = json::array();
#ifdef _WIN32
  using Microsoft::WRL::ComPtr;
  ComPtr<IDXGIFactory1> factory;
  Require(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))),
          "Cannot inspect DXGI adapters");
  for (UINT i = 0;; ++i) {
    ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(adapter->GetDesc1(&desc))) continue;
    ComPtr<IDXGIAdapter3> memory;
    DXGI_QUERY_VIDEO_MEMORY_INFO usage{};
    if (SUCCEEDED(adapter.As(&memory)) &&
        SUCCEEDED(memory->QueryVideoMemoryInfo(
            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &usage)))
      result.push_back({{"vendorId", desc.VendorId},
                        {"deviceId", desc.DeviceId},
                        {"processLocalBytes", usage.CurrentUsage}});
  }
#endif
  return result;
}

#include "gpu.h"

struct Generator {
  Ort::Env env;
  std::unique_ptr<NativeGpu> context;
  Ort::AllocatorWithDefaultOptions cpu;
  // ORT's allocator identity remains device 0 even for external context ID 1.
  Ort::MemoryInfo gpuInfo{"WebGPU_Buf", OrtDeviceAllocator, 0,
                          OrtMemTypeDefault};
  Ort::Allocator gpu{nullptr};
  Ort::RunOptions run;
  json manifest, options, model, evidence;
  std::filesystem::path root;
  std::map<std::string, std::unique_ptr<Ort::Session>> sessions;
  Tensors past, present, decodeInputs, decodeOutputs, embeddingInputs,
      embeddingOutputs;
  Tensors prefillCache;
  std::unique_ptr<NativeSampler> sampler;
  wgpu::Buffer logitsReadback;
  std::unique_ptr<Ort::IoBinding> decoderBinding, embeddingBinding;
  size_t pastLength = 0;
  size_t maskFilled = 0;
  bool capture, gpuSampling;
  int vocabulary;
  std::set<int> eos;
  static void Log(void* context, OrtLoggingLevel, const char*, const char*,
                  const char*, const char* message) {
    auto* self = static_cast<Generator*>(context);
    if (std::strstr(message,
                    "Replaying the captured WebGpuExecutionProvider graph"))
      self->replays++;
    if (std::strstr(message, "Replaying") == nullptr)
      std::cerr << message << '\n';
  }
  int replays = 0;

  Generator(const json& entry, const json& settings)
      : env(settings.value("diagnostics", false) ? ORT_LOGGING_LEVEL_VERBOSE
                                                 : ORT_LOGGING_LEVEL_WARNING,
            "native-benchmark", Log, this),
        manifest(entry.at("manifest")),
        options(settings),
        model(manifest.at("config").at("model")),
        root(entry.at("root").get<std::string>()),
        capture(settings.value("capture", true)),
        gpuSampling(settings.value("gpuSampling", "auto") != "cpu" &&
                    !model.contains("embedding")),
        vocabulary(model.at("vocab_size")) {
    if (model.contains("eos_token_id")) {
      auto ids = model["eos_token_id"];
      if (ids.is_array())
        for (int token : ids) eos.insert(token);
      else
        eos.insert(ids.get<int>());
    }
    Require(settings.value("gpuSampling", "auto") != "gpu" || gpuSampling,
            "GPU feedback unsupported for auxiliary embedding models");
    context = std::make_unique<NativeGpu>();
    run.AddConfigEntry("disable_synchronize_execution_providers", "1");
    sessions["prefill"] = Session("decoder", false, false);
    gpu = Ort::Allocator(*sessions["prefill"], gpuInfo);
    sessions["decode"] = Session("decoder", capture, true);
    if (model.contains("embedding")) {
      sessions["embedding-prefill"] = Session("embedding", false, false);
      sessions["embedding-decode"] = Session("embedding", false, true);
    }
    for (const auto& state : manifest["states"]) {
      const std::string input = state["input"], output = state["output"];
      past[input] = Allocate(Type(state["type"]), state["dims"].get<Shape>());
      present[output] =
          state["shared"].get<bool>()
              ? past[input]
              : Allocate(Type(state["type"]), state["dims"].get<Shape>());
    }
    if (model.contains("embedding")) {
      embeddingInputs = Inputs("embedding", {0}, true, true);
      const auto& spec = manifest["sessions"]["embedding"]["outputs"][0];
      Shape dims;
      for (const auto& dim : spec["dims"])
        dims.push_back(dim.is_number_integer() ? dim.get<int64_t>() : 1);
      embeddingOutputs[model["embedding"]["outputs"]["inputs_embeds"]] =
          Allocate(Type(spec["type"]), dims);
      embeddingBinding = Bind(*sessions["embedding-decode"], embeddingInputs,
                              embeddingOutputs);
    }
    decodeInputs = Inputs("decoder", {0}, true, false);
    decodeInputs.insert(past.begin(), past.end());
    decodeOutputs = present;
    if (gpuSampling) {
      const auto logitsName = model["decoder"]["outputs"]["logits"];
      bool half = false;
      for (const auto& output : manifest["sessions"]["decoder"]["outputs"])
        if (output["name"] == logitsName) half = output["type"] == "float16";
      sampler = std::make_unique<NativeSampler>(
          *context, vocabulary, half, eos,
          NativeGpu::BufferOf(
              decodeInputs.at(model["decoder"]["inputs"]["input_ids"])),
          settings.value("maxPending", 16));
    }
    evidence = {
        {"graphCaptureRequested", capture},
        {"captureScope", capture ? "decoder-decode" : "none"},
        {"samplingDevice", gpuSampling ? "gpu" : "cpu"},
        {"sampler", gpuSampling ? "shared-exact-ieee-wgsl" : "cpu-greedy"},
        {"maxPendingReadbacks", settings.value("maxPending", 16)},
        {"tokenDelivery", "bounded-ordered-cpu-delivery"},
        {"kvCapacity", manifest["maxLength"]},
        {"decodeRuns", 0},
        {"embeddingCapture", false},
        {"providerOptions", providerOptions},
        {"validationMode", "basic"},
        {"robustnessRequested", true},
        {"device", context->evidence}};
  }
  // Session options and their effective overrides are kept in the result.
  json providerOptions = json::object();
  Ort::SessionOptions BaseOptions(const std::string& section, bool captured) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(1);
    if (options.value("diagnostics", false)) {
      so.SetLogSeverityLevel(0);
      Ort::ThrowOnError(Ort::GetApi().SetSessionLogVerbosityLevel(so, 1));
    }
    std::map<std::string, std::string> provider;
    if (model[section].contains("session_options")) {
      for (const auto& item : model[section]["session_options"].value(
               "provider_options", json::array()))
        if (item.contains("webgpu"))
          for (auto it = item["webgpu"].begin(); it != item["webgpu"].end();
               ++it)
            provider[it.key()] = it.value().is_string()
                                     ? it.value().get<std::string>()
                                     : it.value().dump();
    }
    provider["enableGraphCapture"] = captured ? "1" : "0";
    provider["dawnBackendType"] = "D3D12";
    provider["powerPreference"] = "high-performance";
    provider["validationMode"] = "basic";
    provider["enableRobustness"] = "1";
    provider["enableInt64"] = "1";
    for (const auto& [key, value] : context->Provider()) provider[key] = value;
    std::vector<const char*> keys, values;
    for (const auto& [key, value] : provider) {
      keys.push_back(key.c_str());
      values.push_back(value.c_str());
    }
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(
        so, "WebGPU", keys.data(), values.data(), keys.size()));
    providerOptions[section + (captured ? "-captured" : "-uncaptured")] =
        provider;
    return so;
  }
  Shape StaticShape(const std::string& section, const json& input) {
    const auto& names = model[section]["inputs"];
    const std::string name = input["name"];
    for (const auto& state : manifest["states"])
      if (state["input"] == name) return state["dims"].get<Shape>();
    if (names.value("input_ids", "") == name) return {1, 1};
    if (names.value("attention_mask", "") == name)
      return {1, manifest["maxLength"].get<int64_t>()};
    if (names.value("position_ids", "") == name)
      return input["dims"].size() == 3 ? Shape{3, 1, 1} : Shape{1, 1};
    if (names.value("inputs_embeds", "") == name)
      return {1, 1, input["dims"][2].get<int64_t>()};
    if (names.value("image_features", "") == name ||
        names.value("audio_features", "") == name)
      return {0, input["dims"][1].get<int64_t>()};
    throw std::runtime_error("Unknown static input: " + name);
  }
  std::unique_ptr<Ort::Session> Session(const std::string& section,
                                        bool captured, bool fixed) {
    auto so = BaseOptions(section, captured);
    if (fixed) {
      std::map<std::string, int64_t> overrides;
      for (const auto& input : manifest["sessions"][section]["inputs"]) {
        const auto shape = StaticShape(section, input);
        for (size_t i = 0; i < shape.size(); ++i)
          if (input["dims"][i].is_string()) {
            const std::string symbol = input["dims"][i];
            if (symbol.empty()) continue;
            Require(
                !overrides.contains(symbol) || overrides[symbol] == shape[i],
                "Conflicting shape override");
            overrides[symbol] = shape[i];
          }
      }
      for (const auto& [name, dimension] : overrides)
        so.AddFreeDimensionOverrideByName(name.c_str(), dimension);
    }
    const auto filename =
        root / manifest["sessions"][section]["file"].get<std::string>();
    return std::make_unique<Ort::Session>(env, filename.c_str(), so);
  }
  json InputSpec(const std::string& section, const std::string& name) {
    for (const auto& spec : manifest["sessions"][section]["inputs"])
      if (spec["name"] == name) return spec;
    throw std::runtime_error("Missing input: " + name);
  }
  Tensor Allocate(ONNXTensorElementDataType type, const Shape& dims,
                  bool onCpu = false) {
    if (onCpu)
      return std::make_shared<Ort::Value>(
          Ort::Value::CreateTensor(cpu, dims.data(), dims.size(), type));
    size_t count = 1;
    for (auto dim : dims) {
      Require(dim >= 0 && (!dim || count <= SIZE_MAX / size_t(dim)),
              "Invalid tensor dimensions");
      count *= size_t(dim);
    }
    Require(count <= SIZE_MAX / Bytes(type), "Tensor size overflow");
    const size_t bytes = count * Bytes(type);
    // Own external buffers exactly as the Web runner does. ORT's allocator can
    // queue a deferred zero-fill; an external WriteBuffer must not race it.
    auto buffer = context->Buffer(bytes, wgpu::BufferUsage::Storage |
                                             wgpu::BufferUsage::CopySrc |
                                             wgpu::BufferUsage::CopyDst);
    auto value =
        Ort::Value::CreateTensor(gpuInfo, static_cast<void*>(buffer.Get()),
                                 bytes, dims.data(), dims.size(), type);
    return Tensor(new Ort::Value(std::move(value)),
                  [buffer](Ort::Value* item) { delete item; });
  }
  void Copy(const Tensor& source, const Tensor& target) {
    const OrtValue* src = *source;
    OrtValue* dst = *target;
    Ort::ThrowOnError(Ort::GetApi().CopyTensors(env, &src, &dst, nullptr, 1));
  }
  void Upload(const Tensor& tensor, const std::vector<int64_t>& values,
              bool onCpu = false) {
    const auto info = tensor->GetTensorTypeAndShapeInfo();
    Require(info.GetElementCount() == values.size(), "Upload size mismatch");
    if (values.empty()) return;
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      if (onCpu)
        std::copy(values.begin(), values.end(),
                  tensor->GetTensorMutableData<int64_t>());
      else
        context->queue.WriteBuffer(NativeGpu::BufferOf(tensor), 0,
                                   values.data(),
                                   values.size() * sizeof(int64_t));
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      std::vector<int32_t> staging(onCpu ? 0 : values.size());
      auto target =
          onCpu ? tensor->GetTensorMutableData<int32_t>() : staging.data();
      for (size_t i = 0; i < values.size(); ++i) {
        Require(values[i] >= INT32_MIN && values[i] <= INT32_MAX,
                "Integer input overflow");
        target[i] = static_cast<int32_t>(values[i]);
      }
      if (!onCpu)
        context->queue.WriteBuffer(NativeGpu::BufferOf(tensor), 0, target,
                                   values.size() * sizeof(int32_t));
    } else
      Require(values.empty(), "Only integer input uploads are supported");
  }
  Tensors Inputs(const std::string& section, const std::vector<int64_t>& tokens,
                 bool fixed, bool onCpu, Tensor embeddings = {}) {
    Tensors result;
    const auto& names = model[section]["inputs"];
    for (const auto& spec : manifest["sessions"][section]["inputs"]) {
      const std::string name = spec["name"];
      if (past.contains(name)) continue;
      Shape dims;
      std::vector<int64_t> values;
      if (names.value("input_ids", "") == name) {
        dims = {1, int64_t(tokens.size())};
        values = tokens;
      } else if (names.value("position_ids", "") == name) {
        const int axes = spec["dims"].size() == 3 ? 3 : 1;
        dims = axes == 3 ? Shape{3, 1, int64_t(tokens.size())}
                         : Shape{1, int64_t(tokens.size())};
        for (int axis = 0; axis < axes; ++axis)
          for (size_t i = 0; i < tokens.size(); ++i)
            values.push_back(pastLength + i);
      } else if (names.value("attention_mask", "") == name) {
        dims = {1, fixed ? manifest["maxLength"].get<int64_t>()
                         : int64_t(pastLength + tokens.size())};
        values.assign(size_t(dims[1]), 0);
        std::fill_n(values.begin(), pastLength + tokens.size(), 1);
      } else if (names.value("inputs_embeds", "") == name) {
        result[name] =
            embeddings ? embeddings
                       : embeddingOutputs.at(
                             model["embedding"]["outputs"]["inputs_embeds"]);
        continue;
      } else if (names.value("image_features", "") == name ||
                 names.value("audio_features", "") == name) {
        dims = {0, spec["dims"][1].get<int64_t>()};
      } else
        throw std::runtime_error("Unrecognized input: " + name);
      const std::string key = section + "/" + name + "/" + json(dims).dump() +
                              (onCpu ? "cpu" : "gpu");
      if (!prefillCache.contains(key))
        prefillCache[key] = Allocate(Type(spec["type"]), dims, onCpu);
      result[name] = prefillCache[key];
      Upload(result[name], values, onCpu);
    }
    return result;
  }
  std::unique_ptr<Ort::IoBinding> Bind(Ort::Session& session,
                                       const Tensors& inputs,
                                       const Tensors& outputs) {
    auto binding = std::make_unique<Ort::IoBinding>(session);
    for (const auto& [name, value] : inputs)
      binding->BindInput(name.c_str(), *value);
    for (const auto& [name, value] : outputs)
      binding->BindOutput(name.c_str(), *value);
    return binding;
  }
  void CopyState() {
    auto encoder = context->device.CreateCommandEncoder();
    bool any = false;
    for (const auto& state : manifest["states"])
      if (!state["shared"].get<bool>()) {
        auto from = present.at(state["output"]), to = past.at(state["input"]);
        const auto info = to->GetTensorTypeAndShapeInfo();
        const size_t bytes =
            info.GetElementCount() * Bytes(info.GetElementType());
        encoder.CopyBufferToBuffer(NativeGpu::BufferOf(from), 0,
                                   NativeGpu::BufferOf(to), 0,
                                   (bytes + 3) / 4 * 4);
        any = true;
      }
    if (any) {
      auto command = encoder.Finish();
      context->queue.Submit(1, &command);
    }
  }
  void Reset() {
    pastLength = 0;
    maskFilled = 0;
    auto encoder = context->device.CreateCommandEncoder();
    for (const auto& state : manifest["states"])
      if (!state["shared"].get<bool>()) {
        for (auto tensor :
             {past.at(state["input"]), present.at(state["output"])}) {
          encoder.ClearBuffer(NativeGpu::BufferOf(tensor));
        }
      }
    const auto& names = model["decoder"]["inputs"];
    if (names.contains("attention_mask"))
      encoder.ClearBuffer(
          NativeGpu::BufferOf(decodeInputs.at(names["attention_mask"])));
    auto command = encoder.Finish();
    context->queue.Submit(1, &command);
  }
  int Greedy(const Tensor& logits) {
    const auto info = logits->GetTensorTypeAndShapeInfo();
    Require(info.GetElementCount() >= size_t(vocabulary) &&
                info.GetElementCount() % vocabulary == 0,
            "Invalid logits shape");
    const size_t bytes = Bytes(info.GetElementType()),
                 offset = (info.GetElementCount() - vocabulary) * bytes;
    const size_t alignedOffset = offset / 4 * 4, skip = offset - alignedOffset,
                 size = (skip + vocabulary * bytes + 3) / 4 * 4;
    if (!logitsReadback || logitsReadback.GetSize() < size)
      logitsReadback = context->Buffer(
          size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
    auto encoder = context->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(NativeGpu::BufferOf(logits), alignedOffset,
                               logitsReadback, 0, size);
    auto command = encoder.Finish();
    context->queue.Submit(1, &command);
    bool ready = false;
    auto future = logitsReadback.MapAsync(
        wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus status, wgpu::StringView, bool* result) {
          *result = status == wgpu::MapAsyncStatus::Success;
        },
        &ready);
    Require(context->instance.WaitAny(future, UINT64_MAX) ==
                    wgpu::WaitStatus::Success &&
                ready,
            "Logits readback failed");
    struct Unmap {
      wgpu::Buffer buffer;
      ~Unmap() { buffer.Unmap(); }
    } unmap{logitsReadback};
    const auto* values = static_cast<const uint8_t*>(
                             logitsReadback.GetConstMappedRange(0, size)) +
                         skip;
    context->Check();
    return GreedyBits(
        values, info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        vocabulary, eos);
  }
  int Prefill(const std::vector<int64_t>& prompt) {
    Tensor embeddings;
    if (model.contains("embedding")) {
      auto inputs = Inputs("embedding", prompt, false, true);
      auto binding = Bind(*sessions["embedding-prefill"], inputs, {});
      const std::string name = model["embedding"]["outputs"]["inputs_embeds"];
      binding->BindOutput(name.c_str(), gpuInfo);
      sessions["embedding-prefill"]->Run(run, *binding);
      auto values = binding->GetOutputValues();
      embeddings = std::make_shared<Ort::Value>(std::move(values[0]));
    }
    auto inputs = Inputs("decoder", prompt, false, false, embeddings);
    inputs.insert(past.begin(), past.end());
    auto binding = Bind(*sessions["prefill"], inputs, present);
    const std::string logitsName = model["decoder"]["outputs"]["logits"];
    binding->BindOutput(logitsName.c_str(), gpuInfo);
    sessions["prefill"]->Run(run, *binding);
    auto values = binding->GetOutputValues();
    auto names = binding->GetOutputNames();
    Tensor logits;
    for (size_t i = 0; i < names.size(); ++i)
      if (names[i] == logitsName)
        logits = std::make_shared<Ort::Value>(std::move(values[i]));
    Require(bool(logits), "Prefill logits absent");
    CopyState();
    if (!decoderBinding) {
      auto info = logits->GetTensorTypeAndShapeInfo();
      auto dims = info.GetShape();
      Require(dims.back() == vocabulary, "Unexpected vocabulary dimension");
      std::fill(dims.begin(), dims.end() - 1, 1);
      decodeOutputs[logitsName] = Allocate(info.GetElementType(), dims);
      decoderBinding = Bind(*sessions["decode"], decodeInputs, decodeOutputs);
      if (gpuSampling)
        sampler->Bind(NativeGpu::BufferOf(decodeOutputs.at(logitsName)));
    }
    pastLength += prompt.size();
    return Greedy(logits);
  }
  void DecodeInputs(int previous) {
    const auto& names = model["decoder"]["inputs"];
    if (!gpuSampling && names.contains("input_ids"))
      Upload(decodeInputs.at(names["input_ids"]), {previous});
    if (names.contains("position_ids")) {
      auto tensor = decodeInputs.at(names["position_ids"]);
      Upload(tensor, std::vector<int64_t>(
                         tensor->GetTensorTypeAndShapeInfo().GetElementCount(),
                         pastLength));
    }
    if (names.contains("attention_mask")) {
      const auto tensor = decodeInputs.at(names["attention_mask"]);
      const size_t active = pastLength + 1, count = active - maskFilled;
      const auto buffer = NativeGpu::BufferOf(tensor);
      if (tensor->GetTensorTypeAndShapeInfo().GetElementType() ==
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        std::vector<int64_t> ones(count, 1);
        context->queue.WriteBuffer(buffer, maskFilled * 8, ones.data(),
                                   count * 8);
      } else {
        std::vector<int32_t> ones(count, 1);
        context->queue.WriteBuffer(buffer, maskFilled * 4, ones.data(),
                                   count * 4);
      }
      maskFilled = active;
    }
    if (model.contains("embedding")) {
      const std::string name = model["embedding"]["inputs"]["input_ids"];
      Upload(embeddingInputs.at(name), {previous}, true);
      // BindInput may stage a CPU tensor for a GPU consumer immediately. Rebind
      // changed CPU values; retaining this binding can silently reuse old IDs.
      embeddingBinding->BindInput(name.c_str(), *embeddingInputs.at(name));
      sessions["embedding-decode"]->Run(run, *embeddingBinding);
    }
  }
  json Generate(const std::vector<int64_t>& prompt, int length) {
    Require(length >= 2 &&
                prompt.size() + length <= manifest["maxLength"].get<size_t>(),
            "Workload exceeds KV capacity");
    for (auto token : prompt)
      Require(token >= 0 && token < vocabulary,
              "Input token outside vocabulary");
    Reset();
    context->Wait();
    std::vector<int> generated;
    std::vector<double> delivery;
    const auto start = Clock::now();
    const int firstToken = Prefill(prompt);
    const auto first = Clock::now();
    generated.push_back(firstToken);
    delivery.push_back(Milliseconds(start, first));
    if (gpuSampling)
      Upload(decodeInputs.at(model["decoder"]["inputs"]["input_ids"]),
             {firstToken});
    auto deliver = [&] {
      generated.push_back(sampler->Deliver());
      delivery.push_back(Milliseconds(start, Clock::now()));
    };
    for (int i = 1; i < length; ++i) {
      DecodeInputs(generated.back());
      sessions["decode"]->Run(run, *decoderBinding);
      CopyState();
      evidence["decodeRuns"] = evidence["decodeRuns"].get<int>() + 1;
      if (gpuSampling) {
        if (sampler->Full()) deliver();
        sampler->Enqueue();
      } else {
        generated.push_back(
            Greedy(decodeOutputs.at(model["decoder"]["outputs"]["logits"])));
        delivery.push_back(Milliseconds(start, Clock::now()));
      }
      ++pastLength;
    }
    if (gpuSampling)
      while (sampler->Pending()) deliver();
    const auto end = Clock::now();
    Require(generated.size() == size_t(length), "Incomplete CPU delivery");
    evidence["runtimeReplayMessages"] = replays;
    const double ttft = Milliseconds(start, first),
                 decode = Milliseconds(first, end);
    Require(ttft > 0 && decode > 0, "Invalid elapsed time");
    return {{"generated", generated},
            {"tokenDeliveryMs", delivery},
            {"ttftMs", ttft},
            {"e2eMs", Milliseconds(start, end)},
            {"prefillTps", prompt.size() * 1000. / ttft},
            {"decodeTps", (length - 1) * 1000. / decode}};
  }
  ~Generator() {
    sampler.reset();
    decoderBinding.reset();
    embeddingBinding.reset();
    // Captured sessions must release references before bound tensors/allocator.
    sessions.erase("sampler");
    sessions.erase("decode");
    sessions.erase("embedding-decode");
    past.clear();
    present.clear();
    decodeInputs.clear();
    decodeOutputs.clear();
    embeddingInputs.clear();
    embeddingOutputs.clear();
    prefillCache.clear();
    gpu = Ort::Allocator(nullptr);
    sessions.clear();
  }
};

#include "probe.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "ort_webgpu_benchmark run-config.json result.json\n";
    return 2;
  }
  json result = {
      {"success", false}, {"rows", json::array()}, {"errors", json::array()}};
  try {
    const auto input = json::parse(std::ifstream(argv[1]));
    if (input.value("probe", false)) {
      result = ProbeSampler(input);
      std::ofstream(argv[2]) << result.dump(2) << '\n';
      return 0;
    }
    Require(input["options"].value("maxPending", 16) >= 1 &&
                input["options"].value("maxPending", 16) <= 64,
            "Invalid pending count");
    Generator generator(input["model"], input["options"]);
    result["name"] = input["model"]["name"];
    result["ortVersion"] = OrtGetApiBase()->GetVersionString();
    for (const auto& c : input["model"]["cases"]) {
      const auto prompt = c["prompt"].get<std::vector<int64_t>>();
      const int length = input["generationLength"];
      auto warmup = generator.Generate(prompt, length);
      json samples = json::array();
      result["gpuMemory"] = AdapterUsage();
      for (int rep = 0; rep < input["repetitions"].get<int>(); ++rep) {
        auto sample = generator.Generate(prompt, length);
        Require(sample["generated"] == warmup["generated"],
                "Output changed after reset: expected " +
                    warmup["generated"].dump() + ", got " +
                    sample["generated"].dump());
        std::cerr << "input=" << prompt.size() << " rep=" << rep
                  << " decode=" << sample["decodeTps"] << '\n';
        samples.push_back(std::move(sample));
      }
      result["rows"].push_back({{"pl", prompt.size()},
                                {"prompt", prompt},
                                {"tg", length},
                                {"warmup", warmup},
                                {"samples", samples}});
    }
    result["executionEvidence"] = generator.evidence;
    result["success"] = true;
    if (input["options"].value("diagnostics", false) &&
        input["options"].value("capture", true))
      Require(generator.replays > 0, "No actual capture replay observed");
  } catch (const std::exception& e) {
    result["success"] = false;
    result["errors"].push_back(e.what());
    std::cerr << e.what() << '\n';
  }
  std::ofstream(argv[2]) << result.dump(2) << '\n';
  return result["success"].get<bool>() ? 0 : 1;
}
