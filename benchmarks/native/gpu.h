#pragma once
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <webgpu/webgpu_cpp.h>

#include <mutex>

#include "greedy_shader.h"

struct NativeGpu {
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;
  std::string failure;
  std::mutex failureMutex;
  json evidence;
  void Fail(std::string message) {
    std::lock_guard lock(failureMutex);
    if (failure.empty()) failure = std::move(message);
  }
  std::string Failure() {
    std::lock_guard lock(failureMutex);
    return failure;
  }
  NativeGpu() {
    dawnProcSetProcs(&dawn::native::GetProcs());
    wgpu::InstanceFeatureName timed = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor desc{};
    desc.requiredFeatureCount = 1;
    desc.requiredFeatures = &timed;
    instance = wgpu::CreateInstance(&desc);
    wgpu::RequestAdapterOptions request{};
    request.backendType = wgpu::BackendType::D3D12;
    request.powerPreference = wgpu::PowerPreference::HighPerformance;
    auto future = instance.RequestAdapter(
        &request, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter value,
           wgpu::StringView message, NativeGpu* self) {
          if (status == wgpu::RequestAdapterStatus::Success)
            self->adapter = std::move(value);
          else
            self->Fail(std::string(std::string_view(message)));
        },
        this);
    Require(instance.WaitAny(future, UINT64_MAX) == wgpu::WaitStatus::Success &&
                bool(adapter),
            "Native D3D12 adapter unavailable: " + Failure());
    wgpu::AdapterInfo info;
    adapter.GetInfo(&info);
    Require(info.adapterType != wgpu::AdapterType::CPU,
            "Software adapter is not a valid benchmark device");
    evidence = {
        {"vendorId", info.vendorID},
        {"deviceId", info.deviceID},
        {"description", std::string(std::string_view(info.description))},
        {"backend", "d3d12"},
        {"robustness", "Dawn default; disable_robustness not enabled"}};
    std::vector<wgpu::FeatureName> features;
    for (auto feature :
         {wgpu::FeatureName::ShaderF16, wgpu::FeatureName::Subgroups,
          wgpu::FeatureName::SubgroupSizeControl,
          wgpu::FeatureName::TimestampQuery})
      if (adapter.HasFeature(feature)) features.push_back(feature);
    Require(adapter.HasFeature(wgpu::FeatureName::ShaderF16),
            "Models require shader-f16");
    wgpu::Limits limits;
    adapter.GetLimits(&limits);
    wgpu::DeviceDescriptor deviceDesc{};
    deviceDesc.requiredFeatures = features.data();
    deviceDesc.requiredFeatureCount = features.size();
    deviceDesc.requiredLimits = &limits;
    deviceDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message,
           NativeGpu* self) {
          self->Fail(std::string(std::string_view(message)));
        },
        this);
    deviceDesc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowProcessEvents,
        [](const wgpu::Device&, wgpu::DeviceLostReason reason,
           wgpu::StringView message, NativeGpu* self) {
          if (reason != wgpu::DeviceLostReason::Destroyed)
            self->Fail(std::string(std::string_view(message)));
        },
        this);
    future = adapter.RequestDevice(
        &deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestDeviceStatus status, wgpu::Device value,
           wgpu::StringView message, NativeGpu* self) {
          if (status == wgpu::RequestDeviceStatus::Success)
            self->device = std::move(value);
          else
            self->Fail(std::string(std::string_view(message)));
        },
        this);
    Require(instance.WaitAny(future, UINT64_MAX) == wgpu::WaitStatus::Success &&
                bool(device),
            "Native device creation failed: " + Failure());
    queue = device.GetQueue();
    evidence["enabledDawnToggles"] = dawn::native::GetTogglesUsed(device.Get());
    for (const auto& toggle : evidence["enabledDawnToggles"])
      Require(toggle != "disable_robustness" && toggle != "skip_validation",
              "Unsafe Dawn toggle is enabled");
    evidence["enabledFeatures"] = json::array();
    for (auto feature : features)
      evidence["enabledFeatures"].push_back(uint32_t(feature));
  }
  void Check() {
    instance.ProcessEvents();
    const auto error = Failure();
    Require(error.empty(), "WebGPU failure: " + error);
  }
  ~NativeGpu() {
    if (device) device.Destroy();
    if (instance) instance.ProcessEvents();
    queue = nullptr;
    device = nullptr;
    adapter = nullptr;
    instance = nullptr;
  }
  std::map<std::string, std::string> Provider() const {
    return {{"deviceId", "1"},
            {"webgpuInstance",
             std::to_string(reinterpret_cast<uintptr_t>(instance.Get()))},
            {"webgpuDevice",
             std::to_string(reinterpret_cast<uintptr_t>(device.Get()))},
            {"dawnProcTable", std::to_string(reinterpret_cast<uintptr_t>(
                                  &dawn::native::GetProcs()))}};
  }
  wgpu::Buffer Buffer(uint64_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc{};
    desc.size = std::max(uint64_t(16), (size + 15) / 16 * 16);
    desc.usage = usage;
    return device.CreateBuffer(&desc);
  }
  static wgpu::Buffer BufferOf(const Tensor& value) {
    return wgpu::Buffer(
        static_cast<WGPUBuffer>(value->GetTensorMutableRawData()));
  }
  void Wait() {
    bool complete = false;
    auto future = queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus status, wgpu::StringView, bool* done) {
          *done = status == wgpu::QueueWorkDoneStatus::Success;
        },
        &complete);
    Require(instance.WaitAny(future, UINT64_MAX) == wgpu::WaitStatus::Success &&
                complete,
            "GPU queue did not complete");
    Check();
  }
};

struct NativeSampler {
  NativeGpu& gpu;
  uint32_t vocabulary, groups;
  wgpu::Buffer partial, result, params, next, logits;
  wgpu::ComputePipeline scan, finish;
  wgpu::BindGroup scanGroup, finishGroup;
  struct Slot {
    wgpu::Buffer buffer;
    wgpu::Future future;
    bool ready = false;
    std::string error;
  };
  std::vector<Slot> slots;
  size_t enqueued = 0, delivered = 0;
  NativeSampler(NativeGpu& gpuValue, uint32_t vocab, bool half,
                const std::set<int>& eos, wgpu::Buffer nextToken, int capacity)
      : gpu(gpuValue),
        vocabulary(vocab),
        groups((vocab + 1023) / 1024),
        next(nextToken) {
    Require(eos.size() <= 8, "At most eight EOS IDs are supported");
    const auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                       wgpu::BufferUsage::CopyDst;
    partial = gpu.Buffer(groups * 16, usage);
    result = gpu.Buffer(16, usage);
    params =
        gpu.Buffer(48, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    uint32_t config[12] = {vocab, 0, groups, uint32_t(eos.size())};
    std::copy(eos.begin(), eos.end(), config + 4);
    gpu.queue.WriteBuffer(params, 0, config, sizeof(config));
    wgpu::ShaderSourceWGSL source{};
    source.code = half ? kGreedyHalf : kGreedyFloat;
    wgpu::ShaderModuleDescriptor descriptor{};
    descriptor.nextInChain = &source;
    auto module = gpu.device.CreateShaderModule(&descriptor);
    wgpu::ComputePipelineDescriptor pipeline{};
    pipeline.compute.module = module;
    pipeline.compute.entryPoint = "scan";
    scan = gpu.device.CreateComputePipeline(&pipeline);
    pipeline.compute.entryPoint = "finish";
    finish = gpu.device.CreateComputePipeline(&pipeline);
    auto bind = [&](uint32_t index, wgpu::Buffer buffer) {
      wgpu::BindGroupEntry e{};
      e.binding = index;
      e.buffer = buffer;
      e.size = buffer.GetSize();
      return e;
    };
    std::vector<wgpu::BindGroupEntry> entries{bind(1, partial), bind(2, params),
                                              bind(3, next), bind(4, result)};
    wgpu::BindGroupDescriptor group{};
    group.layout = finish.GetBindGroupLayout(0);
    group.entryCount = entries.size();
    group.entries = entries.data();
    finishGroup = gpu.device.CreateBindGroup(&group);
    slots.resize(capacity);
    for (auto& slot : slots)
      slot.buffer = gpu.Buffer(
          16, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
    gpu.Check();
  }
  void Bind(wgpu::Buffer values) {
    logits = values;
    wgpu::BindGroupEntry entries[3]{};
    for (int i = 0; i < 3; ++i) {
      entries[i].binding = i;
      entries[i].buffer = i == 0 ? logits : i == 1 ? partial : params;
      entries[i].size = entries[i].buffer.GetSize();
    }
    wgpu::BindGroupDescriptor descriptor{};
    descriptor.layout = scan.GetBindGroupLayout(0);
    descriptor.entryCount = 3;
    descriptor.entries = entries;
    scanGroup = gpu.device.CreateBindGroup(&descriptor);
    gpu.Check();
  }
  bool Full() const { return enqueued - delivered == slots.size(); }
  bool Pending() const { return enqueued != delivered; }
  void Enqueue() {
    Require(!Full(), "Readback pool exhausted");
    auto& slot = slots[enqueued % slots.size()];
    slot.ready = false;
    slot.error.clear();
    auto encoder = gpu.device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(scan);
    pass.SetBindGroup(0, scanGroup);
    pass.DispatchWorkgroups(groups);
    pass.End();
    pass = encoder.BeginComputePass();
    pass.SetPipeline(finish);
    pass.SetBindGroup(0, finishGroup);
    pass.DispatchWorkgroups(1);
    pass.End();
    encoder.CopyBufferToBuffer(result, 0, slot.buffer, 0, 16);
    auto command = encoder.Finish();
    gpu.queue.Submit(1, &command);
    slot.future = slot.buffer.MapAsync(
        wgpu::MapMode::Read, 0, 16, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus status, wgpu::StringView message, Slot* value) {
          value->ready = status == wgpu::MapAsyncStatus::Success;
          if (!value->ready)
            value->error = std::string(std::string_view(message));
        },
        &slot);
    ++enqueued;
  }
  int Deliver() {
    Require(Pending(), "No pending token");
    auto& slot = slots[delivered % slots.size()];
    Require(gpu.instance.WaitAny(slot.future, UINT64_MAX) ==
                    wgpu::WaitStatus::Success &&
                slot.ready,
            "Token readback failed: " + slot.error);
    const auto* words =
        static_cast<const uint32_t*>(slot.buffer.GetConstMappedRange(0, 16));
    const auto token = words[0], error = words[1];
    slot.buffer.Unmap();
    ++delivered;
    gpu.Check();
    Require(error == 0,
            "GPU logits are nonfinite or every token is suppressed");
    Require(token < vocabulary, "GPU token outside vocabulary");
    return int(token);
  }
  ~NativeSampler() {
    while (Pending()) {
      const auto before = delivered;
      try {
        Deliver();
      } catch (...) {
        if (delivered == before) ++delivered;
      }
    }
  }
};
