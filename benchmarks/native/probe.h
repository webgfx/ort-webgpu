#pragma once
// Execute the same WGSL used by Web and native, including buffer reuse.
inline json ProbeSampler(const json& input) {
  NativeGpu context;
  const bool half = input.at("type") == "float16";
  const auto size = input.at("cases")[0].at("bits").size();
  auto logits =
      context.Buffer(size * (half ? 2 : 4),
                     wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  auto next = context.Buffer(
      16, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  NativeSampler sampler(context, uint32_t(size), half, {0}, next, 2);
  sampler.Bind(logits);
  int checked = 0;
  for (const auto& item : input.at("cases")) {
    Require(item["bits"].size() == size, "Probe shape mismatch");
    std::vector<uint8_t> bytes((size * (half ? 2 : 4) + 3) / 4 * 4, 0);
    for (size_t i = 0; i < size; ++i) {
      const auto value = item["bits"][i].get<uint32_t>();
      std::memcpy(bytes.data() + i * (half ? 2 : 4), &value, half ? 2 : 4);
    }
    bool cpuRejected = false;
    int cpuToken = -1;
    try {
      cpuToken = GreedyBits(bytes.data(), half, int(size), {0});
    } catch (const std::exception& error) {
      if (std::string(error.what()).find("Nonfinite") == std::string::npos)
        throw;
      cpuRejected = true;
    }
    Require(cpuRejected == (item["valid"].get<int>() == 0),
            "CPU finite validation mismatch");
    if (!cpuRejected)
      Require(cpuToken == item["token"].get<int>(), "CPU token mismatch");
    context.queue.WriteBuffer(logits, 0, bytes.data(), bytes.size());
    for (int repeat = 0; repeat < 3; ++repeat) {
      sampler.Enqueue();
      bool rejected = false;
      int token = -1;
      try {
        token = sampler.Deliver();
      } catch (const std::exception& error) {
        if (std::string(error.what()).find("nonfinite") == std::string::npos)
          throw;
        rejected = true;
      }
      Require(
          rejected == (item["valid"].get<int>() == 0),
          "GPU finite validation mismatch: " + item["name"].get<std::string>());
      if (!rejected)
        Require(token == item["token"].get<int>(),
                "GPU token mismatch: " + item["name"].get<std::string>());
      ++checked;
    }
  }
  return {
      {"success", true}, {"checked", checked}, {"device", context.evidence}};
}
