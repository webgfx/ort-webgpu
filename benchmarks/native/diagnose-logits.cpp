// Diagnostic only: inspect the first decode logits outside any performance run.
#define main benchmark_main
#include "main.cpp"
#undef main

int main(int argc,char** argv){
  if(argc!=3)return 2;
  json result;
  try{
    auto config=json::parse(std::ifstream(argv[1]));config["options"]["gpuSampling"]="cpu";
    Generator generator(config["model"],config["options"]);
    const auto prompt=config["model"]["cases"][0]["prompt"].get<std::vector<int64_t>>();
    generator.Reset();generator.context->Wait();
    const int first=generator.Prefill(prompt);
    generator.DecodeInputs(first);generator.sessions["decode"]->Run(generator.run,*generator.decoderBinding);generator.CopyState();
    auto tensor=generator.decodeOutputs.at(generator.model["decoder"]["outputs"]["logits"]);
    const auto info=tensor->GetTensorTypeAndShapeInfo();
    const size_t size=(info.GetElementCount()*Bytes(info.GetElementType())+3)/4*4;
    auto& gpu=*generator.context;
    auto readback=gpu.Buffer(size,wgpu::BufferUsage::CopyDst|wgpu::BufferUsage::MapRead);
    auto encoder=gpu.device.CreateCommandEncoder();encoder.CopyBufferToBuffer(NativeGpu::BufferOf(tensor),0,readback,0,size);auto command=encoder.Finish();gpu.queue.Submit(1,&command);
    bool ready=false;
    auto future=readback.MapAsync(wgpu::MapMode::Read,0,size,wgpu::CallbackMode::WaitAnyOnly,[](wgpu::MapAsyncStatus status,wgpu::StringView,bool* done){*done=status==wgpu::MapAsyncStatus::Success;},&ready);
    Require(gpu.instance.WaitAny(future,UINT64_MAX)==wgpu::WaitStatus::Success && ready,"Readback failed");
    const auto* data=readback.GetConstMappedRange(0,size);std::vector<std::pair<float,int>> scores;
    for(int i=0;i<generator.vocabulary;++i){
      const auto value=info.GetElementType()==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16?Ort::Float16_t::FromBits(static_cast<const uint16_t*>(data)[i]).ToFloat():static_cast<const float*>(data)[i];
      if(!generator.eos.contains(i))scores.push_back({value,i});
    }
    readback.Unmap();gpu.Check();
    std::partial_sort(scores.begin(),scores.begin()+16,scores.end(),[](auto a,auto b){return a.first==b.first?a.second<b.second:a.first>b.first;});
    json top=json::array();for(size_t i=0;i<16;++i)top.push_back({{"token",scores[i].second},{"logit",scores[i].first}});
    result={{"success",true},{"firstToken",first},{"topSecondToken",top},{"capture",generator.capture}};
  }catch(const std::exception& error){result={{"success",false},{"error",error.what()}};}
  std::ofstream(argv[2])<<result.dump(2);return result["success"].get<bool>()?0:1;
}
