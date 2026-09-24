// Matched GenAI harness: exact saved prompt IDs and CPU-visible timing.
// This wraps the public GenAI API; it does not change GenAI's implementation.
#include "ort_genai.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b-a).count(); }
int main(int argc, char** argv) {
  if(argc!=3){std::cerr<<"genai_reference config.json result.json\n";return 2;}
  json result={{"success",false},{"rows",json::array()},{"errors",json::array()}};
  try {
    Oga::SetTelemetryEnabled(false);
    OgaHandle handle;
    const auto input=json::parse(std::ifstream(argv[1]));
    auto model=OgaModel::Create(input.at("modelPath").get<std::string>().c_str());
    auto params=OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length",8192);params->SetSearchOptionBool("do_sample",false);
    params->SetSearchOption("repetition_penalty",1);params->SetSearchOption("num_beams",1);
    const int length=input.at("generationLength"),repetitions=input.at("repetitions");
    if(length<2 || repetitions<1)throw std::runtime_error("Invalid workload");
    for(const auto& item:input.at("cases")){
      auto prompt=item.at("prompt").get<std::vector<int32_t>>();
      if(prompt.empty() || prompt.size()+length>8192)throw std::runtime_error("Invalid cache capacity");
      params->SetSearchOption("min_length",double(prompt.size()+length));
      auto generator=OgaGenerator::Create(*model,*params);
      json row={{"pl",prompt.size()},{"prompt",prompt},{"tg",length},{"samples",json::array()}};
      for(int repetition=-1;repetition<repetitions;++repetition){
        generator->RewindTo(0);
        auto start=Clock::now();generator->AppendTokens(prompt.data(),prompt.size());auto appended=Clock::now();
        std::vector<int32_t> generated;std::vector<double> delivery;
        // GetSequenceData makes each selected token explicitly CPU-visible.
        generator->GenerateNextToken();bool done=generator->IsDone();
        auto count=generator->GetSequenceCount(0);auto sequence=generator->GetSequenceData(0);
        generated.push_back(sequence[count-1]);auto first=Clock::now();delivery.push_back(ms(start,first));
        for(int index=1;index<length && !done;++index){
          generator->GenerateNextToken();done=generator->IsDone();
          count=generator->GetSequenceCount(0);sequence=generator->GetSequenceData(0);
          generated.push_back(sequence[count-1]);delivery.push_back(ms(start,Clock::now()));
        }
        auto end=Clock::now();
        if(generated.size()!=size_t(length) || count!=prompt.size()+length)throw std::runtime_error("Incomplete GenAI output");
        json sample={{"generated",generated},{"tokenDeliveryMs",delivery},{"ttftMs",ms(start,first)},
          {"e2eMs",ms(start,end)},{"prefillTps",prompt.size()*1000./ms(start,first)},
          {"decodeTps",(length-1)*1000./ms(first,end)},{"appendOnlyMs",ms(start,appended)}};
        if(repetition==-1)row["warmup"]=sample;
        else {if(sample["generated"]!=row["warmup"]["generated"])throw std::runtime_error("GenAI output changed after rewind");row["samples"].push_back(sample);}
        std::cout<<"input="<<prompt.size()<<" rep="<<repetition<<" decode="<<sample["decodeTps"]<<std::endl;
      }
      result["rows"].push_back(row);
    }
    result["executionEvidence"]={{"samplingDevice","cpu"},{"reuseGenerator",true},{"tokenDelivery","explicit-GetSequenceData-each-step"}};
    result["success"]=true;
  }catch(const std::exception& error){result["errors"].push_back(error.what());std::cerr<<error.what()<<'\n';}
  std::ofstream(argv[2])<<result.dump(2)<<'\n';return result["success"].get<bool>()?0:1;
}
