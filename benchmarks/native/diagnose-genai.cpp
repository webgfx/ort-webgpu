#include "ort_genai.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <vector>
using json=nlohmann::json;
int main(int argc,char** argv){
 if(argc!=3)return 2;json result;
 try{
  Oga::SetTelemetryEnabled(false);OgaHandle handle;
  const auto input=json::parse(std::ifstream(argv[1]));
  auto model=OgaModel::Create(input["modelPath"].get<std::string>().c_str());
  auto params=OgaGeneratorParams::Create(*model);params->SetSearchOption("max_length",8192);params->SetSearchOption("min_length",1152);
  params->SetSearchOptionBool("do_sample",false);params->SetSearchOption("repetition_penalty",1);params->SetSearchOption("num_beams",1);
  const auto prompt=input["cases"][0]["prompt"].get<std::vector<int32_t>>();
  auto generator=OgaGenerator::Create(*model,*params);generator->AppendTokens(prompt.data(),prompt.size());generator->GenerateNextToken();
  const auto first=generator->GetSequenceData(0)[generator->GetSequenceCount(0)-1];
  auto logits=generator->GetLogits();if(logits->Type()!=OgaElementType_float32)throw std::runtime_error("Expected float32 logits");
  const auto vocabulary=logits->Shape().back();const auto* values=static_cast<const float*>(logits->Data());std::vector<std::pair<float,int>> scores;
  for(int i=0;i<vocabulary;++i)scores.push_back({values[i],i});
  std::partial_sort(scores.begin(),scores.begin()+16,scores.end(),[](auto a,auto b){return a.first==b.first?a.second<b.second:a.first>b.first;});
  json top=json::array();for(size_t i=0;i<16;++i)top.push_back({{"token",scores[i].second},{"logit",scores[i].first}});
  result={{"success",true},{"firstToken",first},{"topSecondToken",top}};
 }catch(const std::exception& error){result={{"success",false},{"error",error.what()}};}
 std::ofstream(argv[2])<<result.dump(2);return result["success"].get<bool>()?0:1;
}
