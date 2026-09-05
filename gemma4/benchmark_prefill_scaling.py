"""Standalone Gemma 4 prefill-scaling benchmark.

Runs normal 128-token generation at fixed prompt-size targets while measuring
time to the first emitted token. Each invocation tests one
stack so GPU allocations are released between sessions.
"""
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

TARGETS = [128, 512, 1024, 2048, 4096]
WARMUPS_PER_LENGTH = 1
SAMPLES_PER_LENGTH = 3
OUTPUT_TOKENS = 128
SEED = "Gemma is an efficient language model designed for local inference and useful assistant tasks. "
ORT_MODEL = "E:/workspace/models/gemma4-webgpu-int4"
LITERT_MODEL = "E:/workspace/models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm"
GGUF_MODEL = "E:/workspace/models/gemma4-e2b-gguf/google_gemma-4-E2B-it-Q4_K_M.gguf"
LLAMA_SERVER = "E:/workspace/llama.cpp-bin/vulkan/llama-server.exe"

def ort_prompt(tokenizer, target):
    from benchmark_ort_gemma4 import format_prompt
    text = SEED * (target // 8 + 20)
    while len(tokenizer.encode(format_prompt(text))) > target:
        text = text[:-1]
    while len(tokenizer.encode(format_prompt(text))) < target:
        text += " x"
        if len(tokenizer.encode(format_prompt(text))) > target: text = text[:-2] + "."
    return text, len(tokenizer.encode(format_prompt(text)))

def result(target, input_tokens, samples):
    return {"target_tokens": target, "input_tokens": input_tokens,
            "samples_ms": [round(x, 2) for x in samples],
            "median_ms": round(statistics.median(samples), 2),
            "min_ms": round(min(samples), 2), "max_ms": round(max(samples), 2)}

def run_ort():
    import onnxruntime_genai as og
    from benchmark_ort_gemma4 import format_prompt, generate
    model=og.Model(ORT_MODEL); tok=og.Tokenizer(model); generate(model,tok,format_prompt("warmup"),max_tokens=OUTPUT_TOKENS,greedy=True)
    out=[]
    for target in TARGETS:
        prompt,count=ort_prompt(tok,target)
        for _ in range(WARMUPS_PER_LENGTH): generate(model,tok,format_prompt(prompt),max_tokens=OUTPUT_TOKENS,greedy=True)
        samples=[generate(model,tok,format_prompt(prompt),max_tokens=OUTPUT_TOKENS,greedy=True)["ttft_s"]*1000 for _ in range(SAMPLES_PER_LENGTH)]
        out.append(result(target,count,samples)); print(target,out[-1])
    return out,{"stack":"INT4 ONNX + ONNX Runtime GenAI","backend":"WebGPU EP","graph_capture":True,"mtp":False}

def run_litert(mtp):
    import litert_lm
    from benchmark_litert_gemma4 import generate_streaming
    kw={"enable_speculative_decoding":True} if mtp else {}
    engine=litert_lm.Engine(LITERT_MODEL,backend=litert_lm.Backend.GPU(),max_num_tokens=8192,**kw)
    generate_streaming(engine,"warmup",max_tokens=OUTPUT_TOKENS,greedy=True); out=[]
    # Use ORT tokenizer only to construct the shared fixed-size payloads.
    import onnxruntime_genai as og
    om=og.Model(ORT_MODEL); ot=og.Tokenizer(om)
    for target in TARGETS:
        prompt,_=ort_prompt(ot,target)
        for _ in range(WARMUPS_PER_LENGTH): r=generate_streaming(engine,prompt,max_tokens=OUTPUT_TOKENS,greedy=True)
        samples=[]
        for _ in range(SAMPLES_PER_LENGTH): r=generate_streaming(engine,prompt,max_tokens=OUTPUT_TOKENS,greedy=True); samples.append(r["ttft_s"]*1000)
        out.append(result(target,r["input_tokens"],samples)); print(target,out[-1])
    return out,{"stack":f"INT4 LITERTLM + LiteRT-LM, MTP {'on' if mtp else 'off'}","backend":"WebGPU / D3D12","mtp":mtp}

def run_llama():
    from benchmark_llamacpp_gemma4 import start_llama_server, stop_llama_server
    import httpx
    import onnxruntime_genai as og
    start_llama_server(LLAMA_SERVER,GGUF_MODEL,port=8080,main_gpu=0,cache_prompt=False)
    try:
        def streamed_ttft(prompt):
            payload={"messages":[{"role":"user","content":prompt}],"max_tokens":OUTPUT_TOKENS,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
            start=time.perf_counter(); first=None; usage={}
            with httpx.Client(timeout=300.0) as client:
                with client.stream("POST","http://127.0.0.1:8080/v1/chat/completions",json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: " ) or line == "data: [DONE]": continue
                        event=json.loads(line[6:]); usage=event.get("usage") or usage
                        choices=event.get("choices") or []
                        if first is None and choices and choices[0].get("delta",{}).get("content"):
                            first=(time.perf_counter()-start)*1000
            if first is None: raise RuntimeError("llama.cpp stream emitted no content token")
            return first,usage
        streamed_ttft("warmup")
        om=og.Model(ORT_MODEL); ot=og.Tokenizer(om); out=[]
        for target in TARGETS:
            prompt,_=ort_prompt(ot,target)
            for _ in range(WARMUPS_PER_LENGTH): streamed_ttft(prompt)
            measured=[streamed_ttft(prompt) for _ in range(SAMPLES_PER_LENGTH)]; samples=[x[0] for x in measured]
            out.append(result(target,measured[-1][1].get("prompt_tokens",target),samples)); print(target,out[-1])
        return out,{"stack":"Q4_K_M GGUF + llama.cpp","backend":"Vulkan","mtp":False}
    finally: stop_llama_server()

def main():
    p=argparse.ArgumentParser(); p.add_argument("stack",choices=["ort","litert-off","litert-on","llama"]); p.add_argument("--output",required=True); a=p.parse_args()
    result,meta = run_ort() if a.stack=="ort" else run_litert(a.stack=="litert-on") if a.stack.startswith("litert") else run_llama()
    boundary = "request submission to first streamed content token"
    Path(a.output).write_text(json.dumps({"meta":meta|{"timestamp":datetime.now(timezone.utc).isoformat(),"method":"per-length warmup then median TTFT from normal 128-token generations","requested_output_tokens":OUTPUT_TOKENS,"warmups_per_length":WARMUPS_PER_LENGTH,"samples_per_length":SAMPLES_PER_LENGTH,"timing_boundary":boundary,"model_loading_included":False},"prefill_scaling":result},indent=2),encoding="utf-8")
if __name__=="__main__": main()
