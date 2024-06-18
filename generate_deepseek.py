import gc
from vllm import LLM, SamplingParams
import torch

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0,  max_tokens=100, stop=["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"])

# Create an LLM.
from human_eval_infilling.data import write_jsonl, read_problems

def get_prompts(benchmark: str):
  problems = read_problems(benchmark_name=benchmark)
  num_samples_per_task = 1
  samples = [
      dict(task_id=task_id, prompt=f"<｜fim▁begin｜>{problems[task_id]['prompt']}<｜fim▁hole｜>{problems[task_id]['suffix'].strip()}<｜fim▁end｜>")
      for task_id in problems
      for _ in range(num_samples_per_task)
  ]

  task_ids = [sample["task_id"] for sample in samples]
  prompts = [sample["prompt"] for sample in samples]

  return task_ids, prompts

llm = LLM(model="deepseek-ai/DeepSeek-Coder-V2-Lite-Base", max_model_len=8192, trust_remote_code=True)

for benchmark in ["random-span-light", "random-span", "multi-line", "single-line"]:
  task_ids, prompts = get_prompts(benchmark)
  outputs = llm.generate(prompts, sampling_params=sampling_params)

  write_jsonl(f"samples_deepseek_{benchmark}.jsonl", [{"task_id": task_id, "completion": output.outputs[0].text} for task_id, output in zip(task_ids, outputs)])

del llm
gc.collect()
torch.cuda.empty_cache()
