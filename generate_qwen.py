import gc
from aphrodite import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0,  max_tokens=100, stop=["<fim_prefix>", "<fim_suffix>", "<fim_middle>"])
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B")

# Create an LLM.
from human_eval_infilling.data import write_jsonl, read_problems

def get_prompts(benchmark: str):
    problems = read_problems(benchmark_name=benchmark)
    num_samples_per_task = 1

    samples = [
        dict(task_id=task_id, prompt=f"<fim_prefix>{problems[task_id]['prompt']}<fim_suffix>{problems[task_id]['suffix'].strip()}<fim_middle>")
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]

    task_ids = [sample["task_id"] for sample in samples]
    prompts = [sample["prompt"] for sample in samples]

    return task_ids, prompts

llm = LLM(model="Qwen/CodeQwen1.5-7B", max_model_len=8192)

def trim_first_space(text):
    if len(text) > 1:
        if text[0] == " ":
            return text[1:]
    return text

for benchmark in ["random-span-light", "random-span", "multi-line", "single-line"]:
    task_ids, prompts = get_prompts(benchmark)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    write_jsonl(f"samples_qwencode_{benchmark}.jsonl", [{"task_id": task_id, "completion": trim_first_space(output.outputs[0].text) } for task_id, output in zip(task_ids, outputs)])
        

del llm
gc.collect()
torch.cuda.empty_cache()
