
"""inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2-7B-Instruct --gpu-memory-utilization 0.8 --tensor-parallel-size 1

training:
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config-file configs/zero3.yaml  symbolic_regression_task/symbolic_search.py
"""
from symbolic_regression_task.symbolic_search_env import SymbolicSearchEnv, get_next_iteration
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import verifiers as vf
import wandb
import accelerate 
from sklearn.utils import shuffle 
from verifiers.parsers import XMLParser

# np.random.seed(42)

df = pd.read_csv('symbolic_regression_task/data.csv')

df['question'] = df['prompts']

df_train = df.iloc[:300]
# df_train = shuffle(df_train)


df_test = df.iloc[700:]



dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)


dataset = DatasetDict({"train":dataset_train, 'eval':dataset_test})


model_name = "Qwen/Qwen2-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = SymbolicSearchEnv(
    dataset=dataset
)

run_name = f"binary-search-{model_name}-1-filter"

def get_next_iteration(state, prompt, completion):
    llm_parser = XMLParser(fields=["think", "answer"])
    answer = state['answer']
    base = prompt[-1]['content']
    if 'last tries' not in base:
        base = base + '\n\nYour last tries and the feedback:\n'
    base = base + 'Try:\n' + completion[-1]["content"]
    try:
        parsed = llm_parser.parse(completion[-1]["content"])
        # Check if we got a valid code field (not just None from failed parsing)
        answer = state['answer']
        if hasattr(parsed, 'answer') and parsed.answer is not None:
            if str(parsed.answer).isdigit() and parsed.answer is not None:
                if int(parsed.answer) == answer:
                    return {"role": "user", "content": base + "\nCongratulations!! Right answer!"}, state
                elif int(parsed.answer) > answer:
                    return {"role": "user", "content": base + f"\nn is lower than {str(parsed.answer)}"}, state
                elif int(parsed.answer) < answer:
                    return {"role": "user", "content": base + f"\nn is higher than {str(parsed.answer)}"}, state
                else:
                    return {"role": "user", "content": base + "\nError: invalid answer, you must pass an integer number only inside the tags <answer> </answer>."}, state
            else:
                return {"role": "user", "content": base + "\nError: answer is not an integer, you must pass an integer number only inside the tags <answer> </answer>."}, state
    except Exception:
        pass
    return {"role": "user", "content": base + '\nYour last try was Invalid'}, state


training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=2
training_args.num_generations=2
training_args.gradient_accumulation_steps=4
# training_args.max_prompt_length=1024
training_args.max_completion_length=512
training_args.max_steps=100
training_args.mask_env_responses=True
training_args.rollout_filter_ratio=1
training_args.max_prompt_length=None
training_args.async_generation_timeout = 1200
training_args.beta = 0.0
training_args.num_refinement_turns = 0

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env_response=get_next_iteration,
    env=vf_env,
    args=training_args,
)

if trainer.accelerator.is_main_process:
    wandb.init(project="rollout_filtering_test", name=run_name)

try:
    trainer.train()
except:
    trainer.vllm_client.close_communicator()
