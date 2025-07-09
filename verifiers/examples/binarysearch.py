from verifiers.envs.binaryserach_env import BinarySearchEnv
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import verifiers as vf
import wandb

"""inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B --gpu-memory-utilization 0.8 --tensor-parallel-size 1

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --config-file configs/zero3.yaml --num-processes 1 binarysearch.py
"""

prompt = """
- Try to guess a number between 0 and {max_item}.
- At each guessing, you will recieve a hint if the number is higer or lower than your guess.
- Your objective is to guess the right number using the less hints as possible.
- Your answer at each step must be between <answer> </answer> tags, and must be an integer number.
- Use tags <think> </think> to think before answering.
- Use tags <think> </think> to think before answering.
""".strip()

def generate_data():
    data = []
    for i in range(1000):
        arr = [100, 500 , 1000]
        max_number = arr[np.random.randint(0, 3)]
        number_chosen = np.random.randint(1, max_number+1)
        
        prompt_atual = prompt.format(max_item=max_number)

        data.append((prompt_atual, number_chosen, max_number))
        
    df = pd.DataFrame(data, columns=["question", "answer", "max_range"])
    dataset = Dataset.from_pandas(df)
    return dataset

train_data = generate_data()
eval_data = generate_data()
dataset = DatasetDict({"train":train_data, "eval":eval_data})



model_name = "Qwen/Qwen2.5-0.5B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = BinarySearchEnv(
    dataset=dataset
)

run_name = f"binary-search-{model_name}-binary-search"
wandb.init(project="rollout_filtering_test", name=run_name)

training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=2
training_args.num_generations=2
training_args.gradient_accumulation_steps=4
# training_args.max_prompt_length=1024
training_args.max_completion_length=4096
training_args.max_steps=100
training_args.mask_env_responses=True
training_args.rollout_filter_ratio=1
training_args.max_prompt_length=None

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

try:
    trainer.train()
except:
    trainer.vllm_client.close_communicator()
