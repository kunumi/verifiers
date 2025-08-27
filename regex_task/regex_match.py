
"""inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.8 --tensor-parallel-size 1

training:
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config-file configs/zero3.yaml  regex_task/regex_match.py
"""
from regex_task.regex_match_env import RegexMatchEnv, get_next_step
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import verifiers as vf
import wandb
import accelerate 
from sklearn.utils import shuffle 
from verifiers.parsers import XMLParser

# np.random.seed(42)

df = pd.read_csv('regex_task/data.csv')

df['question'] = df['prompts']

df_train = df.iloc[:336]
# df_train = shuffle(df_train)


df_test = df.iloc[700:]



dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)


dataset = DatasetDict({"train":dataset_train, 'eval':dataset_test})


model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = RegexMatchEnv(
    dataset=dataset
)

run_name = f"regex_match_run"

training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=2
training_args.num_generations=8
training_args.gradient_accumulation_steps=4
# training_args.max_prompt_length=1024
training_args.max_completion_length=1024
training_args.max_grad_norm=0.001
training_args.max_steps=300 # always give this value
training_args.mask_env_responses=True
training_args.rollout_filter_ratio=0.5
training_args.max_prompt_length=None
training_args.async_generation_timeout = 1200
# training_args.beta = 0.0
training_args.sync_ref_model = True
training_args.ref_model_sync_step = 20
training_args.num_refinement_turns = 2
training_args.loss_type = 'dr_grpo'

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env_response=get_next_step,
    env=vf_env,
    args=training_args,
)

if trainer.accelerator.is_main_process:
    wandb.init(project="regex_task", name=run_name)

try:
    trainer.train()
except:
    trainer.vllm_client.close_communicator()
