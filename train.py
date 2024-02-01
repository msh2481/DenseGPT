from typing import Mapping, Literal

import torch as t
import torch.nn.functional as F
import yaml  # type: ignore
from beartype import beartype as typed
from datasets import load_dataset  # type: ignore
from datasets import Dataset
from dvclive.huggingface import DVCLiveCallback  # type: ignore
from jaxtyping import Float, Int
from torch import Tensor as TT
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import DPOTrainer  # type: ignore

from gpt import DenseGPTConfig, DenseGPTForCausalLM

with open("params.yaml") as f:
    params = yaml.safe_load(f)
print(params)
model_name = params["model_name"]
dataset_name = params["dataset_name"]
tokenizer_name = params["tokenizer_name"]
use_dense = params["use_dense"]
lr = params["lr"]
batch_size = params["batch_size"]
dataset_size = params["dataset_size"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

config = DenseGPTConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    num_layers=8,
    attention_types=[[["global"], 8]],
    num_heads=16,
    use_dense=use_dense,
)
if not model_name:
    model = DenseGPTForCausalLM(config)
else:
    model = DenseGPTForCausalLM.from_pretrained(model_name, config=config)

dataset = load_dataset(dataset_name)

if "TinyStories" in dataset_name:

    @typed
    def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
        result = tokenizer(
            example["text"], max_length=256, padding="max_length", truncation=True
        )
        result["labels"] = result["input_ids"]
        return result

else:

    @typed
    def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
        result = tokenizer(example["text"])
        result["labels"] = result["input_ids"]
        return result


subset = dataset["train"].select(range(dataset_size)).to_iterable_dataset()
tokenized = subset.map(tokenize_function, batched=True).remove_columns(["text"])

training_args = TrainingArguments(
    output_dir="trainer",
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    logging_steps=50,
    num_train_epochs=1,
    max_steps=dataset_size // batch_size,
    save_total_limit=1,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.add_callback(DVCLiveCallback())
trainer.train()

"""
train_dataset = Dataset.from_dict({
    "prompt": [
        "hello",
        "how are you",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
    ],
})
"""


def dpo(
    prompt_chosen_rejected: dict[str, list[str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int,
    lr: float,
    beta: float,
    max_steps: int,
    loss_type: Literal["sigmoid", "ipo"] = "ipo",
):
    train_dataset = Dataset.from_dict(prompt_chosen_rejected)
    max_steps = 100

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=100,
        remove_unused_columns=False,
        learning_rate=lr,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        output_dir="trainer",
        optim="adamw_torch",
        warmup_steps=15,
        report_to="none",
        save_total_limit=1,
        # bf16=True,
    )

    dpo_trainer = DPOTrainer(
        model,
        # ref_model,
        args=training_args,
        beta=beta,
        loss_type=loss_type,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=128,
        max_target_length=128,
        max_prompt_length=128,
    )

    dpo_trainer.train()


"""
Plan:
Take TinyStories texts as positive examples, model generated texts as negative (pairing them arbitrarily) and set prompts to be empty, or some fixed string.

For several epochs:
    Run DPO
    Updated some (e.g. each with probability 0.1) of the generated texts with new ones
    Observe what happens to average loss on dataset samples, on self-generated samples and on corrupted dataset samples (which should indicate how stable the model is)
"""
