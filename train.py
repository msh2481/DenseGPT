import json
from typing import Literal, Mapping

import torch as t
import torch.nn.functional as F
import yaml  # type: ignore
from beartype import beartype as typed
from datasets import load_dataset  # type: ignore
from datasets import Dataset
from jaxtyping import Float, Int
from torch import Tensor as TT
from tqdm.auto import tqdm
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import DPOTrainer  # type: ignore

from utils import tokenize

with open("params.yaml") as f:
    params = yaml.safe_load(f)
print(params)
model_name = params["model_name"]
dataset_name = params["dataset_name"]
tokenizer_name = params["tokenizer_name"]
lr = params["lr"]
batch_size = params["batch_size"]
dataset_size = params["dataset_size"]
beta = params["beta"]
max_steps = params["max_steps"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"

device = "cuda" if t.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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


subset = dataset["train"].select(range(dataset_size))

dataloader = t.utils.data.DataLoader(subset, batch_size=batch_size)


# %%
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

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=100,
        remove_unused_columns=False,
        learning_rate=lr,
        logging_first_step=True,
        output_dir="trainer",
        optim="adamw_torch",
        warmup_steps=2,
        report_to="none",
        save_total_limit=1,
        # bf16=True,
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_args,
        beta=beta,
        loss_type=loss_type,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()


for it, batch in enumerate(dataloader):
    prompts = [s.split()[0] for s in batch["text"]]
    tokenized_prompts = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4
    ).to(device)
    tokenized_generations = model.generate(
        **tokenized_prompts,
        do_sample=True,
        max_new_tokens=128,
    )
    generations = tokenizer.batch_decode(
        tokenized_generations, skip_special_tokens=True
    )
    chosen = [s[len(p) :] for s, p in zip(batch["text"], prompts)]
    rejected = [s[len(p) :] for s, p in zip(generations, prompts)]
    current_dataset = {
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
    }

    with open("log.txt", "w") as f:
        json.dump(current_dataset, f, indent=2)

    dpo(
        current_dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        max_steps=max_steps,
        loss_type="sigmoid",
    )

    if it % 10 == 0:
        model.save_pretrained("model")
