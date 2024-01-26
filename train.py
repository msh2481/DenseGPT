from typing import Mapping

import yaml  # type: ignore
from beartype import beartype as typed
from datasets import load_dataset  # type: ignore
from dvclive.huggingface import DVCLiveCallback  # type: ignore
from transformers import AutoTokenizer, Trainer, TrainingArguments  # type: ignore

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
