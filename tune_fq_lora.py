from optimum.intel.openvino import (
    FQLoraModel,
    get_fq_lora_model,
    FQLoraTrainingArguments,
    FQLoraTrainerMVP
)
import torch
from datasets import load_dataset
from transformers.training_args import OptimizerNames
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from optimum.gptq.data import get_dataset, prepare_dataset
from nncf import (
    Dataset,
    CompressWeightsMode,
    BackupMode
)
from typing import Any
from pathlib import Path

# Load and compress a model from Hugging Face.
# model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET = "wikitext2"
CONTEXT_LENGTH=10
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# def tokenize(element):
#     outputs = tokenizer(
#         element["text"],
#         truncation=False,
#         max_length=CONTEXT_LENGTH,
#         return_overflowing_tokens=True,
#         return_length=True,
#         # return_tensors="pt"
#     )
#     # Combine all tokens
#     combined = []
#     for tokenized_doc in outputs['input_ids']:
#         combined += tokenized_doc + [tokenizer.eos_token_id]
#     # Chunk
#     input_batch = []
#     num_tokens = 100
#     # for i in range(0, len(combined) - CONTEXT_LENGTH, CONTEXT_LENGTH):
#     for i in range(0, num_tokens-CONTEXT_LENGTH, CONTEXT_LENGTH):
#         input_batch.append(combined[i:i+CONTEXT_LENGTH])
#     return {"input_ids": input_batch}

# data = load_dataset("wikitext", "wikitext-2-raw-v1")
# tokenized_data = data.map(
#     tokenize, batched=True, remove_columns=data["train"].column_names,
# )
# total_tokens = tokenized_data['train'].num_rows * CONTEXT_LENGTH
# print(f"Training on {total_tokens:_} tokens")
# tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

train_dataset = get_dataset('wikitext2', tokenizer, seqlen=10, nsamples=10)
# train_dataset = tokenized_data["train"]
print(train_dataset[0])
example_input = train_dataset[0]
# print(example_input)
# print(next(iter(td_for_nncf.get_data(indices=[0]))))
# exit()
# if hasattr(data, "input_ids"):
#     data = data.input_ids

fq_lora_dir = Path('fq_lora')
if (fq_lora_dir / FQLoraModel.CKPT_NAME).exists():
    print('#'*50 + ' Loading FQLora model')
    model = FQLoraModel.from_pretrained(fq_lora_dir, model, example_input)
else:
    # quantization_config=OVWeightQuantizationConfig(
    config = dict(
        ratio=1,
        group_size=32,
        mode=CompressWeightsMode.INT4_ASYM,
        backup_mode=BackupMode.NONE,
        # dataset="wikitext2" # or tokenized example input from dataset
    )
    print('#'*50 + ' Creating FQLora model')
    model = get_fq_lora_model(model, config, example_input)
    model.print_trainable_parameters()

    print('#'*50 + ' Saving FQLora model')
    model.save_pretrained(save_directory=fq_lora_dir)


training_args = FQLoraTrainingArguments(
    learning_rate_fq=5e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    save_strategy='epoch',
    output_dir='fq_lora_train',
    report_to='none'
)
training_args = training_args.set_optimizer(
    name=OptimizerNames.ADAMW_TORCH,
    beta1=0.9, beta2=0.999, weight_decay=5e-4, learning_rate=5e-4
)

trainer = FQLoraTrainerMVP(
    model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    # data_collator=data_collator,
)

trainer.train()

trainer.save_model()  # Saves the tokenizer too for easy upload