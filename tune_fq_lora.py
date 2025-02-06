from optimum.intel.openvino import (
    FQLoraModel,
    get_fq_lora_model,
    FQLoraTrainingArguments,
    FQLoraTrainerMVP
)
from datasets import load_dataset
from transformers.training_args import OptimizerNames
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
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
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
train_dataset = get_dataset('wikitext2', tokenizer, seqlen=10, nsamples=10)
# print(train_dataset[0])
example_input = Dataset([train_dataset[0]])
# print(next(iter(example_input.get_data(indices=[0]))))
# if hasattr(data, "input_ids"):
#     data = data.input_ids

fq_lora_dir = Path('fq_lora')
if (fq_lora_dir / FQLoraModel.CKPT_NAME).exists():
    print('#'*50 + ' Loading FQLora model')
    model = FQLoraModel.from_pretrained(fq_lora_dir, model, train_dataset[0])
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
    per_gpu_train_batch_size=32,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    save_strategy='epoch',
    output_dir='fq_lora_train'
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
)

trainer.train()

trainer.save_model()  # Saves the tokenizer too for easy upload