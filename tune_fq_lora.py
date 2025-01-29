from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


# Load and compress a model from Hugging Face.
model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
model = AutoModelForCausalLM.from_pretrained(model_id)
# quantization_config=OVWeightQuantizationConfig(
#     bits=4,
#     dataset="wikitext2",
#     group_size=64,
#     ratio=1.0
# )
model = FQLoRAModel.from_pretrained(model, config)
model.print_trainable_parameters()

trainer = FQLoraTrainer(
    model,
    training_args,
    train_dataset=
    tokenizer=tokenizer,
)

trainer.train()