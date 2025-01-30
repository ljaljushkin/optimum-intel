from optimum.intel.openvino import (
    FQLoRAModel,
    # FQLoraTrainer
)
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from optimum.gptq.data import get_dataset, prepare_dataset
from nncf import (
    Dataset,
    compress_weights,
    CompressWeightsMode,
    BackupMode
)
# Load and compress a model from Hugging Face.
model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
train_dataset = get_dataset('wikitext2', tokenizer, seqlen=10, nsamples=10)
# print(train_dataset[0])
example_input = Dataset([train_dataset[0]])
# print(next(iter(example_input.get_data(indices=[0]))))
# exit()

# quantization_config=OVWeightQuantizationConfig(
config = dict(
    ratio=1,
    group_size=1,
    mode=CompressWeightsMode.INT4_ASYM,
    backup_mode=BackupMode.NONE,
    # dataset="wikitext2" # or tokenized example input from dataset
)
# TODO: 2 types of loading:
#   from model id/path -> NNCF checkpoint
#       from peft import AutoPeftModel
#       model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")
#   from PeftConfig -> NNCFConfig + example inputs
#       from peft import get_peft_model
#       model = get_peft_model(model, peft_config)
def get_fq_lora_model(model_, config_, example_input_):
    compress_weights(
            model_,
            dataset=example_input_,
            **config_,
        )
    model_.nncf.get_graph().visualize_graph("fq_model.dot")
    for param in model.parameters():
        param.requires_grad = False

    for quantizer in model_._nncf.external_quantizers.values():
        quantizer.enable_gradients()

    return FQLoRAModel(model_)

print('#'*50 + ' Creating FQLora model')
model = get_fq_lora_model(model, config, example_input)
model.print_trainable_parameters()

# def save_pretrained(
#     self,
#     save_directory: str,
#     safe_serialization: bool = True,
#     selected_adapters: Optional[list[str]] = None,
#     save_embedding_layers: Union[str, bool] = "auto",
#     is_main_process: bool = True,
#     path_initial_model_for_weight_conversion: Optional[str] = None,
#     **kwargs: Any,
# ) -> None:

fq_lora_dir = 'fq_lora'
print('#'*50 + ' Saving FQLora model')
FQLoRAModel.save_pretrained(model, save_directory=fq_lora_dir)
# TODO: how to load checkpoint via trainer?? need to pass example_input

print('#'*50 + ' Loading FQLora model')
model = AutoModelForCausalLM.from_pretrained(model_id)
FQLoRAModel.from_pretrained(fq_lora_dir, model, train_dataset[0])

# # calibration_dataset = prepare_dataset(calibration_dataset)
# # if hasattr(data, "input_ids"):
# #     data = data.input_ids

# TODO: use ordinary trainer to test, take simple train pipeline.
# trainer = FQLoraTrainer(
#     model,
#     training_args,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
# )

# trainer.train()