#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pathlib import Path
from nncf.torch import load_from_config
from nncf import compress_weights
import torch
from transformers import PreTrainedModel
from nncf import compress_weights


# TODO: should it be optimized optimum model??
class FQLoraModel(torch.nn.Module):
    CKPT_NAME = 'nncf_ckpt.pth'

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self._base_model = model

    # TODO: OptimizedModel in optimum has _save_pretrained and _from_pretrained. Is better??
    def save_pretrained(self, save_directory):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert hasattr(self._base_model, 'nncf')
        nncf_state_dict = self._base_model.nncf.state_dict()
        nncf_config = self._base_model.nncf.get_config()
        torch.save(
            {
                "nncf_state_dict": nncf_state_dict,
                "nncf_config": nncf_config,
            },
            save_dir / self.CKPT_NAME,
        )

    def forward(self, *args, **kwargs):
        return self._base_model(*args, **kwargs)
    # config = PeftConfig.from_pretrained(peft_model_id)
    # model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    # model = PeftModel.from_pretrained(model, peft_model_id)

    # TODO: how to implement that?? can pass with kwargs??
    # TODO: is it OK to be incompatible with other Trainer, only with FQLoraTrainer??
    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    #     *model_args,
    #     config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    #     cache_dir: Optional[Union[str, os.PathLike]] = None,
    #     ignore_mismatched_sizes: bool = False,
    #     force_download: bool = False,
    #     local_files_only: bool = False,
    #     token: Optional[Union[str, bool]] = None,
    #     revision: str = "main",
    #     use_safetensors: bool = None,
    #     **kwargs,
    # ):
    @classmethod
    def from_pretrained(cls, save_directory, model, example_input):
        nncf_ckpt = torch.load(Path(save_directory) / cls.CKPT_NAME)
        nncf_model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=example_input)
        nncf_model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
        return cls(nncf_model)

    def convert_to_ov(self, save_directory):
        # https://github.com/daniil-lyakhov/nncf/commit/9d66219f2036a78ddcae7ffcbec735b90310b053#diff-fc9a2a5b10bfcb3a766468829dc10e1397236c523576fe72dde8c42f381fde0f
        from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model
        stripped_model = strip_tuned_lora_model(self)
        from optimum.exporters.openvino.convert import export_from_model
        # Call ov.convert_model()
        export_from_model(stripped_model, save_directory, stateful=True, compression_option="fp32")
        pass

    def temporary_original_inference(self):
        class Mgr:
            def __init__(self, model: FQLoraModel):
                self.model = model

            def __enter__(self):
                for quantizer in self.model._base_model._nncf.external_quantizers.values():
                    quantizer.disable_quantization()
                return self.model

            def __exit__(self):
                for quantizer in self.model._base_model._nncf.external_quantizers.values():
                    quantizer.enable_quantization()

        return Mgr(self)

    def get_nb_trainable_parameters(self):
        """
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        # note: same as PeftModel.get_nb_trainable_parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )



# TODO: 2 types of loading:
#   from model id/path -> NNCF checkpoint
#       from peft import AutoPeftModel
#       model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")
#   from PeftConfig -> NNCFConfig + example inputs
#       from peft import get_peft_model
#       model = get_peft_model(model, peft_config)
# export to OV
def get_fq_lora_model(model, config, example_input):
    compress_weights(
            model,
            dataset=example_input,
            **config,
        )
    model.nncf.get_graph().visualize_graph("fq_model.dot")

    for param in model.parameters():
        param.requires_grad = False
    for quantizer in model._nncf.external_quantizers.values():
        quantizer.enable_gradients()

    return FQLoraModel(model)
