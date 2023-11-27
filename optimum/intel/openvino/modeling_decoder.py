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

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union, Dict, List, Tuple, Callable, Iterable, Any

import numpy as np
import openvino
import torch
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Tensor, Type
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers.generation.utils import GenerateOutput


from optimum.utils import NormalizedConfigManager

from ...exporters.openvino import main_export
from ..utils.import_utils import is_transformers_version
from ..utils.modeling_utils import MULTI_QUERY_ATTN_MODELS
from .modeling import _TOKENIZER_FOR_DOC, INPUTS_DOCSTRING, MODEL_START_DOCSTRING, OVModel
from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME, STR_TO_OV_TYPE


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)

core = Core()
core.set_property("CPU", {"EXECUTION_MODE_HINT": "ACCURACY"})


TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> inputs = tokenizer("I love this story because", return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)
    ```
    Example using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    >>> text = "I love this story because"
    >>> gen = gen_pipeline(text)
    ```
"""

_SUPPORTED_ARCHITECTURES = {
    "bart",
    "blenderbot",
    "blenderbot-small",
    "bloom",
    "codegen",
    "gpt2",
    "gpt-bigcode",
    "gpt-neo",
    "gpt-neox",
    "llama",
    "marian",
    "opt",
    "pegasus",
}


@add_start_docstrings(
    """
    Base OVBaseDecoderModel class.
    """,
)
class OVBaseDecoderModel(OVModel):
    def __init__(
        self,
        model: openvino.runtime.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        if not dynamic_shapes:
            raise ValueError(
                "`dynamic_shapes` was set to `False` but static shapes are not supported for causal language model. Please set `dynamic_shapes=True`."
            )

        enable_compilation = kwargs.get("compile", True)
        kwargs["compile"] = False  # avoid extra compilation in the base class

        super().__init__(
            model,
            config,
            device=device,
            dynamic_shapes=False,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            **kwargs,
        )

        self.is_dynamic = dynamic_shapes
        use_cache = kwargs.pop("use_cache", True)
        self.use_cache = any("past_key_values" in key.get_any_name() for key in model.inputs)
        self.main_input_name = "input_ids"
        self.num_pkv = 2
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.key_value_output_names = [key for key in self.output_names if "present" in key]
        self._original_model = self.model.clone()  # keep original model for serialization
        self._pkv_precision = Type.f32
        self.update_pkv_precision()
        if self.is_dynamic:
            self.model = self._reshape(self.model, -1, -1)
        if enable_compilation:
            self.compile()

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

    def update_pkv_precision(self, force_fp32=False):
        if not self.use_cache:
            return

        pkv_precision = Type.f32
        if not force_fp32:
            device = self._device.upper()
            try:
                if "INFERENCE_PRECISION_HINT" in core.get_property(device, "SUPPORTED_PROPERTIES"):
                    pkv_precision = core.get_property(device, "INFERENCE_PRECISION_HINT")
            except RuntimeError:  # use default precision when get_property fails, e.g. when device is "AUTO:GPU"
                pass

            # ov_config["INFERENCE_PRECISION_HINT"] may override the prefer precision
            if self.ov_config:
                inference_precision_hint = self.ov_config.get("INFERENCE_PRECISION_HINT", "")
                if inference_precision_hint in STR_TO_OV_TYPE:
                    pkv_precision = STR_TO_OV_TYPE[inference_precision_hint]

            ppp = PrePostProcessor(self.model)
            for key in self.model.inputs:
                if "past_key_values" in key.get_any_name() and pkv_precision != key.get_element_type():
                    ppp.input(key.get_any_name()).tensor().set_element_type(pkv_precision)
            for key in self.model.outputs:
                if "present" in key.get_any_name() and pkv_precision != key.get_element_type():
                    ppp.output(key.get_any_name()).tensor().set_element_type(pkv_precision)

            self.model = ppp.build()
            self._pkv_precision = pkv_precision
        else:
            if hasattr(self, "_pkv_precision") and self._pkv_precision != Type.f32:
                self._pkv_precision = Type.f32
                self.model = self._original_model.clone()
                if self.is_dynamic:
                    self.model = self._reshape(self.model, -1, -1)
                self.request = None

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        model_to_save = self.model if self._pkv_precision == Type.f32 else self._original_model
        dst_path = os.path.join(save_directory, OV_XML_FILE_NAME)
        openvino.save_model(model_to_save, dst_path, compress_to_fp16=True)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        use_cache: bool = True,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        if config.model_type.replace("_", "-") not in _SUPPORTED_ARCHITECTURES:
            logger.warning(
                f"This architecture : {config.model_type} was not validated, only :{', '.join(_SUPPORTED_ARCHITECTURES)} architectures were "
                "validated, use at your own risk."
            )
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        if task is None:
            task = cls.export_feature

            if use_cache:
                task = task + "-with-past"

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            int8=load_in_8bit,
        )

        config.is_decoder = True
        config.is_encoder_decoder = False
        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(
            model_id=save_dir_path, config=config, use_cache=use_cache, load_in_8bit=load_in_8bit, **kwargs
        )

    def _reshape(
        self,
        model: openvino.runtime.Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        if height is not None:
            logger.warning(f"`height` set to `{height}` will be ignored during reshaping operation.")

        if width is not None:
            logger.warning(f"`width` set to `{width}` will be ignored during reshaping operation.")

        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith("past_key_values"):
                if len(inputs.partial_shape) == 3 and input_name.endswith("value"):
                    shapes[inputs][1] = -1
                else:
                    shapes[inputs][2] = -1
            else:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int):
        logger.warning("Static shapes are not supported for causal language model.")
        return self

    def compile(self):
        if self.request is None:
            super().compile()
            self.request = self.request.create_infer_request()


@add_start_docstrings(
    """
    OpenVINO Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForCausalLM(OVBaseDecoderModel, GenerationMixin):
    export_feature = "text-generation"
    auto_model_class = AutoModelForCausalLM

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForCausalLM",
            checkpoint="gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()
        inputs = {}

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        past_len = 0
        if past_key_values is not None:
            if self.config.model_type not in MULTI_QUERY_ATTN_MODELS:
                past_len = past_key_values[0][1].shape[-2]
                if self._pkv_precision == Type.bf16:
                    # numpy does not support bf16, pretending f16, should change to bf16
                    past_key_values = tuple(
                        Tensor(past_key_value, past_key_value.shape, Type.bf16)
                        for pkv_per_layer in past_key_values
                        for past_key_value in pkv_per_layer
                    )
                else:
                    # Flatten the past_key_values
                    past_key_values = tuple(
                        past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                    )
            else:
                past_len = past_key_values[0].shape[-2]

            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            batch_size = input_ids.shape[0]
            if self.config.model_type == "bloom":
                batch_size *= self.normalized_config.num_attention_heads

            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = batch_size
                if shape[2].is_dynamic:
                    shape[2] = 0
                else:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs["input_ids"] = np.array(input_ids)
        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names:
            if attention_mask is not None:
                inputs["attention_mask"] = np.array(attention_mask)
            else:
                inputs["attention_mask"] = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            if self.config.model_type not in MULTI_QUERY_ATTN_MODELS:
                # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
                past_key_values = tuple(
                    past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past) for layer_past in past_key_values
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)

        model_type = config.model_type.replace("_", "-")
        if model_type == "bloom":
            init_cls = OVBloomForCausalLM
        elif model_type == "mpt":
            init_cls = OVMPTForCausalLM
        elif model_type == "opt":
            init_cls = OVOPTForCausalLM
        elif model_type == "gpt-bigcode":
            init_cls = OVGPTBigCodeForCausalLM
        else:
            init_cls = cls

        return init_cls(model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs)


class OVBloomForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        # only last token for input_ids if past is not None
        if past_key_values:
            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }

    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called for bloom architecture.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        standardized_past = self._convert_to_standard_cache(past_key_values, batch_size=len(beam_idx))
        reordered_past = tuple(
            (
                np.take(layer_past[0], beam_idx, 0),
                np.take(layer_past[1], beam_idx, 0),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)

    # Copied from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache
    @staticmethod
    def _convert_to_bloom_cache(past_key_value: Tuple[Tuple[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape((batch_size_times_num_heads, head_dim, seq_length)),
                layer_past[1].reshape((batch_size_times_num_heads, seq_length, head_dim)),
            )
            for layer_past in past_key_value
        )

    # Adapted from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache
    def _convert_to_standard_cache(
        self, past_key_value: Tuple[Tuple[torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size, num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape((batch_size, num_heads, head_dim, seq_length)),
                layer_past[1].reshape((batch_size, num_heads, seq_length, head_dim)),
            )
            for layer_past in past_key_value
        )


class OVOPTForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class OVMPTForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class OVGPTBigCodeForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM._reorder_cache
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(np.take(layer_past, beam_idx, 0) for layer_past in past_key_values)

class StopWordsLogitsProcessor(LogitsProcessor):
    '''
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.

    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    '''

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f'`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}.'
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f'`stop_words_ids` has to be a list of lists, but is {stop_words_ids}.'
            )
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f'Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}.'
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), 'Stop words token sequences {} cannot have an empty list'.format(
                stop_words_ids
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    # if tokens do not match continue
                    match = True
                    break
            stopped_samples.append(match)

        return stopped_samples

class OVQwenModel(OVModelForCausalLM):
    def _reshape(
        self,
        model: openvino.runtime.Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = '',
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        init_cls = OVQwenModel

        return init_cls(model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get('past', None)

        # `past_key_values` may be in the stardard format (e.g. in contrastive search), converts to bloom's format if needed
        if past_key_values is not None and self.config.model_type == 'bloom':
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            'input_ids': input_ids,
            'past_key_values': past_key_values,
            'use_cache': self.use_cache,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'token_type_ids': None,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs: 'ModelOutput',
        model_kwargs: Dict[str, 'Any'],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, 'Any']:
        # update past_key_values
        model_kwargs['past_key_values'] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if 'attention_mask' in model_kwargs:
            attention_mask = model_kwargs['attention_mask']
            model_kwargs['attention_mask'] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if 'position_ids' in model_kwargs:
            position_ids = model_kwargs['position_ids']
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs['position_ids'] = torch.cat([position_ids, new_position_id], dim=-1)

        model_kwargs['is_first_forward'] = False
        return model_kwargs


    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        #assistant_model: Optional['PreTrainedModel'] = None,
        #streamer: Optional['BaseStreamer'] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        # Process stop_words_ids.
        stop_words_ids = kwargs.pop('stop_words_ids', [[151643]])
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, 'stop_words_ids', None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, 'stop_words_ids', None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            **kwargs,
        )