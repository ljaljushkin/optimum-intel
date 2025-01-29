import time
import datasets
import matplotlib.pyplot as plt
import numpy as np
import transformers
from pathlib import Path
from openvino.runtime import Core
from optimum.intel import OVConfig, OVQuantizer, OVStableDiffusionPipeline, OVWeightQuantizationConfig
from optimum.intel.openvino.configuration import OVQuantizationMethod

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

# MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATASET_NAME = "jxie/coco_captions"

base_model_path = Path(f"/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/{MODEL_ID}")
fp32_model_path = base_model_path.with_name(base_model_path.name + "_FP32")
int8_model_path = base_model_path.with_name(base_model_path.name + "_UNET_W8A8_NOT_TRANSFORMED_REST_W8")
# int8_model_path = base_model_path.with_name(base_model_path.name + "_UNET_W8A8_SQ_conv0.15_NOT_TRANSFORMED_REST_W8")
# int8_model_path = base_model_path.parent / "w8a8_biascorr_rest_w16_NOT_TRANSFORMED"

dataset = datasets.load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(seed=42)
print(next(iter(dataset)))


def preprocess_fn(example):
    return {"prompt": example["caption"]}

# NUM_SAMPLES = 200
NUM_SAMPLES = 6
dataset = dataset.take(NUM_SAMPLES)
calibration_dataset = dataset.map(lambda x: preprocess_fn(x), remove_columns=dataset.column_names)
print(type(calibration_dataset), calibration_dataset)
print(list(calibration_dataset))
# calibration_dataset = [
#     {"prompt": "a photo of an astronaut riding a horse on mars"}
# ]
# print(type(calibration_dataset), calibration_dataset)

# int8_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=MODEL_ID, export=True)
import os
# os.environ["FP32_LORA_ACTIVATION_STATS_SAVE_PATH"] = '/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml/X32_STATS_horse.npz'
# os.environ["FP32_LORA_ACTIVATION_STATS_LOAD_PATH"] = '/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml/X32_STATS_horse.npz'

# os.environ["FP32_LORA_ACTIVATION_STATS_SAVE_PATH"] = '/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml/X32_BIAS_CORRECT_STATS.npz'
# os.environ["FP32_LORA_ACTIVATION_STATS_LOAD_PATH"] = '/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml/X32_BIAS_CORRECT_STATS.npz'

# os.environ["ACTIVATION_STATS_SAVE_PATH"] = str(fp32_model_path / 'sX_lora_stats_horse.npz')
# os.environ["ACTIVATION_STATS_LOAD_PATH"] = str(fp32_model_path / 'sX_lora_stats_horse.npz')

int8_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=fp32_model_path)
quantization_config = OVWeightQuantizationConfig(bits=8, num_samples=NUM_SAMPLES, quant_method=OVQuantizationMethod.HYBRID)
quantizer = OVQuantizer(int8_pipe)
quantizer.quantize(
    ov_config=OVConfig(quantization_config=quantization_config),
    calibration_dataset=calibration_dataset,
    save_directory=base_model_path.with_name(base_model_path.name + "_TMP")
)
