import shutil
import nncf
import time
import datasets
import matplotlib.pyplot as plt
import numpy as np
import openvino.runtime as ov
import transformers
from pathlib import Path
from openvino.runtime import Core
from optimum.intel import OVConfig, OVQuantizer, OVStableDiffusionPipeline, OVWeightQuantizationConfig
from optimum.intel.openvino.configuration import OVQuantizationMethod
from nncf.quantization.advanced_parameters import AdvancedLoraCorrectionParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

def form_experiment_folder(model_dir, unet_folder):
    config_path = model_dir / 'config.json'
    configs_dir = model_dir / 'CONFIGS' / 'links'
    rest_w16_dir = model_dir / 'REST_W16'/ 'links'
    pipe_folder = unet_folder.parent / f'{curr_exp}_rest_w16'

    shutil.copy(config_path, unet_folder)
    shutil.rmtree(pipe_folder, ignore_errors=True)
    pipe_folder.mkdir(exist_ok=True, parents=True)
    target_link = pipe_folder / 'unet'
    target_link.unlink(missing_ok=True)
    target_link.symlink_to(unet_folder, target_is_directory=False)

    def copy_links_from_folder(src_dir_with_links, dst_dir):
        for src_path in src_dir_with_links.iterdir():
            shutil.copy(src_path, dst_dir, follow_symlinks=False)

    copy_links_from_folder(configs_dir, pipe_folder)
    copy_links_from_folder(rest_w16_dir, pipe_folder)
    print('Directory is ready', pipe_folder.resolve())

# MODEL_ID = "stabilityai/stable-diffusion-2-1"
# DATASET_NAME = "jxie/coco_captions"
# MODEL_ID = "runwayml/stable-diffusion-v1-5"

# base_model_path = Path(f"models/{MODEL_ID}")
# fp32_model_path = base_model_path.with_name(base_model_path.name + "_FP32")
# int8_model_path = base_model_path.with_name(base_model_path.name + "_INT8")

# dataset = datasets.load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(seed=42)
# print(next(iter(dataset)))


# def preprocess_fn(example):
#     return {"prompt": example["caption"]}

# NUM_SAMPLES = 200
# dataset = dataset.take(NUM_SAMPLES)
# calibration_dataset = dataset.map(lambda x: preprocess_fn(x), remove_columns=dataset.column_names)

# def _prepare_unet_dataset():
# calibration_dataset = _prepare_unet_dataset(NUM_SAMPLES, dataset=calibration_dataset)

# model = ov.Core().read_model(int8_model_path / 'unet' / "openvino_model.xml")

# model = nncf.quantize(
#     model=model_fp32,
#     calibration_dataset=calibration_dataset,
#     model_type=nncf.ModelType.TRANSFORMER,
#     # ignored_scope=ptq_ignored_scope,
#     # SQ algo should be disabled for MatMul nodes because their weights are already compressed
#     advanced_parameters=nncf.AdvancedQuantizationParameters(
#         smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=-1, convolution=-1)
#     ),
#     subset_size=NUM_SAMPLES,
# )
calibration_dataset = {}
from nncf.common.utils.debug import nncf_debug
wc_advanced = AdvancedCompressionParameters(lora_correction_params=AdvancedLoraCorrectionParameters(rank=32, num_iters=3))


MODEL_DIR = Path('/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml')
fp32_model_path = MODEL_DIR / "UNET_FP32" / "openvino_model.xml"
model_fp32 = ov.Core().read_model(fp32_model_path)

import os
os.environ["ACTIVATION_STATS_LOAD_PATH"] = str(fp32_model_path.parent / 'sX_lora_stats.npz')
calibration_dataset = {}

# with nncf_debug():
model_fp32 = nncf.compress_weights(
    model_fp32,
    ratio=1.0,
    group_size=128,
    dataset=nncf.Dataset(calibration_dataset),
    mode=nncf.CompressWeightsMode.INT4_SYM,
    # ignored_scope=nncf.IgnoredScope(patterns=[
    #     ".*time_emb_proj.*"
    # ]),
    lora_correction=True,
    scale_estimation=True,
    advanced_parameters=wc_advanced
)

# from openvino._offline_transformaqtions import compress_quantize_weights_transformation
# compress_quantize_weights_transformation(model)


exp_names = [
    # 'w8a16',
    # 'w4a16_datafree',
    # 'w4a16_scale',
    # 'w4a16_svd32',
    # 'w4a16_lora32',
    # 'w4a16_scale_svd32',
    'w4a16_scale_lora32',
]

curr_exp = exp_names[0]

unet_folder = MODEL_DIR / f'UNET_{curr_exp}'
unet_folder.mkdir(exist_ok=True, parents=True)
ov.save_model(model_fp32, unet_folder / "openvino_model.xml", compress_to_fp16=False)

form_experiment_folder(MODEL_DIR, unet_folder)
