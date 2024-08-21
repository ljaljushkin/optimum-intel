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

MODEL_ID = "stabilityai/stable-diffusion-2-1"
DATASET_NAME = "jxie/coco_captions"

base_model_path = Path(f"models/{MODEL_ID}")
fp32_model_path = base_model_path.with_name(base_model_path.name + "_FP32")
int8_model_path = base_model_path.with_name(base_model_path.name + "_INT8")

dataset = datasets.load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(seed=42)
print(next(iter(dataset)))


def preprocess_fn(example):
    return {"prompt": example["caption"]}

NUM_SAMPLES = 200
dataset = dataset.take(NUM_SAMPLES)
calibration_dataset = dataset.map(lambda x: preprocess_fn(x), remove_columns=dataset.column_names)


# int8_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=MODEL_ID, export=True)
int8_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=int8_model_path)
quantization_config = OVWeightQuantizationConfig(bits=8, num_samples=NUM_SAMPLES, quant_method=OVQuantizationMethod.HYBRID)
quantizer = OVQuantizer(int8_pipe)
quantizer.quantize(
    ov_config=OVConfig(quantization_config=quantization_config),
    calibration_dataset=calibration_dataset,
    save_directory=base_model_path.with_name(base_model_path.name + "_TMP")
)



# fp32_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=MODEL_ID, export=True)
# fp32_pipe.save_pretrained(fp32_model_path)


# def visualize_results(lora_img):
#     im_w, im_h = lora_img.size
#     is_horizontal = im_h <= im_w
#     figsize = (20, 30) if is_horizontal else (30, 20)
#     fig, axs = plt.subplots(1 if is_horizontal else 2, 2 if is_horizontal else 1, figsize=figsize, sharex='all', sharey='all')
#     fig.patch.set_facecolor('white')
#     list_axes = list(axs.flat)
#     for a in list_axes:
#         a.set_xticklabels([])
#         a.set_yticklabels([])
#         a.get_xaxis().set_visible(False)
#         a.get_yaxis().set_visible(False)
#         a.grid(False)
#     list_axes[0].imshow(np.array(lora_img))
#     # list_axes[1].imshow(np.array(int8_img))
#     img1_title = "INT8 lora result"
#     # img2_title = "INT8 result"
#     list_axes[0].set_title(img1_title, fontsize=20)
#     # list_axes[1].set_title(img2_title, fontsize=20)
#     fig.subplots_adjust(wspace=0.0 if is_horizontal else 0.01 , hspace=0.01 if is_horizontal else 0.0)
#     fig.tight_layout()


# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
# prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
# prompt = "a photo of an astronaut riding a horse on mars"
# prompt = "the best place in Bayern"

# def generate_image(pipeline, prompt):
#     transformers.set_seed(1)
#     return pipeline(
#         prompt=prompt,
#         guidance_scale=8.0,
#         output_type="pil"
#     ).images[0]

# # fp32_img = generate_image(fp32_pipe, prompt)
# lora_img = generate_image(int8_pipe, prompt)

# visualize_results(lora_img)
