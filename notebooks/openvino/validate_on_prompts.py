from dataclasses import dataclass
import time
import datasets
import matplotlib.pyplot as plt
import numpy as np
import transformers
from pathlib import Path
from openvino.runtime import Core
from optimum.intel import OVConfig, OVQuantizer, OVStableDiffusionPipeline, OVWeightQuantizationConfig
from optimum.intel.openvino.configuration import OVQuantizationMethod
from tqdm import tqdm
from diffusers import EulerDiscreteScheduler
transformers.logging.set_verbosity_error()
from torchmetrics.functional.multimodal import clip_score
import json
import torch
from prompts import DESCS
from prompts import PROMPTS_MAP
from prompts import encode_prompt
from functools import partial

def generate_image(pipeline, prompt, seed, negative_prompt, num_inference_steps):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    transformers.set_seed(seed)
    return pipeline(
        prompt=prompt,
        generator=rng,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        # guidance_scale=8.0,
        guidance_scale=7.5,
        output_type="pil"
    ).images[0]

MODEL_IDS = [
    "runwayml/stable-diffusion-v1-5",
    # "stabilityai/stable-diffusion-2-1"
]

PREFIXES = [
    # '_UNET_W8A8_LORA_8_REST_W8',
    # '_UNET_W8A8_LORA_32_REST_W8',
    # '_UNET_W8A8_LORA_256_REST_W8',

    # "_FP32",

    # '_UNET_HYBRID_REST_W32',
    # "_UNET_HYBRID_REST_W8",
    # "_UNET_W8A8_REST_W32",
    # "_UNET_W8A8_REST_W16",
    # "_UNET_W8A8_REST_W8",

    # "_UNET_W8A8_LORA_32_REST_W32",
    # "_UNET_W8A8_LORA_32_REST_W16",
    # "_UNET_W8A8_LORA_32_REST_W16_cache"
    # "_UNET_W8A8_LORA_32_REST_W8",

#     "_UNET_W8A8_LORA_32_ADAPT32_REST_W32",
#     "_UNET_W8A8_LORA_32_ADAPT32_REST_W16",
#     "_UNET_W8A8_LORA_32_ADAPT32_REST_W8",

#     "_UNET_W8A8_LORA_256_REST_W32",
#     "_UNET_W8A8_LORA_256_REST_W16",
#     "_UNET_W8A8_LORA_256_REST_W8",


    # "_UNET_W8A8_LORA_8__X32_iter1_noreg__REST_W16"
    # "_UNET_W8A8_SQ_conv0.15_REST_W16"
    # "_UNET_W8A8_LORA_32_SQ_conv0.15_iter3_reg_cache_REST_W16"
    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_cache_REST_W16"

    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_last_REST_W16",
    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_last3_REST_W16",
    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_last10_REST_W16",
    # "w8a8_x32_sq0.15_first_half_rest_w16",
    # "w8a8_x32_sq0.15_last_half_rest_w16",
    # "w8a8_x32_sq0.15_higher_median_rest_w16",
    # "w8a8_x32_sq0.15_less_median_rest_w16"
    # "w8a8_biascorr_rest_w16_TRANSFORMED"
    # "w8a8_partx32_biascorr_lora32_rest_w16"

    # "w4a16_datafree_rest_w16",
    # "w4a16_datafree_pc_rest_w16",
    # "w4a16_lora32_pc_rest_w16",
    # "w4a16_scale_skip_rest_w16",
    # "w4a16_scale_lora32_skip_rest_w16",
    # "w4a16_datafree_skip_ign_scope_rest_w16",
    # "w8a16_rest_w16",

    # 'w8a16_rest_w16',
    # 'w4a16_datafree_rest_w16',
    # 'w4a16_scale_rest_w16',
    # 'w4a16_svd32_rest_w16',
    # 'w4a16_lora32_rest_w16',
    # 'w4a16_scale_svd32_rest_w16',
    # 'w4a16_scale_lora32_rest_w16',
    # "W8A8_biascorr_LORA_32__X32__iter0_rest_w16"

    # overfit on single prompt
    # 'w4a16_datafree_horse_rest_w16',
    # 'w4a16_scale_horse_rest_w16',

    # asym
    # 'w4a16_asym_datafree_rest_w16',

    # sym
    # "_FP16",
    # "_UNET_HYBRID_REST_W16",
    # 'w4a16_sym_datafree_ign_time_emb_rest_w16',
    # 'w4a16_sym_gptq_ign_time_emb_rest_w16',
    # 'w4a16_sym_gptq_scale_ign_time_emb_rest_w16',
    # 'w4a16_sym_scale_ign_time_emb_rest_w16',
    # "_UNET_W8A8_LORA_32__X32__REST_W16",
    # 'w4a16_lora_rank32_gptq_style_rest_w16'
    'w4a16_lora_rank256_gptq_style_rest_w16'
]

NUM_STEPS = [
    20,
    # 50
]


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

for model_id in tqdm(MODEL_IDS, desc='Evaluation per Model'):
    border = '#'*50 + ' '
    print(f'\n\n\n{border}Current model: {model_id}')
    base_model_path = Path(f"models/{model_id}")
    for prefix in tqdm(PREFIXES, desc='Evaluating per model\'s mode'):
        print(f'\n\n{border}Current mode: {prefix}')
        if prefix.startswith('_'):
            model_path = base_model_path.with_name(base_model_path.name + prefix)
        else:
            model_path = base_model_path.with_name(prefix)
        if not model_path.exists():
            print('Skipping the mode, path does not exists: ', model_path)
            continue
        scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        sd_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=model_path, scheduler=scheduler)
        scores_map = {}
        for desc in DESCS:
            prompt = desc.prompt
            print(f'{border}Current prompt: {prompt}')
            for num_steps in NUM_STEPS:
                print(f'{border}Current num_steps: {num_steps}')
                desc.num_inference_steps = num_steps
                img = generate_image(sd_pipe, **vars(desc))
                img_name = encode_prompt(prompt)
                im_folder = model_path / f'{desc.num_inference_steps}steps_optimum_1.23'
                im_folder.mkdir(exist_ok=True, parents=True)
                img_path = im_folder / (img_name + '.png')
                plt.imsave(img_path, np.array(img))
                print('save img to %s' % img_path)

                prompt = PROMPTS_MAP[img_name]
                clip_score = clip_score_fn(torch.from_numpy(np.array(img)), prompt).detach()
                sd_clip_score = round(float(clip_score), 4)
                scores_map[img_name] = sd_clip_score
                print(f"CLIP score: {sd_clip_score}")

        with (im_folder / 'clip_scores_on_fly.json').open('w') as f:
            json.dump(scores_map, f)

