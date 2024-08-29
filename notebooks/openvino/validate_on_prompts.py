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


@dataclass
class ExpDesc:
    prompt: str
    seed: int = 1
    negative_prompt: str = ''
    num_inference_steps: int = 20

DESCS = [
    ExpDesc(
        prompt="a portrait of an old coal miner in 19th century, beautiful painting with highly detailed face by greg rutkowskiand magali villanueve",
        negative_prompt="deformed face, Ugly, bad quality, lowres, monochrome, bad anatomy",
        seed=1507302932
    ),
    ExpDesc(
        prompt = "Pikachu commitingtax fraud, paperwork, exhausted, cute, really cute, cozy, by stevehanks, by lisa yuskavage, by serov valentin, by tarkovsky, 8 k render, detailed, cute cartoon style",
        seed = 345,
        negative_prompt="",
    ),
    ExpDesc(
        prompt = "amazon rainforest with many trees photorealistic detailed leaves",
        negative_prompt = "blurry, poor quality, deformed, cartoonish, painting",
        seed = 1137900754
    ),
    ExpDesc(
        prompt="autumn in paris, ornate, beautiful, atmosphere, vibe, mist, smoke, fire, chimney, rain, wet, pristine, puddles, melting, dripping, snow, creek, lush, ice, bridge, forest, roses, flowers, by stanley artgerm lau, greg rutkowski, thomas kindkade, alphonse mucha, loish, norman rockwell",
        negative_prompt="",
        seed = 2132889432
    ),
    ExpDesc(
        prompt="portrait of renaud sechan, pen and ink, intricate line drawings, by craig mullins, ruanjia, kentaro miura, greg rutkowski, loundraw",
        negative_prompt="hyperrealism",
        seed = 206890696,
    ),
    ExpDesc(
        prompt="An astronaut laying down in a bed of millions of vibrant, colorful flowers and plants, photoshoot",
        negative_prompt="deformed face, Ugly, bad quality, lowres, monochrome, bad anatomy",
        seed = 3997429436,
    ),
    ExpDesc(
        prompt="long range view, Beautiful Japanese flower garden, elegant bridges, waterfalls, pink and white, by Akihito Yoshida, Ismail Inceoglu, Karol Bak, Airbrush, Dramatic, Panorama, Cool ColorPalette, Megapixel, Lumen Reflections, insanely detailed and intricate, hypermaximalist, elegant, ornate, hyper realistic, super detailed, unreal engine",
        negative_prompt="lowres, bad, deformed",
        seed = 128694831,
    ),

    ### my
    ExpDesc(
        prompt = "the best place in Bayern",
        seed = 1,
        negative_prompt="",
    ),

    ### Liubov
    ExpDesc(
        prompt = "a photo of an astronaut riding a horse on mars",
        seed = 1,
        negative_prompt="",
    ),
    ExpDesc(
        prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux",
        seed = 1,
        negative_prompt="",
    ),

    # ExpDesc(
    #     prompt = "The spirit of a tamagotchi wandering in the city of Vienna",
    #     seed = 23,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "a beautiful pink unicorn, 8k",
    #     seed = 1,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "Super cute fluffy cat warrior in armor, photorealistic, 4K, ultra detailed, vray rendering, unreal engine",
    #     seed = 1,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "a train that is parked on tracks and has graffiti writing on it, with a mountain range in the background",
    #     seed = 1,
    #     negative_prompt="",
    # ),
]


def generate_image(pipeline, prompt, seed, negative_prompt, num_inference_steps):
    transformers.set_seed(seed)
    return pipeline(
        prompt=prompt,
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
    # "_FP16",
    # '_UNET_HYBRID_REST_W32',
    # "_UNET_HYBRID_REST_W16",
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

    # "_UNET_W8A8_LORA_32__X32__REST_W16",
    # "_UNET_W8A8_LORA_8__X32_iter1_noreg__REST_W16"
    # "_UNET_W8A8_SQ_conv0.15_REST_W16"
    # "_UNET_W8A8_LORA_32_SQ_conv0.15_iter3_reg_cache_REST_W16"
    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_cache_REST_W16"

    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_last_REST_W16",
    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_last3_REST_W16",
    # "_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_last10_REST_W16",
    # "w8a8_x32_sq0.15_first_half_rest_w16",
    # "w8a8_x32_sq0.15_last_half_rest_w16",
    "w8a8_x32_sq0.15_higher_median_rest_w16",
    "w8a8_x32_sq0.15_less_median_rest_w16"
]

NUM_STEPS = [
    20,
    # 50
]

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
        for desc in DESCS:
            prompt = desc.prompt
            print(f'{border}Current prompt: {prompt}')
            for num_steps in NUM_STEPS:
                print(f'{border}Current num_steps: {num_steps}')
                desc.num_inference_steps = num_steps
                lora_img = generate_image(sd_pipe, **vars(desc))
                img_name = (prompt.replace(' ', '_')[:20] + '.png')
                im_folder = model_path / f'{desc.num_inference_steps}steps'
                im_folder.mkdir(exist_ok=True, parents=True)
                img_path = im_folder / img_name
                plt.imsave(img_path, np.array(lora_img))
                print('save img to %s' % img_path)
