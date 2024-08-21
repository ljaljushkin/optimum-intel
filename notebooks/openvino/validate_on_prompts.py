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

transformers.logging.set_verbosity_error()

MODEL_ID = "stabilityai/stable-diffusion-2-1"
base_model_path = Path(f"models/{MODEL_ID}")


@dataclass
class ExpDesc:
    prompt: str
    seed: int = 1
    negative_prompt: str = ''
    num_inference_steps: int = 50

DESCS = [
    ExpDesc(
        prompt = "a photo of an astronaut riding a horse on mars",
        seed = 1,
        negative_prompt="",
    ),
    ExpDesc(
        prompt = "the best place in Bayern",
        seed = 1,
        negative_prompt="",
    ),
    # ExpDesc(
    #     prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux",
    #     seed = 1,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "Pikachu commitingtax fraud, paperwork, exhausted, cute, really cute, cozy, by stevehanks, by lisa yuskavage, by serov valentin, by tarkovsky, 8 k render, detailed, cute cartoon style",
    #     seed = 345,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "amazon rainforest with many trees photorealistic detailed leaves",
    #     negative_prompt = "blurry, poor quality, deformed, cartoonish, painting",
    #     seed = 1137900754
    # ),
    # ExpDesc(
    #     prompt="autumn in paris, ornate, beautiful, atmosphere, vibe, mist, smoke, fire, chimney, rain, wet, pristine, puddles, melting, dripping, snow, creek, lush, ice, bridge, forest, roses, flowers, by stanley artgerm lau, greg rutkowski, thomas kindkade, alphonse mucha, loish, norman rockwell",
    #     negative_prompt="",
    #     seed = 2132889432
    # ),
    # ExpDesc(
    #     prompt="portrait of renaud sechan, pen and ink, intricate line drawings, by craig mullins, ruanjia, kentaro miura, greg rutkowski, loundraw",
    #     negative_prompt="hyperrealism",
    #     seed = 206890696,
    # ),
    # ExpDesc(
    #     prompt="An astronaut laying down in a bed of millions of vibrant, colorful flowers and plants, photoshoot",
    #     negative_prompt="deformed face, Ugly, bad quality, lowres, monochrome, bad anatomy",
    #     seed = 3997429436,
    # ),
    # ExpDesc(
    #     prompt="long range view, Beautiful Japanese flower garden, elegant bridges, waterfalls, pink and white, by Akihito Yoshida, Ismail Inceoglu, Karol Bak, Airbrush, Dramatic, Panorama, Cool ColorPalette, Megapixel, Lumen Reflections, insanely detailed and intricate, hypermaximalist, elegant, ornate, hyper realistic, super detailed, unreal engine",
    #     negative_prompt="lowres, bad, deformed",
    #     seed = 128694831,
    # ),
]


# TODO:
# visualize results for the same model and different prompts in a line
    # Can take saved pictures and locate in the layout afterwards
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # img = mpimg.imread('your_image.png')
    # imgplot = plt.imshow(img)
    # plt.show()
# try 1.5 vs 2.1 base

def generate_image(pipeline, prompt, seed, negative_prompt, num_inference_steps):
    transformers.set_seed(seed)
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.0,
        output_type="pil"
    ).images[0]

PREFIXES = [
    # "_INT8_LORA_8",
    # "_INT8_LORA_256",
    "_INT8",
    # "_INT8_HYBRID",
    "_FP32",
    # "_INT8_LORA_32",
]
for prefix in tqdm(PREFIXES, desc='Evaluating models'):
    model_path = base_model_path.with_name(base_model_path.name + prefix)
    print(f'Evaluating {prefix}')
    sd_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=model_path)
    for desc in DESCS:
        prompt = desc.prompt
        print('process %s' % prompt)
        lora_img = generate_image(sd_pipe, **vars(desc))
        img_name = (prompt.replace(' ', '_')[:20] + '.png')
        im_folder = model_path / f'{desc.num_inference_steps}steps_env2'
        im_folder.mkdir(exist_ok=True, parents=True)
        img_path = im_folder / img_name
        plt.imsave(img_path, np.array(lora_img))
        print('save img to %s' % img_path)


# visualize_results(lora_img)

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

# visualize_results(lora_img)
