import matplotlib.pyplot as plt
import numpy as np
import transformers
from pathlib import Path
import matplotlib.image as mpimg
from collections import OrderedDict
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms as transforms
from functools import partial
import matplotlib.pyplot as plt
import torch
from prompts import DESCS
from prompts import encode_prompt
from prompts import PROMPTS_MAP
import json
# from torchmetrics.functional.multimodal import clip_score
# inception_score = InceptionScore(normalize=True, splits=1)
# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")



transformers.logging.set_verbosity_error()

# MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
base_model_path = Path(f"models/{MODEL_ID}")

PREFIXES = [
    # "_FP32",
    "_FP16",

    # '_UNET_HYBRID_REST_W32',
    '_UNET_HYBRID_REST_W16',
    # "_UNET_HYBRID_REST_W8",

    # "_UNET_W8A8_REST_W32",
    "_UNET_W8A8_REST_W16",
    # "_UNET_W8A8_REST_W8",
    # '_UNET_W8A8_LORA_32_REST_W16',
    # "_UNET_W8A8_LORA_32__X32__REST_W16",

    # Smooth Quant
    # "_UNET_W8A8_SQ_conv0.15_REST_W16",
    # "_UNET_W8A8_LORA_32_SQ_conv0.15_iter3_reg_cache_REST_W16",
    # '_UNET_W8A8_LORA_32__X32__SQ_conv0.15_iter3_reg_cache_REST_W16',
    # "w8a8_x32_sq0.15_last1_rest_w16",
    # "w8a8_x32_sq0.15_last3_rest_w16",
    # "w8a8_x32_sq0.15_last10_rest_w16",
    # "w8a8_x32_sq0.15_first_half_rest_w16",
    # "w8a8_x32_sq0.15_last_half_rest_w16",
    # 'w8a8_x32_sq0.15_less_median_rest_w16',
    # 'w8a8_x32_sq0.15_higher_median_rest_w16',

    # BIAS CORRECTION
    # 'w8a8_biascorr_rest_w16_TRANSFORMED',
    # 'w8a8_partx32_biascorr_lora32_rest_w16',

    # '_UNET_W8A8_LORA_8_REST_W8',

    # '_UNET_W8A8_LORA_32_REST_W32',

    # '_UNET_W8A8_LORA_32_REST_W16_cache',
    # '_UNET_W8A8_LORA_32_REST_W8',

    # '_UNET_W8A8_LORA_256_REST_W32',
    # '_UNET_W8A8_LORA_256_REST_W16',
    # '_UNET_W8A8_LORA_256_REST_W8',

    # '_UNET_W8A8_LORA_32_ADAPT32_REST_W32',
    # '_UNET_W8A8_LORA_32_ADAPT32_REST_W16',
    # '_UNET_W8A8_LORA_32_ADAPT32_REST_W8',

    # "w8a16_rest_w16",
    # "w4a16_datafree_skip_ign_scope_rest_w16",
    # # 'w4a16_datafree_rest_w16',
    # # 'w4a16_datafree_pc_rest_w16',
    # "w4a16_scale_skip_rest_w16",
    # # 'w4a16_lora32_pc_rest_w16',
    # "w4a16_scale_lora32_skip_rest_w16",

    'w8a16_rest_w16',
    'w4a16_datafree_rest_w16',
    'w4a16_scale_rest_w16',
    'w4a16_svd32_rest_w16',
    'w4a16_lora32_rest_w16',
    'w4a16_scale_svd32_rest_w16',
    'w4a16_scale_lora32_rest_w16',
]



NUM_STEPS = '20steps'
FILENAME = 'w4a16.png'
IMG_NAMES = [
    # Xiaofan
    'a_portrait_of_an_old',
    'Pikachu_commitingtax',
    'amazon_rainforest_wi',
    'autumn_in_paris,_orn',
    'portrait_of_renaud_s',
    'An_astronaut_laying_',
    'long_range_view,_Bea',
    # my
    'the_best_place_in_Ba',
    # Liubov
    'a_photo_of_an_astron',
    'close-up_photography',
]
# IMG_INDEXES = [1,3,4,5,7,8]
img_path_per_prefix = OrderedDict()
N_MODELS = len(PREFIXES)
num_images = set()
for prefix in PREFIXES:
    if prefix.startswith('_'):
        imgs_dir = base_model_path.with_name(base_model_path.name + prefix) / NUM_STEPS
    else:
        imgs_dir = base_model_path.with_name(prefix) / NUM_STEPS
    assert imgs_dir.exists(), f'Directory with images does not exist: {imgs_dir}'
    paths = [img_path for img_name in IMG_NAMES if (img_path := (imgs_dir / img_name).with_suffix('.png')).exists()]
    # paths = list(imgs_dir.glob('*.png'))
    # if IMG_INDEXES is not None:
    #     paths = np.array(paths).take(IMG_INDEXES)
    num_images.add(len(paths))
    img_path_per_prefix[prefix] = paths

assert len(num_images) == 1, f'Not equal number of images: {num_images}'
N_IMAGES = num_images.pop()

FIG_SIZE = (N_IMAGES * 10, N_MODELS * 10)
FONT_SIZE = 40
fig, axs = plt.subplots(N_MODELS, N_IMAGES, figsize=FIG_SIZE, sharex='all', sharey='all')
fig.patch.set_facecolor('white')
list_axes = list(axs.flat)
for a in list_axes:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.grid(False)


for i, (prefix, paths) in enumerate(img_path_per_prefix.items()):

    scores_map = {}
    img_path = paths[0]
    print('Process ', img_path.parent)
    clip_scores_path = img_path.parent / 'clip_scores_on_fly.json'
    image_name = img_path.stem
    avg_clip_score_str = ''
    if clip_scores_path.exists():
        with clip_scores_path.open('r') as f:
            scores_map = json.load(f)
            avg_clip_score = np.mean(list(scores_map.values()))
            avg_clip_score_str = f"     Avg CLIP Score: {avg_clip_score:.2f}"
            print(avg_clip_score_str)
    inception_score_str = ''
    inception_scores_path = img_path.parent.parent / 'inception_score.json'
    if inception_scores_path.exists():
        with inception_scores_path.open('r') as f:
            inception_score = json.load(f)['inception_score']
            inception_score_str = f"     Inception Score: {inception_score:.2f}"
            print(inception_score_str)
    for j, img_path in enumerate(paths):
        # print('Process ', img_path)
        img = mpimg.imread(img_path)[:,:,:3]
        image_name = img_path.stem
        if image_name in scores_map:
            sd_clip_score = scores_map[image_name]
            clip_score_str = f"{sd_clip_score:.2f}"
            # print(f"CLIP score: {sd_clip_score}")
        else:
            clip_score_str = ''
            # prompt = PROMPTS_MAP[image_name]
            # clip_score = clip_score_fn(torch.from_numpy(img), prompt).detach()
            # sd_clip_score = round(float(clip_score), 4)
            # scores_map[image_name] = sd_clip_score

        title = clip_score_str
        if i==0 and j==0:
            title = f"{MODEL_ID}\n{NUM_STEPS}\n{prefix}\n{clip_score_str}"
        elif i > 0 and j==0:
            title = f"{prefix}{inception_score_str}{avg_clip_score_str}\n{clip_score_str}"
        list_axes[i * N_IMAGES + j].set_title(title, fontsize=FONT_SIZE)
        list_axes[i * N_IMAGES + j].imshow(img)
    # with cached_scores_path.open('w') as f:
    #     json.dump(scores_map, f)
    # fig.subplots_adjust(wspace=0.01, hspace=0.0)
    # fig.tight_layout()

    # indexes = 'all' if IMG_INDEXES is None else 'IDs' + ''.join(map(str, IMG_INDEXES))
    # modes = ''.join(PREFIXES)[1:]
    # name = f"SD_v{base_model_path.name[-3:]}_{modes}_{NUM_STEPS}_{N_IMAGES}imgs"
    # name = name.replace('.', '_')
    name = FILENAME

print('Saving result to: ', name)
plt.savefig('results/' + name)
