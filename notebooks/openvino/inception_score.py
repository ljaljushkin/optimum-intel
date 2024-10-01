from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms as transforms
from itertools import islice
import datasets
import time
import openvino as ov
from tqdm import tqdm
import torch
from pathlib import Path
from transformers import set_seed
import json
torch.manual_seed(42)
set_seed(42)

# VALIDATION_DATASET_SIZE = 100
VALIDATION_DATASET_SIZE = 25

core = ov.Core()


# model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "sd_vit_per_token/runwayml/stable-diffusion-v1-5/models"
# pipe = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False)


def compute_inception_score(pipe, unet_path, validation_set_size, batch_size=100):
    unet = core.read_model(unet_path)
    pipe.unet.model = unet
    pipe.unet.request = None

    dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", "unlabeled", split="validation", trust_remote_code=True).shuffle(seed=42)
    dataset = islice(dataset, validation_set_size)

    inception_score = InceptionScore(normalize=True, splits=1)

    images = []
    infer_times = []
    for batch in tqdm(dataset, total=validation_set_size, desc="Computing Inception Score"):
        prompt = batch["caption"]
        if len(prompt) > pipe.tokenizer.model_max_length:
            continue
        start_time = time.perf_counter()
        image = pipe(prompt).images[0]
        infer_times.append(time.perf_counter() - start_time)
        image = transforms.ToTensor()(image)
        images.append(image)

    mean_perf_time = sum(infer_times) / len(infer_times)

    while len(images) > 0:
        images_batch = torch.stack(images[-batch_size:])
        images = images[:-batch_size]
        inception_score.update(images_batch)
    kl_mean, kl_std = inception_score.compute()

    pipe.unet.model = unet
    return kl_mean, mean_perf_time


PREFIXES = [
    # '_UNET_W8A8_LORA_8_REST_W8',
    # '_UNET_W8A8_LORA_32_REST_W8',
    # '_UNET_W8A8_LORA_256_REST_W8',

    # "_FP32",
    # "_FP16", # GOLD
    # '_UNET_HYBRID_REST_W32',
    # "_UNET_HYBRID_REST_W16", # GOLD
    # "_UNET_HYBRID_REST_W8",
    # "_UNET_W8A8_REST_W32",
    # "_UNET_W8A8_REST_W16", # GOLD
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

    # "_UNET_W8A8_LORA_32__X32__REST_W16", # GOLD
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

    # 'w4a16_datafree_horse_rest_w16',
    # 'w4a16_scale_horse_rest_w16',
    'stable-diffusion-v1-5_FP16',
    'w4a16_sym_datafree_ign_time_emb_rest_w16',
    'w4a16_sym_gptq_ign_time_emb_rest_w16',
    'w4a16_sym_gptq_scale_ign_time_emb_rest_w16',
    'w4a16_sym_scale_ign_time_emb_rest_w16',
]

MODEL_IDS = [
    "runwayml/stable-diffusion-v1-5",
    # "stabilityai/stable-diffusion-2-1"
]

for model_id in tqdm(MODEL_IDS, desc='Evaluation per Model'):
    border = '#'*50 + ' '
    print(f'\n\n\n{border}Current model: {model_id}')
    base_model_path = Path(f"models/{model_id}")
    pipeline_dir = base_model_path.with_name(base_model_path.name + '_FP16')
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    sd_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=pipeline_dir, scheduler=scheduler)

    for prefix in tqdm(PREFIXES, desc='Evaluating per model\'s mode'):
        print(f'\n\n{border}Current mode: {prefix}')
        if prefix.startswith('_'):
            model_path = base_model_path.with_name(base_model_path.name + prefix)
        else:
            model_path = base_model_path.with_name(prefix)
        unet_path = model_path / 'unet' / 'openvino_model.xml'
        if not unet_path.exists():
            print('Skipping the mode, path does not exists: ', unet_path)
            continue

        inception_score, elapsed_time = compute_inception_score(sd_pipe, unet_path, VALIDATION_DATASET_SIZE)
        print(f"Inception Score: {inception_score}")

        result_dict = {
            'pipeline_dir': str(pipeline_dir),
            'inception_score': inception_score.item(),
            'dataset_size': VALIDATION_DATASET_SIZE,
        }
        with (model_path / 'inception_score_optimum_1.23.json').open('w') as f:
            json.dump(result_dict, f)


