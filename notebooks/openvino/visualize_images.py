import matplotlib.pyplot as plt
import numpy as np
import transformers
from pathlib import Path
import matplotlib.image as mpimg
from collections import OrderedDict

transformers.logging.set_verbosity_error()

MODEL_ID = "stabilityai/stable-diffusion-2-1"
base_model_path = Path(f"models/{MODEL_ID}")

PREFIXES = [
    # "_INT8_HYBRID",
    "_FP32",
    "_INT8",
    # "_INT8_LORA_8",
    # "_INT8_LORA_32",
    # "_INT8_LORA_256",
]
NUM_STEPS = '50steps'
IMG_INDEXES = [0,1,5]
img_path_per_prefix = OrderedDict()
N_MODELS = len(PREFIXES)
num_images = set()
for prefix in PREFIXES:
    imgs_dir = base_model_path.with_name(base_model_path.name + prefix) / NUM_STEPS
    assert imgs_dir.exists(), f'Directory with images does not exist: {imgs_dir}'
    paths = list(imgs_dir.glob('*.png'))
    paths = np.array(paths).take(IMG_INDEXES)
    num_images.add(len(paths))
    img_path_per_prefix[prefix] = paths

assert len(num_images) == 1, f'Not equal number of images: {num_images}'
N_IMAGES = num_images.pop()

figsize = (N_IMAGES * 20, N_MODELS * 20)
fig, axs = plt.subplots(N_MODELS, N_IMAGES, figsize=figsize, sharex='all', sharey='all')
fig.patch.set_facecolor('white')
list_axes = list(axs.flat)
for a in list_axes:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.grid(False)

for i, (prefix, paths) in enumerate(img_path_per_prefix.items()):
    for j, img_path in enumerate(paths):
        print('Process ', img_path)
        img = mpimg.imread(img_path)
        if i==0 and j==0:
            list_axes[i * N_IMAGES + j].set_title(f"{MODEL_ID}\n{NUM_STEPS}\n{prefix[1:]}", fontsize=40)
        if i > 0 and j==0:
            list_axes[i * N_IMAGES + j].set_title(prefix[1:], fontsize=40)
        list_axes[i * N_IMAGES + j].imshow(img)
    # fig.subplots_adjust(wspace=0.01, hspace=0.0)
    fig.tight_layout()

    indexes = 'all' if IMG_INDEXES is None else 'IDs' + ''.join(map(str,IMG_INDEXES))
    modes = ''.join(PREFIXES)[1:]
    name = f"{modes}_{NUM_STEPS}_{indexes}.png"
    plt.savefig(name)
