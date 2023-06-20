#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
Fine-tuning a 🤗 Transformers model for image classification while applying quantization aware training with NNCF.
"""

from copy import deepcopy
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
import torch

import evaluate
import jstyleson as json
import numpy as np
import torch
import transformers
from datasets import load_dataset
from nncf.common.utils.os import safe_open
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AdamW,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from optimum.intel.openvino import OVConfig, OVTrainer, OVTrainingArguments


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_train_steps: Optional[int] = field(
        default=None
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    teacher_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models as teacher model in distillation."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    nncf_compression_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to NNCF configuration .json file for adapting the model to compression-enabled training."
        },
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OVTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    from datasets.splits import Split
    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            task="image-classification",
            # split=Split.VALIDATION,
            # streaming=True,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task="image-classification",
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    # for streaming mode
    # labels = dataset["train"].features["label"].names
    labels = dataset["train"].features["labels"].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    teacher_model = None
    if model_args.teacher_model_name_or_path is not None:
        teacher_model = AutoModelForImageClassification.from_pretrained(
            model_args.teacher_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.teacher_model_name_or_path),
            cache_dir=model_args.cache_dir,
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    if isinstance(feature_extractor.size, dict):
        if "shortest_edge" in feature_extractor.size:
            size = feature_extractor.size["shortest_edge"]
        else:
            size = (feature_extractor.size["height"], feature_extractor.size["width"])
    else:
        size = feature_extractor.size
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    if model_args.nncf_compression_config is not None:
        file_path = Path(model_args.nncf_compression_config).resolve()
        with safe_open(file_path) as f:
            compression = json.load(f)
        ov_config = OVConfig(compression=compression)
    else:
        ov_config = OVConfig()

    # Initialize our trainer
    teacher_model = deepcopy(model).to(training_args.device)
    trainer = OVTrainer(
        model=model,
        teacher_model=None,
        ov_config=ov_config,
        task="image-classification",
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )

    # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()
    #     trainer.log_metrics("train", train_result.metrics)
    #     trainer.save_metrics("train", train_result.metrics)
    #     trainer.save_state()

    do_lkd_tuning(
        training_args,
        student_model=model,
        teacher_model=teacher_model,
        trainer=trainer,
        data_args=data_args,
    )

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    # kwargs = {
    #     "finetuned_from": model_args.model_name_or_path,
    #     "tasks": "image-classification",
    #     "dataset": data_args.dataset_name,
    #     "tags": ["image-classification", "vision"],
    # }
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

def get_learning_rate(lr_scheduler, optimizer):
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        last_lr = optimizer.param_groups[0]["lr"]
    else:
        last_lr = lr_scheduler.get_last_lr()[0]
    if torch.is_tensor(last_lr):
        last_lr = last_lr.item()
    return last_lr

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output

def do_lkd_tuning(training_args, student_model, teacher_model, trainer, data_args):
    print(student_model)
    student_model.eval()
    teacher_model.eval()

    no_decay=['bias', 'LayerNorm.weight']
    ignored_names = ['signed_tensor', '_num_bits', '_scale_param_storage']
    target_names = ['weight', 'bias'] #'_scale_param_storage']

    num_train_epochs=1
    max_train_steps=data_args.max_train_steps
    weight_decay=training_args.weight_decay
    learning_rate=training_args.learning_rate

    hparam_dict = {
        'lr': learning_rate,
        'wd': weight_decay,
        'steps': max_train_steps,
    }

    from datetime import datetime
    exp_name = '__'.join(f'{k}={v}'for k,v in hparam_dict.items()) + "__" + datetime.now().strftime("%b%d_%H-%M-%S")
    tb = SummaryWriter(log_dir=Path('/home/nlyaly/sandbox/models/KD/iterative/ImageNet/Optimum') / exp_name)
    tb.add_text('ignored_names', str(ignored_names))
    tb.add_text('target_names', str(target_names))
    tb.add_text('no_decay', str(no_decay))

    num_layers = student_model.config.num_hidden_layers
    num_improved = 0
    diff_improved = []

    tb.add_text('training_args', str(training_args))
    tb.add_text('data_args', str(data_args))

    first_names_wd = None
    first_names_no_wd = None
    for l in range(num_layers): # iterate across BERT layers
        print(f'process layer{l}')
        student_layer = recursive_getattr(student_model, f'vit.encoder.layer.{l}')  # extract the lth layer of student
        ignored_target_fn = lambda pair: not any(i in pair[0] for i in ignored_names) or any(t in pair[0] for t in target_names)
        wd_fn = lambda pair: not any(nd in pair[0] for nd in no_decay)
        no_wd_fn = lambda pair: any(nd in pair[0] for nd in no_decay)
        np_with_wd = dict(filter(wd_fn, filter(ignored_target_fn, student_layer.named_parameters())))
        np_no_wd = dict(filter(no_wd_fn, filter(ignored_target_fn, student_layer.named_parameters())))

        optimizer_param = [
            {
                "params": np_with_wd.values(),
                "weight_decay": weight_decay,
            },
            {
                "params": np_no_wd.values(),
                "weight_decay": 0.0,
            },
        ]
        if l == 0:
            print(' With wd:')
            print(*np_with_wd.keys(), sep='\n')
            print(' No_wd:')
            print(*np_no_wd.keys(), sep='\n')
            first_names_wd = np_with_wd.keys()
            first_names_no_wd = np_no_wd.keys()
            tb.add_text('wd_names', str(list(first_names_wd)))
            tb.add_text('no_wd_names', str(list(first_names_no_wd)))


        optimizer = AdamW(optimizer_param, lr=learning_rate)
        lr_scheduler = trainer.create_scheduler(num_training_steps=max_train_steps, optimizer=optimizer)
        # lr_scheduler.cooldown = 100
        # lr_scheduler.min_lr = 1e-8

        updated_steps = 0
        for _ in range(num_train_epochs):
            first_loss = None
            for data_idx, batch in enumerate(trainer.nik_train_dataloader):  # load each batch
                batch = to_device(batch, training_args.device)
                with torch.no_grad():
                    # for simplicity, we always run the full inference of the teacher model.
                    # To get the best performance, you can run the teacher model only for the first l layers,
                    # which requires some modifications to the modeling code.
                    teacher_out = teacher_model(**batch, output_hidden_states=True) # get the output of the teacher model
                layer_input = teacher_out.hidden_states[l] # extract the lth-layer's input of teacher
                teacher_o = teacher_out.hidden_states[l+1] # extract the lth-layer's output of teacher

                # real_mask = teacher_model.vit.get_extended_attention_mask(batch['attention_mask'], \
                #     batch['input_ids'].shape, batch['input_ids'].device) # get the mask
                student_o = student_layer(layer_input)[0]#, real_mask)[0] # run inference for the student
                loss = torch.nn.functional.mse_loss(student_o, teacher_o)
                if first_loss is None:
                    first_loss = loss
                tb.add_scalar(f"loss for {l} layer", loss.item(), data_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(data_idx)
                tb.add_scalar(f"LR for {l} layer", get_learning_rate(lr_scheduler, optimizer), data_idx)
                updated_steps += 1
                if updated_steps >= max_train_steps :  # break when the number of steps is reached, typically in hundreds
                    diff = loss - first_loss
                    prefix = '=)' if diff < 0 else '=('
                    percent = diff / first_loss * (-100)
                    tb.add_scalar("kd_loss_decrease", percent, l)
                    if diff < 0:
                        num_improved+=1
                    diff_improved.append(abs(percent.item()))
                    # print(f"diff={diff} is_good={diff<0}")
                    print(f"{prefix} {percent.item():.2f}% for {l}")
                    break

            if updated_steps >= max_train_steps:
                break

        metrics = trainer.evaluate()
        accuracy=metrics['eval_accuracy']
        student_model.eval()
        tb.add_scalar("per_layer_accuracy", accuracy, l)

    from statistics import mean
    metric_dict={
        "accuracy": accuracy,
        "num_improved": num_improved,
        "mean improvement": mean(diff_improved)
    }
    tb.add_hparams(hparam_dict, metric_dict)


if __name__ == "__main__":
    main()
