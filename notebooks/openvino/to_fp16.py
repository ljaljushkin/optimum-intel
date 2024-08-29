import openvino.runtime as ov
from pathlib import Path

source_dir = Path('/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml/REST_W32')
target_dir = source_dir.with_name('REST_W16')

source_dir = Path('/home/nlyaly/projects/optimum-intel/notebooks/openvino/models/runwayml/UNET_FP32')
target_dir = source_dir.with_name('UNET_FP16')


model = ov.Core().read_model(source_dir / "openvino_model.xml")
target_dir.mkdir(exist_ok=True, parents=True)
ov.save_model(model, target_dir / 'openvino_model.xml', compress_to_fp16=True)

# for model_dir in source_dir.iterdir():
#     model_name = model_dir.name
#     model = ov.Core().read_model(model_dir / "openvino_model.xml")
#     (dir_to_save := (target_dir / model_name)).mkdir(exist_ok=True, parents=True)
#     print(model_name)
#     print(dir_to_save)
#     ov.save_model(model, dir_to_save / 'openvino_model.xml', compress_to_fp16=True)