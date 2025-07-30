import onnx
from onnxconverter_common import float16

model = onnx.load("/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-20240816_list_weighted_sampler_data_balanced_pack-satur0.6aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_9200.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-20240816_list_weighted_sampler_data_balanced_pack-satur0.6aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_9200_qt_float16.onnx")
