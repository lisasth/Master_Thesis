import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = '/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-encode-no-space-train-CCTVMirror-0214-0201-and-top-down-0201-and-MS-PG-small-and-SCO-and-UK-1913-select-200-little-aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_31100.onnx'
model_quant = '/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-encode-no-space-train-CCTVMirror-0214-0201-and-top-down-0201-and-MS-PG-small-and-SCO-and-UK-1913-select-200-little-aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_31100_uint8.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
