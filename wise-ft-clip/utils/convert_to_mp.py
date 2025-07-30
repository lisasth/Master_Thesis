from onnxconverter_common import auto_mixed_precision
import onnx
import numpy as np

model = onnx.load("/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-encode-no-space-train-CCTVMirror-0214-0201-and-top-down-0201-and-MS-PG-small-and-SCO-and-UK-1913-select-200-little-aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_31100.onnx")
test_data = {"input": np.random.rand(1, 3, 224, 224).astype(np.float32)}
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-encode-no-space-train-CCTVMirror-0214-0201-and-top-down-0201-and-MS-PG-small-and-SCO-and-UK-1913-select-200-little-aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_31100_mp.onnx")
