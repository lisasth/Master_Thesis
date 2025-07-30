import os
from src.datasets.common import get_dataset
from src.args import parse_arguments
import nncf
import onnx


ORIGINAL_MODEL_PATH = "/home/jingjie/wise-ft-clip/models/wiseft/MobileNetV2-quantizable-class_2024_10_17_uk_hersham_contrastive-20250401/fnv_clip_wise_ft_15000.onnx"
QUANTIZED_MODEL_PATH = "/home/jingjie/wise-ft-clip/models/wiseft/MobileNetV2-quantizable-class_2024_10_17_uk_hersham_contrastive-20250401/fnv_clip_wise_ft_15000_openvino_QT_INT8.onnx"


def transform_fn(data_item, input_name="input"):
    images, _ = data_item
    return {input_name: images.numpy()}
        
def main(args):
    if args.eval_datasets == "SampleFileLoader":
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, "overall_balanced.json")
    val_dataset = get_dataset(args, is_train=False)
    val_loader = val_dataset.data_loader
    model = onnx.load(ORIGINAL_MODEL_PATH)

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    onnx_quantized_model = nncf.quantize(model, calibration_dataset)
    onnx.save(onnx_quantized_model, QUANTIZED_MODEL_PATH)



if __name__ == '__main__':
    args = parse_arguments()
    main(args)