from src.models import utils
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np
import os
from src.datasets.common import get_dataset
from src.args import parse_arguments
import torch

ORIGINAL_MODEL_PATH = "/home/jingjie/wise-ft-clip/models/wiseft/Efficientnet_b0-20241106-lr-1e-4-add-uk-persimmon_20241017_list_corrected_val_select_265-satur0.6aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_24000.onnx"
QUANTIZED_MODEL_PATH = "/home/jingjie/wise-ft-clip/models/wiseft/Efficientnet_b0-20241106-lr-1e-4-add-uk-persimmon_20241017_list_corrected_val_select_265-satur0.6aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_24000_QT_INT8.onnx"


class CalibrationLoader(CalibrationDataReader):
    def __init__(self, pytorch_dataloader, input_name="input"):
        self.dataloader = pytorch_dataloader
        self.input_name = input_name
        self.iter = None
    
    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.dataloader)

        try:
            inputs, _ = next(self.iter)
            inputs = inputs.numpy().astype(np.float32)
            return {self.input_name: inputs}
        
        except StopIteration:
            return None

    def rewind(self):
        self.iter = None
        
def main(args):
    if args.eval_datasets == "SampleFileLoader":
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, "overall_balanced.json")
    val_dataset = get_dataset(args, is_train=False)
    val_loader = val_dataset.data_loader

    calibration_reader = CalibrationLoader(
        pytorch_dataloader = val_loader
    )

    quantize_static(
        model_input=ORIGINAL_MODEL_PATH,
        model_output=QUANTIZED_MODEL_PATH,
        calibration_data_reader=calibration_reader,
        quant_format=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False
    )




if __name__ == '__main__':
    args = parse_arguments()
    main(args)