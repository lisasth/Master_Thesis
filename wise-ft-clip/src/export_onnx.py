import os
import torch
from src.models.modeling import ImageClassifier
from src.args import parse_arguments
from src.models import utils
from src.train_classification_head_on_clip_backbone import MLPHead


def export_model(export_path, model, x, epoch, input_name="input", output_name="output", model_type="clip_wise_ft"):
    os.makedirs(export_path, exist_ok=True)
    print(f"save onnx model in {export_path}")
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      os.path.join(export_path, f"fnv_{model_type}_{epoch}.onnx"),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=[input_name],  # the model's input names
                      output_names=[output_name],  # the model's output names
                      dynamic_axes={input_name: {0: 'batch_size'}})


def load_and_export_model(args):
    assert args.save is not None, 'Please provide a path to store models'

    finetuned_checkpoint = args.load

    # Load models
    finetuned = utils.torch_load(finetuned_checkpoint)

    export_model(args.save, finetuned, torch.rand((1, 3, 224, 224)), epoch=args.epochs, model_type=args.model_type)


if __name__ == '__main__':
    args = parse_arguments()
    load_and_export_model(args)
