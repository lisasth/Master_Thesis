import torch
import torchvision.transforms as transforms

class TransposeResizeNormalize(torch.nn.Module):
    def __init__(
        self, 
        size, 
        mean=[123, 117, 104], 
        std=[69, 67, 70],
        rgb=False
    ):
        super(TransposeResizeNormalize, self).__init__()
        self.resize = transforms.Resize(
            (size, size),
            antialias=True,
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.normalize = transforms.Normalize(
            mean=mean,
            std=std
        )
        self.rgb = rgb

    def forward(self, x):
        if not self.rgb:
            x = torch.flip(x, dims=[-1])
        x = x.permute(0,3,1,2)
        x = self.resize(x)
        x = self.normalize(x)
        return x

model_prep = TransposeResizeNormalize(size=224)

#                         N, H, W, C
dummy_input = torch.randn(1, 123, 555, 3)

dynamic = {'input': {0: 'batch', 1: 'height', 2: 'width'},
            'output': {0 : 'batch'}}

path_export_model_prep = 'prep_convert_rgb.onnx'

torch.onnx.export(model_prep,
                  dummy_input,
                  path_export_model_prep,
                  opset_version=14,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names=['output'],
                  dynamic_axes=dynamic,
                  verbose=True)