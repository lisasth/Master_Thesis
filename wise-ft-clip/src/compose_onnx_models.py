import onnx
from onnx import compose
from onnx.compose import merge_models

model = onnx.load('/home/jingjie/wise-ft-clip/models/wiseft/ViTB16-20240923_include_UK_figs_20240917_list_corrected_val_select_265_train_select_349_v1_less_laser_more_figs_20240923_v2-satur0.6aug-RandomResizedCrop-from-0.5-lr1e-5-encode-fnv-template-wd-1e-4-contrastive/fnv_clip_wise_ft_3400_qt_float16.onnx')
prep = onnx.load('/home/jingjie/wise-ft-clip/prep_convert_rgb.onnx')

# add prefix, resolve names conflits
prep_with_prefix = compose.add_prefix(prep, prefix="prep_")

model_prep = compose.merge_models(
    prep_with_prefix,
    model,    
    io_map=[('prep_output', # output prep model
             'input')])     # input classifcation model

onnx.save_model(model_prep, 'class_model_with_prep_0923_convert_rgb.onnx')