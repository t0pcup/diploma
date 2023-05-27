import segmentation_models_pytorch as smp
import torch.onnx

from helpers import InferDataset, DataLoader, coll_fn

prefix = 'C:/diploma/backend/deep_api/'
sub_dir = ['images/source', 'images/input', 'images/output', 'models/ice.pth']
workspace, save_path, out_path, model_path = [prefix + i for i in sub_dir]
model = smp.DeepLabV3(
    encoder_name="timm-mobilenetv3_small_075",
    encoder_weights=None,
    in_channels=4,
    classes=6,
).to(torch.device('cpu'), dtype=torch.float32)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
model.load_state_dict(state_dict)
model.eval()
print(model)
# torch.onnx.export(model,  # model being run
#                   torch.zeros((16, 4, 256, 256)),  # model input (or a tuple for multiple inputs)
#                   "model_architecture13.onnx",  # where to save the model (can be a file or file-like object)
#                   export_params=True,  # store the trained parameter weights inside the model file
#                   opset_version=13,  # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],  # the model's input names
#                   output_names=['output'],  # the model's output names
#                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                 'output': {0: 'batch_size'}}
#                   )
