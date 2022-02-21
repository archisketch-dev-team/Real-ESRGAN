import os
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


model_name = 'finetune_RealESRGANx4plus_pairdata_ARCHI4K'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
tile = 1000

# determine model paths
model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
if not os.path.isfile(model_path):
    raise ValueError(f'Model {model_name} does not exist.')

# initialize model
# if the model_path starts with https, it will first download models to the folder: realesrgan/weights
loadnet = torch.load(model_path, map_location=torch.device('cpu'))
# prefer to use params_ema
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'
model.load_state_dict(loadnet[keyname], strict=True)
model.eval()

input_tensor = torch.rand(1, 3, 224, 224)
script_model = torch.jit.trace(model, input_tensor)
script_model.save(model_name + '.pt')



