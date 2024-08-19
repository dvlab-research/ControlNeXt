

import argparse
import torch
import os
from collections import OrderedDict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    args = parser.parse_args()
    state_dict = torch.load(os.path.join(args.checkpoint_dir, "pytorch_model.bin"))

    unet_state = OrderedDict()
    controlnet_state = OrderedDict()
    for name, data in state_dict.items():
        model = name.split('.', 1)[0]
        module = name.split('.', 1)[1]
        if model == 'unet':
            unet_state[module] = data
        elif model == 'controlnet':
            controlnet_state[module] = data
    
    for model in ['unet', 'controlnet']:
        if not os.path.exists(os.path.join(args.checkpoint_dir, model)):
            os.makedirs(os.path.join(args.checkpoint_dir, model))


    torch.save(unet_state, os.path.join(args.checkpoint_dir, "unet", "diffusion_pytorch_model.bin"))
    torch.save(controlnet_state, os.path.join(args.checkpoint_dir, "controlnet", "diffusion_pytorch_model.bin"))