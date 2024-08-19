

import argparse
import torch
import os
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNetModel
from safetensors.torch import save_file, load_file


"""
python -m utils.extract_learned_paras \
    /home/llm/bhpeng/generation/svd-temporal-controlnet/outputs_sdxt_mid_upblocks/ft_on_duqi_after_hands500_checkpoint-200/unet/unet_fp16.bin \
    /home/llm/bhpeng/generation/svd-temporal-controlnet/outputs_sdxt_mid_upblocks/ft_on_duqi_after_hands500_checkpoint-200/unet/unet_fp16_increase.bin \
    --pretrained_path /home/llm/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/snapshots/a423ba0d3e1a94a57ebc68e98691c43104198394
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("src_path",
                        type=str,
                        help="path to the video")
    parser.add_argument("dst_path",
                        type=str,
                        help="path to the save_dict")
    parser.add_argument("--pretrained_path",
                        type=str,
                        default="/home/llm/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/ae8391f7321be9ff8941508123715417da827aa4")
    parser.add_argument("--save_as_fp32",
                        action="store_true",)
    parser.add_argument("--save_weight_increase",
                        action="store_true",)
    args = parser.parse_args()

    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    pretrained_state_dict = unet.state_dict()
    
    if os.path.splitext(args.src_path)[1] == ".bin":
        src_state_dict = torch.load(args.src_path)
    elif os.path.splitext(args.src_path)[1] == ".safetensors":
        src_state_dict = load_file(args.src_path)
        
    for k in list(src_state_dict.keys()):
        src_state_dict[k] = src_state_dict[k].to(pretrained_state_dict[k])
        if torch.allclose(src_state_dict[k], pretrained_state_dict[k]):
            src_state_dict.pop(k)
            continue
        if args.save_weight_increase:
            src_state_dict[k] = src_state_dict[k] - pretrained_state_dict[k]
        if not args.save_as_fp32:
            src_state_dict[k] = src_state_dict[k].half()
                
    
    if os.path.splitext(args.dst_path)[1] == ".bin":
        torch.save(src_state_dict, args.dst_path)
    elif os.path.splitext(args.dst_path)[1] == ".safetensors":
        save_file(src_state_dict, args.dst_path)
