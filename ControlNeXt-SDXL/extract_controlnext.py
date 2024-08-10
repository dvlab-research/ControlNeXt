import os
import torch
import argparse
from safetensors.torch import load_file, save_file
from models.unet import UNet2DConditionModel, UNET_CONFIG
from utils import utils


def extract_controlnext(args):
    print(f"loading unet from {args.pretrained_model_name_or_path}")
    if os.path.isfile(args.pretrained_model_name_or_path):
        unet_sd = load_file(args.pretrained_model_name_or_path)
        unet_sd = utils.extract_unet_state_dict(unet_sd)
        unet_sd = utils.convert_sdxl_unet_state_dict_to_diffusers(unet_sd)
        unet = UNet2DConditionModel.from_config(UNET_CONFIG)
        unet.load_state_dict(unet_sd, strict=True)
        if args.variant == "fp16":
            unet = unet.half()
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            subfolder="unet",
            # use_safetensors=True,
            cache_dir=args.hf_cache_dir,
            torch_dtype=torch.float16 if args.variant == "fp16" else None,
        )
    utils.log_model_info(unet, "unet")

    print(f"loading controlnext unet from {args.unet_model_name_or_path}")
    controlnext_unet_sd = load_file(args.unet_model_name_or_path)
    controlnext_unet_sd = utils.convert_to_controlnext_unet_state_dict(controlnext_unet_sd)
    utils.log_model_info(controlnext_unet_sd, "controlnext unet")

    unet_sd = unet.state_dict()
    unet_sd_diff = {}
    for k in controlnext_unet_sd.keys():
        unet_sd_diff[k] = controlnext_unet_sd[k] - unet_sd[k]
    utils.log_model_info(unet_sd_diff, "controlnext unet diff")

    save_path = f"{os.path.splitext(args.unet_model_name_or_path)[0]}_diff.safetensors"
    save_file(unet_sd_diff, save_path)
    print(f"saved: {save_path}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or subset"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Path to the cache directory for huggingface datasets and models.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    extract_controlnext(args)
