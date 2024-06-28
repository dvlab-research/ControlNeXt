import os
import torch
import time
import cv2
import numpy as np
from PIL import Image
import argparse
from safetensors.torch import load_file
import torch.nn as nn

from models.unet import UNet2DConditionModel
from models.controlnet import ControlNetModel
from models.pipeline_controlnext import StableDiffusionControlNextPipeline
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from transformers import AutoTokenizer, PretrainedConfig

def log_validation(
    vae, 
    text_encoder, 
    tokenizer, 
    unet, 
    controlnet, 
    args, 
    device='cuda'
):

    pipeline = StableDiffusionControlNextPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config()
    if args.lora_path is not None:
        pipeline.load_lora_weights(args.lora_path)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    if args.negative_prompt is not None:
        negative_prompts = args.negative_prompt
        assert len(validation_prompts) == len(validation_prompts)
    else:
        negative_prompts = None

    image_logs = []
    inference_ctx = torch.autocast(device)

    for i, (validation_prompt, validation_image) in enumerate(zip(validation_prompts, validation_images)):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []
        negative_prompt = negative_prompts[i] if negative_prompts is not None else None

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator, negative_prompt=negative_prompt
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    save_dir_path = os.path.join(args.output_dir, "eval_img")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    for i, log in enumerate(image_logs):
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        validation_image = log["validation_image"]

        formatted_images = []
        formatted_images.append(np.asarray(validation_image))
        for image in images:
            formatted_images.append(np.asarray(image))
        formatted_images = np.concatenate(formatted_images, 1)

        file_path = os.path.join(save_dir_path, "image_{}.png".format(i))
        formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_BGR2RGB)
        print("Save images to:", file_path)
        cv2.imwrite(file_path, formatted_images)

    return image_logs
    


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
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or subset"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to lora"
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
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--load_weight_increasement",
        action="store_true",
        help="Only load weight increasement",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        state_dict = load_file(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    args = parse_args()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        revision=args.revision, 
        variant=args.variant
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, 
        args.revision
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        revision=args.revision, 
        variant=args.variant
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )


    controlnet = ControlNetModel()
    if args.controlnet_model_name_or_path is not None:
        load_safetensors(controlnet, args.controlnet_model_name_or_path)
    else:
        controlnet.scale = nn.Parameter(torch.tensor(0.), requires_grad=False)


    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        variant=args.variant
    )
    if args.unet_model_name_or_path is not None:
        load_safetensors(unet, args.unet_model_name_or_path, strict=False, load_weight_increasement=args.load_weight_increasement)


    log_validation(
        vae=vae, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        unet=unet, 
        controlnet=controlnet, 
        args=args, 
    )