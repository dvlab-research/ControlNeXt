import os
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import gc
from safetensors.torch import load_file
from models.pipeline_controlnext import StableDiffusionXLControlNeXtPipeline
from models.unet import ControlNeXtUNet2DConditionModel
from models.controlnet import ControlNetModel
from diffusers import UniPCMultistepScheduler, StableDiffusionXLPipeline, AutoencoderKL
from transformers import PretrainedConfig


def log_validation(
    args,
    device='cuda'
):
    pipeline_init_kwargs = {}
    if args.controlnet_model_name_or_path is not None:
        print(f"loading controlnet from {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, cache_dir=args.hf_cache_dir).to(device, dtype=torch.float32)
        pipeline_init_kwargs["controlnet"] = controlnet
    unet = ControlNeXtUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.hf_cache_dir,
    )
    if args.unet_model_name_or_path is not None:
        print(f"loading unet from {args.unet_model_name_or_path}")
        controlnext_unet_sd = load_file(args.unet_model_name_or_path)
        if args.load_weight_increasement:
            unet_sd = load_controlnext_unet_state_dict(unet.state_dict(), controlnext_unet_sd)
        else:
            unet_sd = convert_to_controlnext_unet_state_dict(controlnext_unet_sd)
        unet.load_state_dict(unet_sd, strict=False)
    if args.vae_model_name_or_path is not None:
        print(f"loading vae from {args.vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, cache_dir=args.hf_cache_dir).to(device)
        pipeline_init_kwargs["vae"] = vae
    print(f"loading pipeline from {args.pretrained_model_name_or_path}")
    pipeline: StableDiffusionXLControlNeXtPipeline = StableDiffusionXLControlNeXtPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.hf_cache_dir,
        **pipeline_init_kwargs,
    ).to(device, dtype=torch.float16)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
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
                image = pipeline.__call__(
                    prompt=validation_prompt, controlnet_image=validation_image, num_inference_steps=20, generator=generator, negative_prompt=negative_prompt
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
        "--vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae model or subset"
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


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = None
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, revision=revision, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_safetensors(model, safetensors_path, strict=True, device="cuda", load_weight_increasement=False):
    if not load_weight_increasement:
        state_dict = load_file(safetensors_path, device=device)
        model.load_state_dict(state_dict, strict=strict)
    else:
        state_dict = load_file(safetensors_path, device=device)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)


def fix_clip_text_encoder_position_ids(text_encoder):
    if hasattr(text_encoder.text_model.embeddings, "position_ids"):
        text_encoder.text_model.embeddings.position_ids = text_encoder.text_model.embeddings.position_ids.long()


def convert_unet_to_controlnext_unet(orig_unet):
    print(f"converting unet to controlnext unet")
    unet = ControlNeXtUNet2DConditionModel.from_config(orig_unet.config)
    unet.load_state_dict(orig_unet.state_dict())
    unet = unet.to(orig_unet.device, dtype=orig_unet.dtype)
    del orig_unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return unet


def load_controlnext_unet_state_dict(unet_sd, controlnext_unet_sd):
    assert all(k in unet_sd for k in controlnext_unet_sd), "controlnext unet state dict is not compatible with unet state dict"
    for k in controlnext_unet_sd.keys():
        unet_sd[k] = controlnext_unet_sd[k]
    return unet_sd


def convert_to_controlnext_unet_state_dict(state_dict):
    if contains_unet_keys(state_dict):
        state_dict = extract_unet_state_dict(state_dict)
    if is_sdxl_state_dict(state_dict):
        state_dict = convert_sdxl_unet_state_dict_to_diffusers(state_dict)
    state_dict = {k: v for k, v in state_dict.items() if 'to_out' in k}
    return state_dict


def make_unet_conversion_map():
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


def convert_unet_state_dict(src_sd, conversion_map):
    converted_sd = {}
    for src_key, value in src_sd.items():
        src_key_fragments = src_key.split(".")[:-1]  # remove weight/bias
        while len(src_key_fragments) > 0:
            src_key_prefix = ".".join(src_key_fragments) + "."
            if src_key_prefix in conversion_map:
                converted_prefix = conversion_map[src_key_prefix]
                converted_key = converted_prefix + src_key[len(src_key_prefix):]
                converted_sd[converted_key] = value
                break
            src_key_fragments.pop(-1)
        assert len(src_key_fragments) > 0, f"key {src_key} not found in conversion map"

    return converted_sd


def convert_sdxl_unet_state_dict_to_diffusers(sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_dict = {sd: hf for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(sd, conversion_dict)


def convert_diffusers_unet_state_dict_to_sdxl(du_sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_map = {hf: sd for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(du_sd, conversion_map)


def extract_unet_state_dict(state_dict):
    unet_sd = {}
    UNET_KEY_PREFIX = "model.diffusion_model."
    for k, v in state_dict.items():
        if k.startswith(UNET_KEY_PREFIX):
            unet_sd[k[len(UNET_KEY_PREFIX):]] = v
    return unet_sd


def is_sdxl_state_dict(state_dict):
    return any(key.startswith('input_blocks') for key in state_dict.keys())


def contains_unet_keys(state_dict):
    UNET_KEY_PREFIX = "model.diffusion_model."
    return any(k.startswith(UNET_KEY_PREFIX) for k in state_dict.keys())


def convert_controlnext_unet_state_dict_to_unet_state_dict(controlnext_unet_sd, unet_sd, weight_increasement):
    if contains_unet_keys(controlnext_unet_sd):
        controlnext_unet_sd = extract_unet_state_dict(controlnext_unet_sd)
    if is_sdxl_state_dict(controlnext_unet_sd):
        controlnext_unet_sd = convert_sdxl_unet_state_dict_to_diffusers(controlnext_unet_sd)
    if weight_increasement:
        for k in controlnext_unet_sd.keys():
            controlnext_unet_sd[k] = unet_sd[k] + controlnext_unet_sd[k]
    return controlnext_unet_sd


if __name__ == "__main__":
    args = parse_args()
    log_validation(args)
