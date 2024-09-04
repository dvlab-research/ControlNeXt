import os
import gc
import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
from safetensors.torch import load_file
from pipeline.pipeline_controlnext import StableDiffusionXLControlNeXtPipeline
from models.unet import UNet2DConditionModel
from models.controlnet import ControlNetModel
from . import utils

UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [
        5,
        10,
        20
    ],
    "block_out_channels": [
        320,
        640,
        1280
    ],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 2048,
    "cross_attention_norm": None,
    "down_block_types": [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D"
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_attention_heads": None,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "transformer_layers_per_block": [
        1,
        2,
        10
    ],
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D"
    ],
    "upcast_attention": None,
    "use_linear_projection": True
}

CONTROLNET_CONFIG = {
    'in_channels': [128, 128],
    'out_channels': [128, 256],
    'groups': [4, 8],
    'time_embed_dim': 256,
    'final_out_channels': 320,
    '_use_default_values': ['time_embed_dim', 'groups', 'in_channels', 'final_out_channels', 'out_channels']
}


def get_pipeline(
    pretrained_model_name_or_path,
    unet_model_name_or_path,
    controlnet_model_name_or_path,
    vae_model_name_or_path=None,
    lora_path=None,
    load_weight_increasement=False,
    enable_xformers_memory_efficient_attention=False,
    revision=None,
    variant=None,
    hf_cache_dir=None,
    use_safetensors=True,
    device=None,
):
    pipeline_init_kwargs = {}

    print(f"loading unet from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        # load unet from local checkpoint
        unet_sd = load_file(pretrained_model_name_or_path) if pretrained_model_name_or_path.endswith(".safetensors") else torch.load(pretrained_model_name_or_path)
        unet_sd = utils.extract_unet_state_dict(unet_sd)
        unet_sd = utils.convert_sdxl_unet_state_dict_to_diffusers(unet_sd)
        unet = UNet2DConditionModel.from_config(UNET_CONFIG)
        unet.load_state_dict(unet_sd, strict=True)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=hf_cache_dir,
            variant=variant,
            torch_dtype=torch.float16,
            use_safetensors=use_safetensors,
            subfolder="unet",
        )
    unet = unet.to(dtype=torch.float16)
    pipeline_init_kwargs["unet"] = unet

    if vae_model_name_or_path is not None:
        print(f"loading vae from {vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(vae_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)
        pipeline_init_kwargs["vae"] = vae

    if controlnet_model_name_or_path is not None:
        pipeline_init_kwargs["controlnet"] = ControlNetModel.from_config(CONTROLNET_CONFIG).to(device, dtype=torch.float32)  # init

    print(f"loading pipeline from {pretrained_model_name_or_path}")
    if os.path.isfile(pretrained_model_name_or_path):
        pipeline: StableDiffusionXLControlNeXtPipeline = StableDiffusionXLControlNeXtPipeline.from_single_file(
            pretrained_model_name_or_path,
            use_safetensors=pretrained_model_name_or_path.endswith(".safetensors"),
            local_files_only=True,
            cache_dir=hf_cache_dir,
            **pipeline_init_kwargs,
        )
    else:
        pipeline: StableDiffusionXLControlNeXtPipeline = StableDiffusionXLControlNeXtPipeline.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            use_safetensors=use_safetensors,
            cache_dir=hf_cache_dir,
            **pipeline_init_kwargs,
        )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    if unet_model_name_or_path is not None:
        print(f"loading controlnext unet from {unet_model_name_or_path}")
        pipeline.load_controlnext_unet_weights(
            unet_model_name_or_path,
            load_weight_increasement=load_weight_increasement,
            use_safetensors=True,
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )
    if controlnet_model_name_or_path is not None:
        print(f"loading controlnext controlnet from {controlnet_model_name_or_path}")
        pipeline.load_controlnext_controlnet_weights(
            controlnet_model_name_or_path,
            use_safetensors=True,
            torch_dtype=torch.float32,
            cache_dir=hf_cache_dir,
        )
    pipeline.set_progress_bar_config()
    pipeline = pipeline.to(device, dtype=torch.float16)

    if lora_path is not None:
        pipeline.load_lora_weights(lora_path)
    if enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline


def get_scheduler(
    scheduler_name,
    scheduler_config,
):
    if scheduler_name == 'Euler A':
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_name == 'UniPC':
        from diffusers.schedulers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
    elif scheduler_name == 'Euler':
        from diffusers.schedulers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_name == 'DDIM':
        from diffusers.schedulers import DDIMScheduler
        scheduler = DDIMScheduler.from_config(scheduler_config)
    elif scheduler_name == 'DDPM':
        from diffusers.schedulers import DDPMScheduler
        scheduler = DDPMScheduler.from_config(scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    return scheduler
