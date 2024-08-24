import os
import gc
import torch
from torch import nn
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from safetensors.torch import load_file
from pipeline.pipeline_controlnext import StableDiffusionXLControlNeXtPipeline
from models.unet import UNet2DConditionModel, UNET_CONFIG
from models.controlnet import ControlNetModel
from . import utils


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

    if controlnet_model_name_or_path is not None:
        print(f"loading controlnet from {controlnet_model_name_or_path}")
        controlnet = ControlNetModel()
        if controlnet_model_name_or_path is not None:
            utils.load_safetensors(controlnet, controlnet_model_name_or_path)
        else:
            controlnet.scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        controlnet.to(device, dtype=torch.float32)
        pipeline_init_kwargs["controlnet"] = controlnet

        utils.log_model_info(controlnet, "controlnext")
    else:
        print(f"no controlnet")

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
    utils.log_model_info(unet, "unet")

    if unet_model_name_or_path is not None:
        print(f"loading controlnext unet from {unet_model_name_or_path}")
        controlnext_unet_sd = load_file(unet_model_name_or_path)
        controlnext_unet_sd = utils.convert_to_controlnext_unet_state_dict(controlnext_unet_sd)
        unet_sd = unet.state_dict()
        assert all(
            k in unet_sd for k in controlnext_unet_sd), \
            f"controlnext unet state dict is not compatible with unet state dict, missing keys: {set(controlnext_unet_sd.keys()) - set(unet_sd.keys())}, extra keys: {set(unet_sd.keys()) - set(controlnext_unet_sd.keys())}"
        if load_weight_increasement:
            print("loading weight increasement")
            for k in controlnext_unet_sd.keys():
                controlnext_unet_sd[k] = controlnext_unet_sd[k] + unet_sd[k]
        unet.load_state_dict(controlnext_unet_sd, strict=False)
        utils.log_model_info(controlnext_unet_sd, "controlnext unet")

    pipeline_init_kwargs["unet"] = unet

    if vae_model_name_or_path is not None:
        print(f"loading vae from {vae_model_name_or_path}")
        vae = AutoencoderKL.from_pretrained(vae_model_name_or_path, cache_dir=hf_cache_dir, torch_dtype=torch.float16).to(device)
        pipeline_init_kwargs["vae"] = vae

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
