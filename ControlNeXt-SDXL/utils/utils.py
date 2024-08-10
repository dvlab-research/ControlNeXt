from safetensors.torch import load_file
from transformers import PretrainedConfig


def count_num_parameters_of_safetensors_model(safetensors_path):
    state_dict = load_file(safetensors_path)
    return sum(p.numel() for p in state_dict.values())


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


def fix_clip_text_encoder_position_ids(text_encoder):
    if hasattr(text_encoder.text_model.embeddings, "position_ids"):
        text_encoder.text_model.embeddings.position_ids = text_encoder.text_model.embeddings.position_ids.long()


def load_controlnext_unet_state_dict(unet_sd, controlnext_unet_sd):
    assert all(
        k in unet_sd for k in controlnext_unet_sd), f"controlnext unet state dict is not compatible with unet state dict, missing keys: {set(controlnext_unet_sd.keys()) - set(unet_sd.keys())}, extra keys: {set(unet_sd.keys()) - set(controlnext_unet_sd.keys())}"
    for k in controlnext_unet_sd.keys():
        unet_sd[k] = controlnext_unet_sd[k]
    return unet_sd


def convert_to_controlnext_unet_state_dict(state_dict):
    import re
    pattern = re.compile(r'.*attn2.*to_out.*')
    state_dict = {k: v for k, v in state_dict.items() if pattern.match(k)}
    # state_dict = extract_unet_state_dict(state_dict)
    if is_sdxl_state_dict(state_dict):
        state_dict = convert_sdxl_unet_state_dict_to_diffusers(state_dict)
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


def log_model_info(model, name):
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    print(
        f"{name}:",
        f"  number of parameters: {sum(p.numel() for p in sd.values())}",
        f"  dtype: {sd[next(iter(sd))].dtype}",
        sep='\n'
    )
