# 🌀 ControlAny-SDXL

This is our implementation of ControlAny based on [Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0).

> Please refer to [Examples](#examples) for further intuitive details.\
> Please refer to [Inference](#inference) for more details regarding installation and inference.\

Our method demonstrates the advantages listed below:

- **Few trainable parameters**: only requiring **5~200M** trainable parameters (occupying 20~80 MB of memory).
- **Fast training speed**: no sudden convergence.
- **Efficient**: no need for additional brunch; only a lightweight module is required.
- **Compatibility**: can serve as a **plug-and-play** lightweight module and can be combined with other LoRA weights.

# Examples

The demo examples are generated using the ControlAny trained on vidit_depth dataset with utilizing [DreamShaperXL](https://huggingface.co/Lykon/dreamshaper-xl-1-0) as the base model. Our method demonstrates excellent compatibility and can be applied to most other models based on sd1.5 architecture and LoRA. And you can retrain your own model for better performance.

## BaseModel

Our model can be applied to various base models without the need for futher training as a plug-and-play module.

> 📌 Of course, you can retrain your owm model, especially for complex tasks and to achieve better performance.

- [Stable Diffusion XL 1.0 Base](stabilityai/stable-diffusion-xl-base-1.0)

<p align="center">
  <img src="examples/vidit_depth/eval_img/StableDiffusionXL.jpg" width="70%" alt="StableDiffusionXL">
</p>

- [DreamShaperXL](https://huggingface.co/Lykon/dreamshaper-xl-1-0)

<p align="center">
  <img src="examples/vidit_depth/eval_img/DreamShaper.jpg" width="70%" alt="DreamShaper">
</p>

## LoRA

Our model can also be directly combined with other publicly available LoRA weights.

- [Glass Sculptures](https://civitai.com/models/11203/glass-sculptures?modelVersionId=177888)

<p align="center">
  <img src="examples/vidit_depth/eval_img/StableDiffusionXL_glass.jpg" width="70%" alt="StableDiffusionXL">
</p>

# Inference

1. Clone our repository
2. `cd ControlAny-SDXL`
3. Download the pretrained weight into `pretrained/` from [here](https://huggingface.co/Pbihao/ControlAny/tree/main/ControlAny-SDXL).
4. (Optional) Download the LoRA weight, such as [Glass Sculptures](https://civitai.com/models/11203/glass-sculptures?modelVersionId=177888). And put them under `lora/`
5. Run the scipt

```python
CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
 --unet_model_name_or_path "pretrained/vidit_depth/unet.safetensors" \
 --controlnet_model_name_or_path "pretrained/vidit_depth/controlnet.safetensors" \
 --output_dir="examples/vidit_depth" \
 --validation_image "examples/vidit_depth/condition_0.jpg"  \
 --validation_prompt "a translucent diamond tower in the middle of a lava lake" \
 (Optional)--lora_path lora/sdxl_glass.sagetensors \
```

> --pretrained_model_name_or_path : pretrained base model, we try on [Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0) \
> --controlnet_model_name_or_path : the model path of controlnet (a light weight module) \
> --lora_path : downloaded other LoRA weight \
> --unet_model_name_or_path : the model path of a subset of unet parameters

> 📌 In most cases, it is enough to just load the control module by `--controlnet_model_name_or_path`. However, sometime the task is hard so it is need to select some subset of the original unet parameters to fit the task (Can be seen as another kind of LoRA). \
> More parameters mean weaker generality, so you can make your own tradeoff. Or directly train your own models based on your own data. The training is also fast.

# TODO
