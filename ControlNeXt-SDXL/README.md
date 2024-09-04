# 🌀 ControlNeXt-SDXL

This is our implementation of ControlNeXt based on [Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0).

> Please refer to [Examples](#examples) for further intuitive details.\
> Please refer to [Inference](#inference) for more details regarding installation and inference.\

Our method demonstrates the advantages listed below:

- **Few trainable parameters**: only requiring **5~200M** trainable parameters.
- **Fast training speed**: reduce sudden convergence.
- **Efficient**: no need for additional brunch; only a lightweight module is required.
- **Compatibility**: can serve as a **plug-and-play** lightweight module and can be combined with other LoRA weights.

# Examples

The demo examples are generated using the ControlNeXt trained on

- (i) our vidit_depth dataset with utilizing [Stable Diffusion XL 1.0 Base](stabilityai/stable-diffusion-xl-base-1.0) as the base model.
- (ii) our anime_canny dataset with utilizing [Neta Art XL 2.0](https://civitai.com/models/410737/neta-art-xl) as the base model.

Our method demonstrates excellent compatibility and can be applied to most other models based on SDXL1.0 architecture and LoRA. And you can retrain your own model for better performance.

<p align="center">
  <img src="examples/demo/demo1.jpg" width="70%" alt="demo1">
  <img src="examples/demo/demo2.jpg" width="70%" alt="demo2">
  <img src="examples/demo/demo3.jpg" width="70%" alt="demo3">
  <img src="examples/demo/demo5.jpg" width="70%" alt="demo5">
</p>

## BaseModel

Our model can be applied to various base models without the need for futher training as a plug-and-play module.

> 📌 Of course, you can retrain your owm model, especially for complex tasks and to achieve better performance.

- [Stable Diffusion XL 1.0 Base](stabilityai/stable-diffusion-xl-base-1.0)

<p align="center">
  <img src="examples/vidit_depth/eval_img/StableDiffusionXL.jpg" width="70%" alt="StableDiffusionXL">
</p>

- [AAM XL](https://huggingface.co/Lykon/AAM_XL_AnimeMix)

<p align="center">
  <img src="examples/anime_canny/eval_img/AAM.jpg" width="70%" alt="AAM">
</p>

- [Neta XL V2](https://civitai.com/models/410737/neta-art-xl)

<p align="center">
  <img src="examples/anime_canny/eval_img/NetaXLV2.jpg" width="70%" alt="NetaXLV2">
</p>

## LoRA

Our model can also be directly combined with other publicly available LoRA weights.

- [Glass Sculptures](https://civitai.com/models/11203/glass-sculptures?modelVersionId=177888)

<p align="center">
  <img src="examples/vidit_depth/eval_img/StableDiffusionXL_GlassSculpturesLora.jpg" width="70%" alt="StableDiffusionXL">
</p>

# Inference

## Quick Start

Clone the repository:

```bash
git clone https://github.com/dvlab-research/ControlNeXt
cd ControlNeXt/ControlNeXt-SDXL
```

Install the required packages:

```bash
pip install -r requirements.txt
```

(Optional) Download the LoRA weight, such as [Amiya (Arknights) Fresh Art Style](https://civitai.com/models/231598/amiya-arknights-fresh-art-style-xl-trained-with-6k-images). And put them under `lora/`.

Run the example:

```bash
bash examples/anime_canny/run.sh
```

## Usage

### Canny Condition

```python
# examples/anime_canny/run.sh
python run_controlnext.py --pretrained_model_name_or_path "neta-art/neta-xl-2.0" \
  --unet_model_name_or_path "Eugeoter/controlnext-sdxl-anime-canny" \
  --controlnet_model_name_or_path "Eugeoter/controlnext-sdxl-anime-canny" \
  --controlnet_scale 1.0 \ # controlnet scale factor used to adjust the strength of the control condition
  --vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
  --validation_prompt "3d style, photorealistic style, 1girl, arknights, amiya (arknights), solo, white background, upper body, looking at viewer, blush, closed mouth, low ponytail, black jacket, hooded jacket, open jacket, hood down, blue neckwear" \
  --negative_prompt "worst quality, abstract, clumsy pose, deformed hand, fused fingers, extra digits, fewer digits, fewer fingers, extra fingers, extra arm, missing arm, extra leg, missing leg, signature, artist name, multi views, disfigured, ugly" \
  --validation_image "examples/anime_canny/condition_0.png" \ # input canny image
  --output_dir "examples/anime_canny" \
  --load_weight_increasement # load weight increasement
```

We use a `controlnet_scale` factor to adjust the strength of the control condition.

We recommend to only save & load the weights difference of the UNet's trainable parameters, i.e., $\Delta W = W_{finetune} - W_{pretrained}$, rather than the actual weight.
This is useful when adapting to various base models since the weights difference is model-agnostic.

### Depth Condition

```python
# examples/vidit_depth/run.sh
python run_controlnext.py  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --unet_model_name_or_path "Eugeoter/controlnext-sdxl-vidit-depth" \
    --controlnet_model_name_or_path "Eugeoter/controlnext-sdxl-vidit-depth" \
    --controlnet_scale 1.0 \
    --vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --validation_prompt "a diamond tower in the middle of a lava lake" \
    --validation_image "examples/vidit_depth/condition_0.png" \ # input depth image
    --output_dir "examples/vidit_depth" \
    --width 1024 \
    --height 1024 \
    --load_weight_increasement \
    --variant fp16
```

## Run with Image Processor

We also provide a simple image processor to help you automatically convert the image to the control condition, such as canny.

```python
# examples/anime_canny/run_with_pp.sh
python run_controlnext.py --pretrained_model_name_or_path "neta-art/neta-xl-2.0" \
  --unet_model_name_or_path "Eugeoter/controlnext-sdxl-anime-canny" \
  --controlnet_model_name_or_path "Eugeoter/controlnext-sdxl-anime-canny" \
  --controlnet_scale 1.0 \
  --vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
  --validation_prompt "3d style, photorealistic style, 1girl, arknights, amiya (arknights), solo, white background, upper body, looking at viewer, blush, closed mouth, low ponytail, black jacket, hooded jacket, open jacket, hood down, blue neckwear" \
  --negative_prompt "worst quality, abstract, clumsy pose, deformed hand, fused fingers, extra digits, fewer digits, fewer fingers, extra fingers, extra arm, missing arm, extra leg, missing leg, signature, artist name, multi views, disfigured, ugly" \
  --validation_image "examples/anime_canny/image_0.png" \ # input image (not canny)
  --validation_image_processor "canny" \ # preprocess `validation_image` to canny condition
  --output_dir "examples/anime_canny" \
  --load_weight_increasement
```
