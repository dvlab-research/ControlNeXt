# 🌀 ControlNeXt-SDXL

This is our **training** demo of ControlNeXt based on [Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0).

Hardware requirement: A single GPU with at least 20GB memory.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/dvlab-research/ControlNeXt
cd ControlNeXt/ControlNeXt-SDXL-Training
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
bash examples/vidit_depth/train.sh
```

The output will be saved in `train/example`.

## Usage

We recommend to only save & load the weights difference of the UNet's trainable parameters, i.e., $\Delta W = W_{finetune} - W_{pretrained}$, rather than the actual weight.
This is useful when adapting to various base models since the weights difference is model-agnostic.

```python
accelerate launch train_controlnext.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--variant fp16 \
--use_safetensors \
--output_dir "train/example" \
--logging_dir "logs" \
--resolution 1024 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--controlnet_scale_factor 1.0 \ # the strength of the controlnet output. For depth, we recommend 1.0, and for canny, we recommend 0.35
--save_weights_increaments \
--mixed_precision fp16 \
--enable_xformers_memory_efficient_attention \
--dataset_name "Nahrawy/VIDIT-Depth-ControlNet" \
--image_column "image" \
--conditioning_image_column "depth_map" \
--caption_column "caption" \
--validation_prompt "a stone tower on a rocky island" \
--validation_image "examples/vidit_depth/condition_0.png"
```
