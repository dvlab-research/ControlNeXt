python run_controlnext.py  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --unet_model_name_or_path pretrained/vidit_depth/unet.safetensors \
    --controlnet_model_name_or_path pretrained/vidit_depth/controlnet.safetensors \
    --vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --validation_prompt "a diamond tower in the middle of a lava lake" \
    --validation_image "examples/vidit_depth/condition_0.png" \
    --output_dir "examples/vidit_depth" \
    --resolution 1024 \
    --variant fp16
