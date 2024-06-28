CUDA_VISIBLE_DEVICES=0 python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="examples/vidit_depth" \
 --validation_image "examples/vidit_depth/condition_0.png"  \
 --validation_prompt "a wooden bridge in the middle of a field"  \
 --controlnet_model_name_or_path pretrained/vidit_depth/controlnet.safetensors \
 --unet_model_name_or_path pretrained/vidit_depth/unet.safetensors