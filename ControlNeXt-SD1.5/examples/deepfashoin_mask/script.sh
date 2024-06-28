CUDA_VISIBLE_DEVICES=0 python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="examples/deepfashoin_mask" \
 --validation_image "examples/deepfashoin_mask/condition_0.png"  \
 --validation_prompt "a woman in white shorts and a tank top"  \
 --controlnet_model_name_or_path pretrained/deepfashoin_mask/controlnet.safetensors \
 --unet_model_name_or_path pretrained/deepfashoin_mask/unet.safetensors