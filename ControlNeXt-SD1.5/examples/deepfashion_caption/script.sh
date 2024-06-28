CUDA_VISIBLE_DEVICES=0 python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="examples/deepfashion_caption" \
 --validation_image "examples/deepfashion_caption/condition_1.png" "examples/deepfashion_caption/condition_0.png" \
 --validation_prompt "a woman wearing a black shirt and black leather skirt" "levi's women's white graphic t - shirt"  \
 --controlnet_model_name_or_path pretrained/deepfashion_caption/controlnet.safetensors \
 --unet_model_name_or_path pretrained/deepfashion_caption/unet.safetensors
