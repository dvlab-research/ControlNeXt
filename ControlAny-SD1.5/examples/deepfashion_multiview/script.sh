CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="admruul/anything-v3.0" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg"  \
 --validation_prompt "fischl_\(genshin_impact\), fischl_\(ein_immernachtstraum\)_\(genshin_impact\), official_alternate_costume, 1girl, eyepatch, detached_sleeves, tiara, hair_over_one_eye, bare_shoulders, purple_dress, white_thighhighs, long_sleeves, hair_ribbon, purple_ribbon, white_pantyhose"  "fischl_\(genshin_impact\), fischl_\(ein_immernachtstraum\)_\(genshin_impact\), official_alternate_costume, 1girl, eyepatch, detached_sleeves, tiara, hair_over_one_eye, bare_shoulders, purple_dress, white_thighhighs, long_sleeves, hair_ribbon, purple_ribbon, white_pantyhose" \
 --negative_prompt "PBH" "PBH"\
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \
 --lora_path lora/yuanshen/genshin_124.safetensors \
 --unet_model_name_or_path pretrained/deepfashion_multiview/unet.safetensors


CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="admruul/anything-v3.0" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg" \
 --validation_prompt "1boy, braid, single_earring, short_sleeves, white_scarf, black_gloves, alternate_costume, standing, black_shirt" "1boy, braid, single_earring, short_sleeves, white_scarf, black_gloves, alternate_costume, standing, black_shirt" \
 --negative_prompt "PBH" "PBH"\
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \
 --unet_model_name_or_path pretrained/deepfashion_multiview/unet.safetensors

CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="admruul/anything-v3.0" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg" \
 --validation_prompt "lisa_\(a_sobriquet_under_shade\)_\(genshin_impact\), lisa_\(genshin_impact\), 1girl, green_headwear, official_alternate_costume, cleavage, twin_braids, hair_flower, vision_\(genshin_impact\), large_breasts, thighlet, puffy_long_sleeves, purple_rose, beret"  "lisa_\(a_sobriquet_under_shade\)_\(genshin_impact\), lisa_\(genshin_impact\), 1girl, green_headwear, official_alternate_costume, cleavage, twin_braids, hair_flower, vision_\(genshin_impact\), large_breasts, thighlet, puffy_long_sleeves, purple_rose, beret" \
 --negative_prompt "PBH" "PBH"\
 --lora_path lora/yuanshen/genshin_124.safetensors \
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \

CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="admruul/anything-v3.0" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg" \
 --validation_prompt "fischl_\(genshin_impact\), fischl_\(ein_immernachtstraum\)_\(genshin_impact\), official_alternate_costume, 1girl, eyepatch, detached_sleeves, tiara, hair_over_one_eye, bare_shoulders, purple_dress, white_thighhighs, long_sleeves, hair_ribbon, purple_ribbon, white_pantyhose"  "fischl_\(genshin_impact\), fischl_\(ein_immernachtstraum\)_\(genshin_impact\), official_alternate_costume, 1girl, eyepatch, detached_sleeves, tiara, hair_over_one_eye, bare_shoulders, purple_dress, white_thighhighs, long_sleeves, hair_ribbon, purple_ribbon, white_pantyhose" \
 --negative_prompt "PBH" "PBH"\
 --lora_path lora/yuanshen/genshin_124.safetensors \
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \
 --unet_model_name_or_path pretrained/deepfashion_multiview/unet.safetensors


# base generation
CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="Lykon/DreamShaper" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg"\
 --validation_prompt "a woman in white shorts and a tank top"  "a woman wearing a black shirt and black leather skirt" \
 --negative_prompt "PBH" "PBH" \
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \
 --unet_model_name_or_path pretrained/deepfashion_multiview/unet.safetensors


# Combine with LoRA
CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg"\
 --negative_prompt "PBH" "PBH" \
 --validation_prompt "c1bo, a woman, Armor, weapon, beautiful"  "c1bo, a man, fight" \
 --lora_path lora/c1bo/cyborg_v_2_SD15.safetensors \
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \
 --unet_model_name_or_path pretrained/deepfashion_multiview/unet.safetensors

# Combine with LoRA, without our control
CUDA_VISIBLE_DEVICES=0 python run_controlany.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="examples/deepfashion_multiview" \
 --validation_image "examples/deepfashion_multiview/condition_0.jpg"  "examples/deepfashion_multiview/condition_1.jpg" \
 --negative_prompt "PBH" "PBH" \
 --validation_prompt "c1bo, a woman, Armor, weapon, beautiful"  "c1bo, a man, fight" \
 --lora_path lora/c1bo/cyborg_v_2_SD15.safetensors


