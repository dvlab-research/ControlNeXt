python run_controlnext.py  --pretrained_model_name_or_path "Lykon/AAM_XL_AnimeMix" \
    --unet_model_name_or_path pretrained/anime_canny/unet.safetensors \
    --controlnet_model_name_or_path pretrained/anime_canny/controlnet.safetensors \
    --vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --validation_prompt "3d style, photorealistic style, 1girl, arknights, amiya (arknights), solo, white background, upper body, looking at viewer, blush, closed mouth, low ponytail, black jacket, hooded jacket, open jacket, hood down, blue neckwear" \
    --negative_prompt "worst quality, abstract, clumsy pose, deformed hand, fused fingers, extra digits, fewer digits, fewer fingers, extra fingers, extra arm, missing arm, extra leg, missing leg, signature, artist name, multi views, disfigured, ugly" \
    --validation_image "examples/anime_canny/condition_0.png" \
    --output_dir "examples/anime_canny" \
    --load_weight_increasement \
    --variant fp16
