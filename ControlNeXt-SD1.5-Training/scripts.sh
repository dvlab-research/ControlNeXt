CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 1234 train_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="checkpoints" \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --checkpoints_total_limit 3 \
 --checkpointing_steps 400 \
 --validation_steps 400 \
 --num_train_epochs 4 \
 --train_batch_size=6 \
 --controlnext_scale 0.35 \
 --save_load_weights_increaments 




CUDA_VISIBLE_DEVICES=4 python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="test" \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
 --controlnet_model_name_or_path checkpoints/checkpoint-1400/controlnext.bin \
 --unet_model_name_or_path checkpoints/checkpoint-1200/unet.bin \
 --controlnext_scale 0.35 



CUDA_VISIBLE_DEVICES=5 python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="test" \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
 --controlnet_model_name_or_path checkpoints/checkpoint-400/controlnext.bin \
 --unet_model_name_or_path checkpoints/checkpoint-400/unet_weight_increasements.bin \
 --controlnext_scale 0.35 \
 --save_load_weights_increaments 