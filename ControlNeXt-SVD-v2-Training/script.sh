


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate  launch --config_file ./deepspeed.yaml train_svd.py \
 --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
 --output_dir= $PATH_TO_THE_SAVE_DIR \
 --dataset_type="ubc" \
 --meta_info_path=$PATH_TO_THE_META_INFO_FILE_FOR_DATASET \
 --validation_image_folder=$PATH_TO_THE_GROUND_TRUTH_DIR_FOR_EVALUATION \
 --validation_control_folder=$PATH_TO_THE_POSE_DIR_FOR_EVALUATION \
 --validation_image=$PATH_TO_THE_REFERENCE_IMAGE_FILE_FOR_EVALUATION \
 --width=576 \
 --height=1024 \
 --lr_warmup_steps 500 \
 --sample_n_frames 14 \
 --interval_frame 3 \
 --learning_rate=1e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=6000 \
 --mixed_precision="bf16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=2000 \
 --validation_steps=500 \
 --gradient_checkpointing \
 --checkpoints_total_limit 4 

# For Resume
 --controlnet_model_name_or_path $PATH_TO_THE_CONTROLNEXT_WEIGHT
 --unet_model_name_or_path $PATH_TO_THE_UNET_WEIGHT


