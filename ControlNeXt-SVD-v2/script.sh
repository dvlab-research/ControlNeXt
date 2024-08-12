CUDA_VISIBLE_DEVICES=0 python run_controlnext.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --output_dir outputs \
  --max_frame_num 240 \
  --guidance_scale 3 \
  --batch_frames 24 \
  --sample_stride 2 \
  --overlap 6 \
  --height 1024 \
  --width 576 \
  --controlnext_path pretrained/controlnet.bin \
  --unet_path pretrained/unet.bin \
  --validation_control_video_path examples/video/02.mp4 \
  --ref_image_path examples/ref_imgs/01.jpeg

