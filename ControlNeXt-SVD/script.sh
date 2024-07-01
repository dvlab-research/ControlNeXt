

python run_controlnext.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --validation_control_video_path examples/pose/pose.mp4 \
  --output_dir outputs/tiktok \
  --controlnext_path pretrained/controlnet.bin \
  --unet_path pretrained/unet_fp16.bin \
  --ref_image_path examples/ref_imgs/tiktok.png
  

python run_controlnext.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --validation_control_video_path examples/pose/pose.mp4 \
  --output_dir outputs/spiderman \
  --controlnext_path pretrained/controlnet.bin \
  --unet_path pretrained/unet_fp16.bin \
  --ref_image_path examples/ref_imgs/spiderman.jpg