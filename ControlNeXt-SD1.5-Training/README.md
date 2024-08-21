# 🌀 ControlNeXt-SD1.5


This is the training script for our ControlNeXt model, based on Stable Diffusion 1.5.

Our training and inference code has undergone some updates compared to the original version. Please refer to this version as the standard.

We provide an example using an open dataset, where our method achieves convergence in just a thousand training steps.

## Train


```
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
```

> --controlnext_scale: Set between [0, 1]; controls the strength of ControlNeXt. A larger value indicates stronger control. For tasks requiring dense conditional controls, such as depth, setting it larger (such as 1.) will provide better control. Increasing this number will lead to faster convergence and stronger control, but it can sometimes overly influence the final generation.


> --save_load_weights_increments: Choose whether to save the trainable parameters directly or just the weight increments, i.e., $W_{finetune} - W_{pretrained}$. This is useful when adapting to various backbones.

## Inference


```
python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="test" \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
 --controlnet_model_name_or_path checkpoints/checkpoint-1400/controlnext.bin \
 --unet_model_name_or_path checkpoints/checkpoint-1200/unet.bin \
 --controlnext_scale 0.35 
```

```
python run_controlnext.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="test" \
 --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  \
 --controlnet_model_name_or_path checkpoints/checkpoint-800/controlnext.bin \
 --unet_model_name_or_path checkpoints/checkpoint-1200/unet_weight_increasements.bin \
 --controlnext_scale 0.35 \
 --save_load_weights_increaments 
```