# 🌀 ControlNeXt-SVD-v2-Training

# Important

I found that sometimes, when I change the version of dependencies, the training may not converge at all. I haven't identified the reason yet, but I've listed all our dependencies in the [requirements.txt](./requirements.txt) file. It's a bit detailed, but you can focus on the key dependencies like `torch`, `deepspeed`, `diffusers`, `accelerate`... When issues arise, checking these first may help.

> The training scripts are still under testing after my reconstruction. Since there has been significant demand, I'm releasing them now. Please keep an eye out for updates.

## Main

Due to privacy concerns, we are unable to release certain resources, such as the training data and the SD3-based model. However, we are committed to sharing as much as possible. If you find this repository helpful, please consider giving us a star or citing our work!

The training scripts are intended for users with a basic understanding of `Python` and `Diffusers`. Therefore, we will not provide every detail. If you have any questions, please refer to the code first. Thank you! If you encounter any bugs, please contact us and let us know.

## Experiences

We share more training experiences in the [Issue](https://github.com/dvlab-research/ControlNeXt/issues/14#issuecomment-2290450333) and [There](../experiences.md).
We spent a lot of time to find these. Now share with all of you. May these will help you!



## Training script

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate  launch --config_file ./deepspeed.yaml train_svd.py \
 --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
 --output_dir= $PATH_TO_THE_SAVE_DIR \
 --dataset_type="ubc" \
 --meta_info_path=$PATH_TO_THE_META_INFO_FILE_FOR_DATASET \
 --validation_image_folder=$PATH_TO_THE_GROUND_TRUTH_DIR_FOR_VALIDATION \
 --validation_control_folder=$PATH_TO_THE_POSE_DIR_FOR_VALIDATION \
 --validation_image=$PATH_TO_THE_REFERENCE_IMAGE_FILE_FOR_VALIDATION \
 --width=576 \
 --height=1024 \
 --lr_warmup_steps 500 \
 --sample_n_frames 21 \
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
```

We set `--num_train_epochs=6000` to ensure no stopped training, but you can stop the process at any point when you believe the results are satisfactory.

## Training validation

Please compile the data for validation like:
```
├───ControlNeXt-SVD-v2-Training
    └─── ...
    ├───validation
    |      ├───ground_truth
    |      |  ├───0.png
    |      |  ├─── ...
    |      |  └───13.png
    |      |
    |      └───pose
    |      |  ├───0.png
    |      |  ├─── ...
    |      |  └───13.png
    |      |
    |      └───reference_image.png
    |
    └─── ...
```

And then replace the `path` to:
```bash
--validation_image_folder=$PATH_TO_THE_GROUND_TRUTH_DIR_FOR_VALIDATION \
--validation_control_folder=$PATH_TO_THE_POSE_DIR_FOR_VALIDATION \
--validation_image=$PATH_TO_THE_REFERENCE_IMAGE_FILE_FOR_VALIDATION \
```

## Meta info

Please construct the training dataset and provide a list of the data entries in a .json file. We give an example in `meta_info_example/meta_info.json` (the data list) and `meta_info_example/meta_info/1.json`(Detailed meta information for each single video recoarding the position and score):

`meta_info.json`
```json
[
    {
        "video_path": "PATH_TO_THE_SOURCE_VIDEO",
        "guide_path": "PATH_TO_THE_CORESEPONDING_POSE_VIDEO",
        "meta_info": "PATH_TO_THE_JSON_FILE_RECORD_THE_DETAILED_DETECTION_RESULTS(we give an example in meta_info/1.json)"
    }
  ...
]
```

## GPU memory

It requires substantial memory for training, as we use a high resolution and long frame batches to achieve optimal performance. However, you can implement certain techniques to reduce memory consumption, although they may result in a trade-off with performance.

> 1. Adopt bf16 and fp16 (we have already implemented this).
> 2. Use DeepSpeed and distributed training across multiple machines.
> 3. Reduce the resolution by set `--width=576 --height=1024 `, such as `512*768`
> 4. Reduce the `--sample_n_frames`


### If you find this work helpful, please consider citing:
```
@article{peng2024controlnext,
  title={ControlNeXt: Powerful and Efficient Control for Image and Video Generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}
```