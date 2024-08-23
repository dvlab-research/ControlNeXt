# 🌀 ControlNeXt-Experiences

As we all know, developing a high-quality model is not just an academic challenge; it also requires extensive engineering experience. Therefore, we are sharing the insights we gained during this project. These insights are the result of significant time and effort. If you find them helpful, please consider giving us a star or citing our work.

May they will help you.

### 1. Human-related generation

As I’ve mentioned, we only select a small subset of parameters, which is fully adapted to the SD1.5 and SDXL backbones. By training fewer than 100 million parameters, we still achieve excellent performance. But this is is not suitable for the SD3 and SVD training. This is because, after SDXL, Stability faced significant legal risks due to the generation of highly realistic human images. After that, they stopped refining their models on human-related data, such as SVD and SD3, to avoid potential risks.

To achieve optimal performance, it's necessary to first continue training SVD and SD3 on human-related data to develop a robust backbone before fine-tuning. Of course, you can also combine the continual pretraining and finetuning (Open all the parameters to train. There will not be a significant differences.). So you can find that we direct provide the full SVD parameters.

Although this may not be directly related to academia, it is crucial for achieving good performance.

### 2. Data

Due to privacy policies, we are unable to share the data. However, data quality is crucial, as many videos on the internet are highly compressed. It’s important to focus on collecting high-quality data without compression.

### 3. Hands

Generating hands is a challenging problem in both video and image generation. To address this, we focus on the following strategies:

a. Use clear and high-quality data, which is crucial for accurate generation.

b. Since the hands occupy a relatively small area, we apply a larger scale for the loss function specifically for this region to improve the generation quality.

### 4. Pose alignment

Thanks [mimic](https://github.com/Tencent/MimicMotion).  SVD performs poorly, especially with large motions. Therefore, it is important to avoid large movements and shifts. So please note that in [preprocess](https://github.com/dvlab-research/ControlNeXt/blob/main/ControlNeXt-SVD-v2/dwpose/preprocess.py), there is a alignment between the refenrece image and pose. This is crucial.


### 5. Control level

You can find that we adopt a magic nuber when adding the conditions. 

Such as in `ControlNeXt-SVD-v2/models/unet_spatio_temporal_condition_controlnext.py`:
```python
sample = sample + conditional_controls * scale * 0.2
```

You can notice that we time a `0.2`. This superparameter is used to adjust the control level: increasing this value will strengthen the control level.

However, if this value is set too high, the control may become overly strong and may not be apparent in the final generated images.

So you can adjust it to get a good result. In our experiences, for the dense controls such as super-resolution or depth, we need to set it as `1`.


### 6. Training parameters 

One of the most important findings is that directly training the base model yields better performance compared to methods like LoRA, Adapter, and others.Even when we train the base model, we only select a small subset of the pre-trained parameters. You can also adaptively adjust the number of selected parameters. For example, with high-quality data, having more trainable parameters can improve performance. However, this is a trade-off, and regardless of the approach, directly training the base model often yields the best results.


### If you find this work useful, please consider citing:
```
@article{peng2024controlnext,
  title={ControlNeXt: Powerful and Efficient Control for Image and Video Generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}
```
