
# 🌀 ControlNeXt



##   [📝 Project Page](https://pbihao.github.io/projects/controlnext/index.html)  |  [📚 Paper](https://arxiv.org/abs/2408.06070) | [🗂️ Demo (SDXL)](https://huggingface.co/spaces/Eugeoter/ControlNeXt)


**ControlNeXt** is our official implementation for controllable generation, supporting both images and videos while incorporating diverse forms of control information. In this project, we propose a new method that reduces trainable parameters by up to 90% compared with ControlNet, achieving faster convergence and outstanding efficiency. This method can be directly combined with other LoRA techniques to alter style and ensure more stable generation. Please refer to the examples for more details.

We provide an online demo of [ControlNeXt-SDXL](./ControlNeXt-SDXL/). Due to the high resource requirements of SVD, we are unable to offer it online.

> This project is still undergoing iterative development. The code and model may be updated at any time. More information will be provided later.

# Experiences
We share more training experiences [there](./experiences.md) and in the [Issue](https://github.com/dvlab-research/ControlNeXt/issues/14#issuecomment-2290450333).
We spent a lot of time to find these. Now share with all of you. May these will help you!

# Model Zoo

- **ControlNeXt-SDXL** [ [Link](ControlNeXt-SDXL) ] : Controllable image generation. Our model is built upon [Stable Diffusion XL ](stabilityai/stable-diffusion-xl-base-1.0). Fewer trainable parameters, faster convergence, improved efficiency, and can be integrated with LoRA.

- **ControlNeXt-SVD-v2** [ [Link](ControlNeXt-SVD-v2) ] :  Generate the video controlled by the sequence of human poses. In the v2 version, we implement several improvements: a higher-quality collected training dataset, larger training and inference batch frames, higher generation resolution, enhanced human-related video generation through continual training, and pose alignment for inference to improve overall performance.

- **ControlNeXt-SVD-v2-Training** [ [Link](ControlNeXt-SVD-v2-Training) ] :  The training scripts for our `ControlNeXt-SVD-v2` [ [Link](ControlNeXt-SVD-v2) ].

- **ControlNeXt-SVD** [ [Link](ControlNeXt-SVD) ] :  Generate the video controlled by the sequence of human poses. This can be seen as an attempt to replicate the implementation of [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone). However, our model is built upon [Stable Video Diffusion](https://stability.ai/stable-video), employing a more concise architecture.

- **ControlNeXt-SD1.5** [ [Link](ControlNeXt-SD1.5) ] : Controllable image generation. Our model is built upon [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). Fewer trainable parameters, faster convergence, improved efficiency, and can be integrated with LoRA.

- **ControlNeXt-SD1.5-Training** : The process is quite simple, so we do not plan to invest additional effort into it. You can directly use the HuggingFace examples. Please refer to the SDXL and SVD sections for our newly updated versions!

- **ControlNeXt-SD3** [ [Link](ControlNeXt-SD3) ] : Stay tuned.



# 🎥 Examples
### For more examples, please refer to our [Project page](https://pbihao.github.io/projects/controlnext/index.html).

### [ControlNeXt-SDXL](ControlNeXt-SDXL)

<p align="center">
  <img src="ControlNeXt-SDXL/examples/demo/demo1.jpg" width="80%" alt="demo1">
  <img src="ControlNeXt-SDXL/examples/demo/demo2.jpg" width="80%" alt="demo2">
  <img src="ControlNeXt-SDXL/examples/demo/demo3.jpg" width="80%" alt="demo3">
  <img src="ControlNeXt-SDXL/examples/demo/demo5.jpg" width="80%" alt="demo5">
</p>

### [ControlNeXt-SVD-v2](ControlNeXt-SVD-v2)
If you can't load the videos, you can also directly download them from [here](examples/demos) and [here](examples/video).
Or you can view them from our [Project Page](https://pbihao.github.io/projects/controlnext/index.html) or [BiliBili](https://www.bilibili.com/video/BV1wJYbebEE7/?buvid=YC4E03C93B119ADD4080B0958DE73F9DDCAC&from_spmid=dt.dt.video.0&is_story_h5=false&mid=y82Gz7uArS6jTQ6zuqJj3w%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=4E5549FC-0710-4030-BD2C-CDED80B46D08&share_source=WEIXIN&share_source=weixin&share_tag=s_i&timestamp=1723123770&unique_k=XLZLhCq&up_id=176095810&vd_source=3791450598e16da25ecc2477fc7983db).

<table style="margin: 0 auto; border-collapse: collapse;">
    <tr>
        <td width="40%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/user-attachments/assets/9d45a00a-d3cd-48e1-aa78-1d3158bfd4f4" muted="false"></video>
        </td>
        <td width="40%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/user-attachments/assets/1004960a-82de-4f0d-a329-ba676b8cbd0d" muted="false"></video>
        </td>
    </tr>
    <tr>
        <td width="40%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/user-attachments/assets/7db1acd1-0c61-4855-91bb-e4e8f8989393" muted="false"></video>
        </td>
        <td width="40%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/user-attachments/assets/0f32df53-1827-404d-806a-23e65d357504" muted="false"></video>
        </td>
    </tr>

</table>

<video width="80%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/user-attachments/assets/c69b4f34-0851-4637-a9ef-fb91beed5666" muted="false"></video>

<video width="80%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/user-attachments/assets/32a4d24b-bc39-4ea9-9fd4-ed78b4eec116" muted="false"></video>


### [ControlNeXt-SVD](ControlNeXt-SVD)
If you can't load the videos, you can also directly download them from [here](ControlNeXt-SVD/outputs).

<video width="80%" height='auto' style="display: block; margin: 20px auto;" controls autoplay loop src="https://github.com/dvlab-research/ControlNeXt/assets/22709028/c0ec7591-2100-42d9-99dd-cc55c5fa006f" muted="false"></video>

<video width="80%" height='auto' style="display: block; margin: 20px auto;" controls autoplay loop src="https://github.com/dvlab-research/ControlNeXt/assets/22709028/5da1ba44-cb92-49c2-95f4-05b8e02ed6eb" muted="false"></video>

<!-- <video width="40%" height='auto' style="display: block; margin: 20px auto;" controls autoplay loop src="https://github.com/dvlab-research/ControlNeXt/assets/22709028/bc819ca7-81f3-4d63-901a-a1a4b4afc395" muted="false"></video>

<video width="40%" height='auto' style="display: block; margin: 20px auto;" controls autoplay loop src="https://github.com/dvlab-research/ControlNeXt/assets/22709028/adc01423-fcec-467e-a7ab-87a6e1ef5f62" muted="false"></video> -->

<table>
<tr>
    <td width=50% style="border: none">
        <video width="80%" height='auto' style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/dvlab-research/ControlNeXt/assets/22709028/bc819ca7-81f3-4d63-901a-a1a4b4afc395" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video width="80%" height='auto' style="display: block; margin: 0px auto;" controls autoplay loop src="https://github.com/dvlab-research/ControlNeXt/assets/22709028/adc01423-fcec-467e-a7ab-87a6e1ef5f62" muted="false"></video>
    </td>
</tr>
</table>



### [ControlNeXt-SD1.5](ControlNeXt-SD1.5)

<p align="center">
  <img src="ControlNeXt-SD1.5/examples/deepfashion_multiview/eval_img/DreamShaper.jpg" width="90%" alt="DreamShaper">
</p>
<p align="center">
  <img src="ControlNeXt-SD1.5/examples/deepfashion_multiview/eval_img/Anythingv3_fischl.jpg" width="90%" alt="Anythingv3">
</p>
<p align="center">
  <img src="ControlNeXt-SD1.5/examples/deepfashion_caption/eval_img/chinese_style.jpg" width="90%" alt="Anythingv3">
</p>


### If you find this work useful, please consider citing:
```
@article{peng2024controlnext,
  title={ControlNeXt: Powerful and Efficient Control for Image and Video Generation},
  author={Peng, Bohao and Wang, Jian and Zhang, Yuechen and Li, Wenbo and Yang, Ming-Chang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2408.06070},
  year={2024}
}
```
