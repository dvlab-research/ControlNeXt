import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from dwpose.preprocess import get_image_pose, get_video_pose

ASPECT_RATIO = 9 / 16

def preprocess(video_path, image_path, width=576, height=1024, sample_stride=2, max_frame_num=None):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    # if h>w:
    #     w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    # else:
    #     w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    w_target, h_target = width, height
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get image&video pose value #################################################
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride, max_frame_num=max_frame_num)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    # image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    image_pixels = Image.fromarray(image_pixels)
    pose_pixels = [Image.fromarray(p.transpose((1,2,0))) for p in pose_pixels]
    # return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1
    return pose_pixels, image_pixels

