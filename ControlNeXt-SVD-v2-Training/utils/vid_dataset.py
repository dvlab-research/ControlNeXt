import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import cv2

from utils.util import zero_rank_print
from copy import deepcopy
from PIL import Image

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255.


def random_select_continual_sequence(sequence, nums, inter=2):
    length = len(sequence)
    inter = min(inter, length // nums)
    inter_length = inter * (nums - 1) + 1
    if inter_length > length:
        return None
    bg_idx = random.randint(0, length - inter_length)
    idx = []
    for i in range(nums):
        idx.append(sequence[bg_idx + i * inter])
    return idx


def draw_mask(frame, x0, y0, x1, y1, score=1., margin=10):
    H, W = frame.shape[-2:]
    x0 = int(x0 * W)
    x1 = int(x1 * W)
    y0 = int(y0 * H)
    y1 = int(y1 * H)
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, W), min(y1 + margin, H)
    frame[..., y0:y1, x0:x1] = score
    return frame


class UBCFashion(Dataset):
    def __init__(
            self,
            meta_info_path, 
            width=512,
            height=768,
            sample_n_frames=14,
            interval_frame=1,
            stage=2,
            ref_aug=True,
            ref_aug_ratio=0.9,
            valid_index=False
        ):
        zero_rank_print(f"loading meta info from {meta_info_path} ...")
        with open(meta_info_path, 'r') as f:
            self.meta_info = json.load(f)

        random.shuffle(self.meta_info)    
        self.length           = len(self.meta_info)
        self.sample_n_frames  = sample_n_frames
        self.width            = width
        self.height           = height
        self.interval_frame   = interval_frame
        self.stage            = stage
        self.ref_aug          = ref_aug
        self.ref_aug_ratio    = ref_aug_ratio
        self.valid_index      = valid_index
        sample_size           = (height, width)
        
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        def random_transpose(tensor: torch.tensor):
            if random.random() > 0.5:
                if len(tensor.shape) == 3 and tensor.shape[0] > 1:
                    if random.random() > 0.5:
                        return tensor
                    else:
                        tensor[[0,-1],...] = tensor[[-1,0],...]
                        return tensor
                elif len(tensor.shape) == 4 and tensor.shape[1] > 1:
                    if random.random() > 0.5:
                        return tensor
                    else:
                        tensor[:, [0,-1],...] = tensor[:, [-1,0],...]
                        return tensor
                else:
                    return tensor
            else:
                return tensor
        self.pose_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
        ])

    def get_batch(self, idx):
        while True:
            video_path = self.meta_info[idx]['video_path']
            guide_path = self.meta_info[idx]['guide_path']
            try:
                vr = VideoReader(video_path)
                length = len(vr)

                # Read videos ...
                init_interval_frame = self.interval_frame
                if init_interval_frame <= 0:
                    init_interval_frame = random.randint(2, 7)


                interval_frame = min(init_interval_frame, int(length // self.sample_n_frames))
                segment_length = interval_frame * self.sample_n_frames
                assert length >= segment_length, "Too short video..."
                bg_frame_id = random.randint(0, length - segment_length)
                frame_ids = list(range(bg_frame_id, bg_frame_id + segment_length, interval_frame))

                # select using score
                # if 'meta_info' in self.meta_info[idx].keys():
                #     with open(self.meta_info[idx]['meta_info']) as f:
                #         meta_info = json.load(f)
                #     sequence_ids = []
                #     for idx_frame, meta in enumerate(meta_info):
                #         if 'hands_score' in meta.keys():
                #             hands_score = meta['hands_score']
                #             for hands_idx in range(len(hands_score)):
                #                 hand_score = np.array(hands_score[hands_idx]).mean()
                #                 if hand_score > 0.4:
                #                     sequence_ids.append(idx_frame)
                #                     break
                #     ids = random_select_continual_sequence(sequence_ids, self.sample_n_frames, init_interval_frame)
                #     if ids is not None:
                #         frame_ids = ids

                reference_id = random.randint(0, length - 1)

                pixel_values = np.array([vr[frame_id].asnumpy() for frame_id in frame_ids])
                pixel_values = numpy_to_pt(pixel_values)

                reference_image = vr[reference_id].asnumpy()
                reference_image = torch.from_numpy(reference_image.transpose(2, 0, 1))
                reference_image = reference_image.float() / 255.

                # Read guide image ...
                vr = VideoReader(guide_path)
                assert abs(len(vr) - length) < 25, "Guide and video lengthes are conflict ..."
                # guide_values = np.array([vr[frame_id].asnumpy() for frame_id in frame_ids])
                # guide_values = numpy_to_pt(guide_values)
                # PBH no reference
                guide_values = np.array([vr[frame_id].asnumpy() for frame_id in frame_ids])
                # convert rgb and bgr pbh
                for idx_guide in range(len(guide_values)):
                    guide_values[idx_guide, ...] = cv2.cvtColor(guide_values[idx_guide, ...], cv2.COLOR_BGR2RGB)
                # guide_values[0, ...] = vr[reference_id].asnumpy()
                guide_values[0, ...] = cv2.cvtColor(vr[reference_id].asnumpy(), cv2.COLOR_BGR2RGB)
                guide_values = numpy_to_pt(guide_values)
                pixel_values[0, ...] = reference_image

                hands_mask = torch.zeros((guide_values.shape[0], 1, guide_values.shape[2], guide_values.shape[3]))
                if 'meta_info' in self.meta_info[idx].keys():
                    with open(self.meta_info[idx]['meta_info']) as f:
                        meta_info = json.load(f)
                    for idx, frame_id in enumerate(frame_ids):
                        meta = meta_info[frame_id]
                        if 'hands_boxes' in meta_info[frame_id].keys() and 'hands_score' in meta.keys():
                            hand_boxes = meta_info[frame_id]['hands_boxes']
                            for hands_idx, ((x0, y0), (x1, y1)) in enumerate(hand_boxes):
                                hands_score = meta['hands_score']
                                hands_score = np.array(hands_score[hands_idx]).mean()
                                if hands_score > 0.5:
                                    hands_score = 0.8 # max((hands_score - 0.5), 0) / 0.5
                                else:
                                    hands_score = 0.
                                draw_mask(hands_mask[idx], x0, y0, x1, y1, hands_score)
                
                # Random crop to the specific ratio
                vid_width = pixel_values.shape[-1]  
                vid_height = pixel_values.shape[-2]  
                if vid_height / vid_width > self.height / self.width:  
                    crop_width = vid_width  
                    crop_height = int(vid_width * self.height / self.width)  
                    h0 = random.randint(0, vid_height - crop_height)  
                    w0 = 0  
                else:  
                    crop_width = int(vid_height * self.width / self.height)  
                    crop_height = vid_height  
                    h0 = 0  
                    w0 = random.randint(0, vid_width - crop_width) 

                pixel_values = torch.stack([pixel_values[i, :, h0:h0+crop_height, w0:w0+crop_width] for i in range(len(pixel_values))], dim=0)  
                guide_values = torch.stack([guide_values[i, :, h0:h0+crop_height, w0:w0+crop_width] for i in range(len(guide_values))], dim=0)  
                reference_image = reference_image[:, h0:h0+crop_height, w0:w0+crop_width]
                if hands_mask is not None:
                    hands_mask = hands_mask[..., h0:h0+crop_height, w0:w0+crop_width]
                    

                return pixel_values, guide_values, reference_image, hands_mask
            except :
                print("****** Filed to load: {} ******".format(video_path))
                idx = random.randint(0, self.length - 1)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, guide_values, reference_image, hands_mask = self.get_batch(idx)

        pixel_values = self.pixel_transforms(pixel_values)
        reference_image = self.pixel_transforms(reference_image)
        guide_values = self.pose_transforms(guide_values)
        if hands_mask is not None:
            hands_mask = self.pose_transforms(hands_mask)
            hands_mask = (hands_mask * 4) + 1.
        
        hands_mask[0, ...] = 1
        

        sample = dict(
            pixel_values = pixel_values, 
            guide_values = guide_values,
            reference_image = reference_image, 
            hands_mask = hands_mask
        )
        return sample

def recover_batch(batch, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(os.path.join(folder_path, "pose")):
            os.makedirs(os.path.join(folder_path, "pose"))
        if not os.path.exists(os.path.join(folder_path, "rgb")):
            os.makedirs(os.path.join(folder_path, "rgb"))
        if not os.path.exists(os.path.join(folder_path, "hands_mask")):
            os.makedirs(os.path.join(folder_path, "hands_mask"))
        pixel_values = batch["pixel_values"]
        guide_values = batch["guide_values"]
        hands_mask = batch["hands_mask"]
        ref_values = batch["reference_image"]
        pixel_values = (((pixel_values + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        guide_values = (guide_values.numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        if hands_mask is not None:
            hands_mask = (hands_mask.numpy() / 5 * 255).clip(min=0, max=255).astype(np.uint8).transpose(0, 2, 3, 1)
        ref_values = (((ref_values + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(1, 2, 0)
        for idx in range(len(pixel_values)):
            frame = pixel_values[idx]
            Image.fromarray(frame).save(os.path.join(folder_path, "rgb", "{}.png".format(idx)))
        for idx in range(len(guide_values)):
            frame = guide_values[idx]
            Image.fromarray(frame).save(os.path.join(folder_path, "pose", "{}.png".format(idx)))
        if hands_mask is not None:
            for idx in range(len(hands_mask)):
                frame = pixel_values[idx]
                frame = hands_mask[idx] // 2 + frame // 2
                Image.fromarray(frame).save(os.path.join(folder_path, "hands_mask", "{}.png".format(idx)))
        Image.fromarray(ref_values).save(os.path.join(folder_path, "ref.png"))




if __name__ == "__main__":
    from utils.vid_dataset import *
    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj_vlm/tiktok_hq/meta_info/meta_v0_v1_v2_pexel.json",
        interval_frame=0,
        sample_n_frames=14,
        ref_aug=False,
        width=512,
        height=768
    )


    recover_batch(dataset[random.randint(0, 6000)], "/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/images")


