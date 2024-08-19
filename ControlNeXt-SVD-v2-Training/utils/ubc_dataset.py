import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
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


class UBCFashion(Dataset):
    def __init__(
            self,
            meta_info_path, 
            width=512,
            height=768,
            sample_n_frames=14,
            interval_frame=1,
            stage=2,
            ref_aug=False,
            ref_aug_ratio=0.,
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
        if self.stage != 2:
            self.ref_aug_ratio = 1.
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        if self.ref_aug:
            # def random_crop_bottom(tensor, min_ratio=0.7):
            #     assert len(tensor.shape) == 3, "only for image"
            #     _, height, width = tensor.size()
            #     crop_ratio =  random.uniform(min_ratio, 1)
            #     crop_width = int(width * crop_ratio)
            #     crop_height = int(height * crop_ratio)
            #     cropped_tensor = tensor[:, :crop_height, (width - crop_width) // 2:(width + crop_width) // 2]
            #     return cropped_tensor
            def random_crop_bottom(tensor, min_ratio=0.7):
                assert len(tensor.shape) == 3, "only for image"
                _, height, width = tensor.size()
                crop_ratio =  min_ratio + random.random() * (1 - min_ratio)
                crop_height = int(height * crop_ratio)
                tensor[:, crop_height:, :] = 0
                return tensor
            
                
            self.ref_transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(p=0.05),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                transforms.Resize(sample_size, antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        def random_transpose(tensor: torch.tensor):
            if random.random() > 0.5:
                if len(tensor.shape) == 3:
                    if random.random() > 0.5:
                        return tensor
                    else:
                        tensor[[0,-1],...] = tensor[[-1,0],...]
                        return tensor
                elif len(tensor.shape) == 4:
                    if random.random() > 0.5:
                        return tensor
                    else:
                        tensor[:, [0,-1],...] = tensor[:, [-1,0],...]
                        return tensor
                else:
                    exit()
            else:
                return tensor
        self.pose_transforms = transforms.Compose([
            transforms.Lambda(random_transpose), 
            transforms.Resize(sample_size, antialias=True),
        ])

    
    def get_batch(self, idx):
        while True:
            video_path = self.meta_info[idx]['video_path']
            guide_path = self.meta_info[idx]['guide_path']
            try:
                if self.stage == 2:
                    vr = VideoReader(video_path)
                    length = len(vr)

                    # Read videos ...
                    if "valid_index" in self.meta_info[idx].keys() and self.valid_index:
                        frame_ids = random.sample(self.meta_info[idx]['valid_index'], self.sample_n_frames)
                        frame_ids = sorted(frame_ids)

                        reference_id = random.randint(min(self.meta_info[idx]['valid_index']), max(self.meta_info[idx]['valid_index']))
                    else:
                        interval_frame = min(self.interval_frame, int(length // self.sample_n_frames))
                        segment_length = interval_frame * self.sample_n_frames
                        assert length >= segment_length, "Too short video..."
                        bg_frame_id = random.randint(0, length - segment_length)
                        frame_ids = list(range(bg_frame_id, bg_frame_id + segment_length, interval_frame))

                        # ref_ids_bg = max(0, bg_frame_id - 2 * segment_length)
                        # ref_ids_ed = min(length, bg_frame_id + 3 * segment_length) - 1
                        # reference_id = random.randint(ref_ids_bg, ref_ids_ed)
                        reference_id = random.randint(0, length - 1)
                        # reference_id = max(0, frame_ids[0] - 4)
                        


                    pixel_values = np.array([vr[frame_id].asnumpy() for frame_id in frame_ids])
                    pixel_values = numpy_to_pt(pixel_values)


                    reference_image = vr[reference_id].asnumpy()
                    reference_image = torch.from_numpy(reference_image.transpose(2, 0, 1))
                    reference_image = reference_image.float() / 255.

                    # Read guide image ...
                    vr = VideoReader(guide_path)
                    assert len(vr) == length, "Guide and video lengthes are conflict ..."
                    guide_values = np.array([vr[frame_id].asnumpy() for frame_id in frame_ids])
                    guide_values = numpy_to_pt(guide_values)

                    return pixel_values, guide_values, reference_image
                elif self.stage == 1:
                    vr = VideoReader(video_path)
                    length = len(vr)

                    assert length >= self.sample_n_frames, "Too short video..."
                    frame_ids = random.sample(range(length), self.sample_n_frames)
                    frame_ids.sort()
                    ref_id = random.randint(0, length - 1)
                    pix_id = random.randint(0, length - 1)


                    pixel_values = np.array([vr[pix_id].asnumpy(), ])
                    pixel_values = numpy_to_pt(pixel_values)

                    reference_image = vr[ref_id].asnumpy()
                    reference_image = torch.from_numpy(reference_image.transpose(2, 0, 1))
                    reference_image = reference_image.float() / 255.

                    # Read guide image ...
                    vr = VideoReader(guide_path)
                    assert len(vr) == length, "Guide and video lengthes are conflict ..."
                    guide_values = np.array([vr[pix_id].asnumpy(), ])
                    guide_values = numpy_to_pt(guide_values)

                    return pixel_values, guide_values, reference_image
                elif self.stage == 3:
                    if "ref_path" in self.meta_info[idx].keys():
                        ref_path = self.meta_info[idx]['ref_path']
                    else:
                        ref_path = None
                    image = pil_image_to_numpy(Image.open(video_path))
                    pose = pil_image_to_numpy(Image.open(guide_path))


                    pixel_values = np.array([image, image])
                    pixel_values = numpy_to_pt(pixel_values)

                    if ref_path is None:
                        reference_image = deepcopy(image)
                    else:
                        reference_image = pil_image_to_numpy(Image.open(ref_path))
                    reference_image = torch.from_numpy(reference_image.transpose(2, 0, 1))
                    reference_image = reference_image.float() / 255.
                    if ref_path is None:
                        _, height, width = reference_image.size()
                        crop_ratio =  0.3 + random.random() * (1 - 0.3)
                        crop_height = int(height * crop_ratio)
                        reference_image[:, crop_height:, :] = 0

                    guide_values = np.array([pose, pose])
                    guide_values = numpy_to_pt(guide_values)

                    return pixel_values, guide_values, reference_image
            except:
                print("****** Filed to load: {} ******".format(video_path))
                idx = random.randint(0, self.length - 1)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, guide_values, reference_image = self.get_batch(idx)

        pixel_values = self.pixel_transforms(pixel_values)
        if self.ref_aug and random.random() <= self.ref_aug_ratio:
            reference_image = self.ref_transforms(reference_image)
        else:
            reference_image = self.pixel_transforms(reference_image)
        guide_values = self.pose_transforms(guide_values)

        sample = dict(
            pixel_values = pixel_values, 
            guide_values = guide_values,
            reference_image = reference_image, 
           )
        return sample

import cv2
def recover_batch(batch, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(os.path.join(folder_path, "pose")):
            os.makedirs(os.path.join(folder_path, "pose"))
        if not os.path.exists(os.path.join(folder_path, "rgb")):
            os.makedirs(os.path.join(folder_path, "rgb"))
        pixel_values = batch["pixel_values"]
        guide_values = batch["guide_values"]
        ref_values = batch["reference_image"]
        pixel_values = (((pixel_values + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        guide_values = (guide_values.numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        ref_values = (((ref_values + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(1, 2, 0)
        ref_values = cv2.cvtColor(ref_values, cv2.COLOR_RGB2BGR)
        for idx in range(len(pixel_values)):
            frame = pixel_values[idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(folder_path, "rgb", "{}.png".format(idx)), frame)
        for idx in range(len(guide_values)):
            frame = guide_values[idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(folder_path, "pose", "{}.png".format(idx)), frame)
        cv2.imwrite(os.path.join(folder_path, "ref.png"), ref_values)

if __name__ == "__main__":

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/meta_info_train.json",
        interval_frame=4,
        sample_n_frames=1,
    )

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/filtered_meta_info.json",
        interval_frame=4,
        sample_n_frames=1,
        ref_aug=False
    )

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/meta_info/hand_score_5000_filtered_56_index.json",
        interval_frame=4,
        sample_n_frames=14,
        ref_aug=False
    )

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/filtered_meta_info_withv3.json",
        interval_frame=4,
        sample_n_frames=14,
        ref_aug=False
    )

    dataset = UBCFashion(
        meta_info_path="/dataset-vlm/bhpeng/controlany/meta_info/human_filter.json",
        interval_frame=4,
        sample_n_frames=14,
        ref_aug=False
    )

    recover_batch(dataset[random.randint(0, 1000)], "/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/iamges")

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/meta_info_train.json",
        interval_frame=4,
    )

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/meta_info_train.json",
        interval_frame=4,
        stage=1
    )

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/image_dataset/meta_info/viton_df2mv_tiktoksub2_tiktokv3.json",
        interval_frame=4,
        stage=3

    )


    import json
    with open("/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/image_dataset/meta_info/df2_viton_fp_dfmv_tiktok.json") as f:
        data = json.load(f)

    ndata = []
    for dic in data:
        path = dic['img_path']
        pose = dic['guide_path']
        # if not("virtualtron/viton" in path or\
        #     "deepfashion_multiview" in path):
        if not "tiktok" in path:
            continue
        ndata.append(dict(video_path=path, guide_path=pose))

    with open("/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/image_dataset/meta_info/tiktok.json", "w") as f:
        json.dump(ndata, f, ensure_ascii=False, indent=2)