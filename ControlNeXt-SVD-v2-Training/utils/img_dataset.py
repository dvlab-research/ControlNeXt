import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from utils.util import zero_rank_print
from PIL import Image
from copy import deepcopy


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


class ImgDataset(Dataset):
    def __init__(
            self,
            meta_info_path, 
            width=512,
            height=768,
        ):
        zero_rank_print(f"loading meta info from {meta_info_path} ...")
        with open(meta_info_path, 'r') as f:
            self.meta_info = json.load(f)

        random.shuffle(self.meta_info)    
        self.length           = len(self.meta_info)
        self.width            = width
        self.height           = height
        sample_size           = (height, width)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.ref_transforms = transforms.Compose([
            # transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 2.3), value=0, inplace=True),
            # transforms.ElasticTransform(alpha=50.0, sigma=5.0, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.RandomAffine(degrees=45, translate=(0.4, 0.4), scale=(0.8, 1.2), shear=15),
            # transforms.RandomHorizontalFlip(p=0.1),
            # transforms.RandomVerticalFlip(p=0.1),
            # transforms.Resize(sample_size,  antialias=True),
            transforms.RandomResizedCrop(sample_size, scale=(0.8, 1.2), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.pose_transforms = transforms.Compose([
            transforms.Resize(sample_size, antialias=True),
        ])

    
    def get_batch(self, idx):
        while True:
            img_path = self.meta_info[idx]['img_path']
            guide_path = self.meta_info[idx]['guide_path']
            if "ref_path" in self.meta_info[idx].keys():
                ref_path = self.meta_info[idx]['ref_path']
            else:
                ref_path = None
            try:
                image = pil_image_to_numpy(Image.open(img_path))
                pose = pil_image_to_numpy(Image.open(guide_path))


                pixel_values = np.array([image, ])
                pixel_values = numpy_to_pt(pixel_values)

                if ref_path is None:
                    reference_image = deepcopy(image)
                else:
                    reference_image = pil_image_to_numpy(Image.open(ref_path))
                reference_image = torch.from_numpy(reference_image.transpose(2, 0, 1))
                reference_image = reference_image.float() / 255.

                guide_values = np.array([pose, ])
                guide_values = numpy_to_pt(guide_values)

                return pixel_values, guide_values, reference_image
            except:
                print("****** Filed to load: {} ******".format(img_path))
                idx = random.randint(0, self.length - 1)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, guide_values, reference_image = self.get_batch(idx)

        pixel_values = self.pixel_transforms(pixel_values)
        reference_image = self.ref_transforms(reference_image)
        guide_values = self.pose_transforms(guide_values)
        sample = dict(
            pixel_values = pixel_values, 
            guide_values = guide_values,
            reference_image = reference_image, 
           )
        return sample



if __name__ == "__main__":
    from utils.util import save_videos_grid
    import time
    import cv2
    import subprocess
    import os
    def reencode_video(input_file):
        output_file = os.path.splitext(input_file)[0] + "_tmp.mp4"
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'copy',
            '-y',  
            output_file
        ]
        # subprocess.run(ffmpeg_cmd)
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert os.path.exists(output_file), "Faild convertion: {}".format(input_file)
        os.remove(input_file)
        os.rename(output_file, input_file)


    def recover_batch(batch):
        pixel_values = batch["pixel_values"]
        guide_values = batch["guide_values"]
        reference_image = batch["reference_image"]
        pixel_values = (((pixel_values + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        guide_values = (guide_values.numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(0, 2, 3, 1)
        reference_image = (((reference_image + 1) / 2).numpy() * 255).astype(np.uint8).clip(min=0, max=255).transpose(1, 2, 0)
        video_writer = cv2.VideoWriter("/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/pixel_values.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 7, (512, 768))
        # reencode_video("/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/pixel_values.mp4")
        for idx in range(len(pixel_values)):    
            frame = pixel_values[idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_writer.write(frame)
        video_writer.release()
        video_writer = cv2.VideoWriter("/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/guide_values.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 7, (512, 768))
        for idx in range(len(guide_values)):
            frame = guide_values[idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_writer.write(frame)
        video_writer.release()
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/reference_image.png", reference_image)
        # reencode_video("/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/reference_image.png")

    
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
        



    dataset = ImgDataset(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/image_dataset/meta_info/df2_viton_fp.json",
    )

    recover_batch(dataset[1], "/home/llm/bhpeng/generation/svd-temporal-controlnet/tmp/svd/outputs")

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/meta_info_train.json",
        interval_frame=4,
    )

    dataset = UBCFashion(
        meta_info_path="/home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/meta_info_train.json",
        interval_frame=4,
        stage=1
    )


    # pixel_values = pixel_values, 
    #         guide_values = guide_values,
    #         reference_image = reference_image, 
    for i in range(64):
        batch = dataset[random.randint(0, 10240)]
        reference_image = batch['reference_image']
        guide_values = batch['guide_values'][0]
        reference_image = ((reference_image + 1) / 2 * 255).numpy().astype(np.uint8).clip(0, 255).transpose(1, 2, 0)
        guide_values = (guide_values * 255).numpy().astype(np.uint8).clip(0, 255).transpose(1, 2, 0)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
        guide_values = cv2.cvtColor(guide_values, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/home/llm/bhpeng/generation/svd-temporal-controlnet/validation_demo/human_first_stage/rgb/{}.png".format(i), reference_image)
        cv2.imwrite("/home/llm/bhpeng/generation/svd-temporal-controlnet/validation_demo/human_first_stage/pose/{}.png".format(i), guide_values)

    # import pdb
    # pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape,
              batch["guide_values"].shape,
              batch['reference_image'].shape,
              batch['motion_values'])
        time.sleep(1)


# vr = VideoReader("/home/llm/bhpeng/generation/svd-temporal-controlnet/dataset/ubc_fashion/pose_test/91-3003CN5S.mp4")
# ids = list(range(60, 60 + 2* 42, 2)) 
# for i, id in enumerate(ids):
#     frame = vr[id].asnumpy()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imwrite("/home/llm/bhpeng/generation/svd-temporal-controlnet/validation_demo/human_long/pose/frame_{}.png".format(i), frame)


# vr = VideoReader("/home/llm/bhpeng/generation/svd-temporal-controlnet/dataset/ubc_fashion/test/91-3003CN5S.mp4")
# ids = list(range(60, 60 + 2* 42, 2)) 
# for i, id in enumerate(ids):
#     frame = vr[id].asnumpy()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imwrite("/home/llm/bhpeng/generation/svd-temporal-controlnet/validation_demo/human_long/rgb/frame_{}.png".format(i), frame)