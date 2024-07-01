import os
import torch
import numpy as np
from PIL import Image
from pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt
from models.controlnext_vid_svd import ControlNeXtSDVModel
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from transformers import CLIPVisionModelWithProjection
import re 
from diffusers import AutoencoderKLTemporalDecoder
from moviepy.editor import ImageSequenceClip
from decord import VideoReader
import argparse
from safetensors.torch import load_file


def write_mp4(video_path, samples, fps=14):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac")

def save_vid_side_by_side(batch_output, validation_control_images, output_folder, fps):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    video_path = output_folder+'/test_1.mp4'
    final_images = []
    outputs = []
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))
    write_mp4(video_path, final_images, fps=fps)

    output_path = output_folder + "/output.mp4"
    write_mp4(output_path, outputs, fps=fps)


def load_images_from_folder_to_pil(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))
        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)
    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images


def load_images_from_video_to_pil(video_path):
    images = []

    vr = VideoReader(video_path)
    length = len(vr)

    for idx in range(length):
        frame = vr[idx].asnumpy()
        images.append(Image.fromarray(frame))
    return images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--validation_control_images_folder",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--validation_control_video_path",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--height",
        type=int,
        default=768,
        required=False
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.5,
        required=False
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=14,
        required=False
    )

    parser.add_argument(
        "--controlnext_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        required=True
    )

    args = parser.parse_args()
    return args


def load_tensor(tensor_path):
    if os.path.splitext(tensor_path)[1] == '.bin':
        return torch.load(tensor_path)
    elif os.path.splitext(tensor_path)[1] == ".safetensors":
        return load_file(tensor_path)
    else:
        print("without supported tensors")
        os._exit()


# Main script
if __name__ == "__main__":
    args = parse_args()

    assert (args.validation_control_images_folder is None) ^ (args.validation_control_video_path is None), "must and only one of [validation_control_images_folder, validation_control_video_path] should be given"
    if args.validation_control_images_folder is not None:
        validation_control_images = load_images_from_folder_to_pil(args.validation_control_images_folder)
    else:
        validation_control_images = load_images_from_video_to_pil(args.validation_control_video_path)

    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    controlnext = ControlNeXtSDVModel()
    controlnext.load_state_dict(load_tensor(args.controlnext_path))
    unet.load_state_dict(load_tensor(args.unet_path), strict=False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", variant="fp16")
    
    pipeline = StableVideoDiffusionPipelineControlNeXt.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnext=controlnext, 
        unet=unet,
        vae=vae,
        image_encoder=image_encoder)
    pipeline.to(dtype=torch.float16)
    pipeline.enable_model_cpu_offload()

    os.makedirs(args.output_dir, exist_ok=True)

    # Inference and saving loop
    final_result = []
    ref_image = Image.open(args.ref_image_path).convert('RGB')
    frames = 14
    num_frames = len(validation_control_images)  

    video_frames = pipeline(
        ref_image, 
        validation_control_images[:num_frames], 
        decode_chunk_size=4,
        num_frames=num_frames,
        motion_bucket_id=127.0, 
        fps=7,
        controlnext_cond_scale=1.0, 
        width=args.width, 
        height=args.height, 
        min_guidance_scale=args.guidance_scale, 
        max_guidance_scale=args.guidance_scale, 
        frames_per_batch=frames, 
        num_inference_steps=args.num_inference_steps, 
        overlap=4).frames[0]
    final_result.append(video_frames)

    save_vid_side_by_side(
        final_result, 
        validation_control_images[:num_frames], 
        args.output_dir, 
        fps=args.fps)