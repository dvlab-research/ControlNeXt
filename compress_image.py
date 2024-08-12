# from PIL import Image
# import cv2
# import numpy as np

# img_path = "/home/llm/bhpeng/github/ControlAny/ControlAny-SDXL/examples/vidit_depth/condition_0.png"
# save_path = "/home/llm/bhpeng/github/ControlAny/ControlAny-SDXL/examples/vidit_depth/condition_02.png"

# length = 1
# select_id = []

# image = cv2.imread(img_path)
# height, width, _ = image.shape
# part_width = width // length

# splited_imgs = []
# for i in range(length):
#     left = i * part_width
#     right = (i + 1) * part_width if i < length - 1 else width  # 确保最后一个分块到图像右边界
    
#     split_img = image[:, left:right]
#     splited_imgs.append(split_img)

# merge_imgs = []
# merge_imgs.append(splited_imgs[0])
# for i in select_id:
#     merge_imgs.append(splited_imgs[i])
# merge_imgs = np.concatenate(merge_imgs, axis=1)
# print(merge_imgs.shape)
# resized_img = cv2.resize(merge_imgs, (merge_imgs.shape[1]//2, merge_imgs.shape[0]//2), interpolation=cv2.INTER_AREA)
# print(resized_img.shape)
# cv2.imwrite(save_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

# img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)


# from moviepy.editor import VideoFileClip
# import moviepy.video.io.ffmpeg_writer as ffmpeg_writer


# video_path = 'ControlNeXt-SVD/outputs/chair/chair.mp4'
# clip = VideoFileClip(video_path)

# gif_path = 'ControlNeXt-SVD/outputs/chair/chair.gif'
# clip.write_gif(gif_path, fps=14, program='ffmpeg', opt="nq", fuzz=1, )


from PIL import Image
import os

def compress_image(input_path, output_path, quality=85):
    """
    压缩图片，同时尽可能保留质量。
    
    :param input_path: 原始图片路径
    :param output_path: 压缩后图片保存路径
    :param quality: 压缩质量，取值范围是 0 到 100，100 代表最高质量
    """
    # 打开图片
    with Image.open(input_path) as img:
        # 确保图片是 RGB 模式
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # 保存压缩后的图片
        img.save(output_path, "JPEG", quality=quality, optimize=True)


img_paths = [
    'ControlNeXt-SDXL/examples/demo/demo1.png',
    'ControlNeXt-SDXL/examples/demo/demo3.png',
    'ControlNeXt-SDXL/examples/demo/demo5.png'
]
quality = 50

for src_path in img_paths:
    dst_path = src_path
    src_path = os.path.join(os.path.split(src_path)[0], 'src_'+os.path.split(src_path)[1])
    os.rename(dst_path, src_path)
    dst_path = '.'.join(dst_path.split('.')[:-1])+'.jpg'
    compress_image(src_path, dst_path, quality=quality)