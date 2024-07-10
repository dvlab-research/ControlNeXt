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


import imageio 
import pygifsicle 
 
# Read the video file into memory 
frames = imageio.v2.imread('/home/llm/bhpeng/github/ControlNeXt/ControlNeXt-SVD/outputs/chair/chair.mp4') 
 
# Convert the frames to GIF format 
gif_image = pygifsicle.optimize(frames) 
 
# Save the GIF image to a file 
with open('/home/llm/bhpeng/github/ControlNeXt/ControlNeXt-SVD/outputs/chair/chair.gif', 'wb') as f: 
    f.write(gif_image) 
