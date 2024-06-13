from PIL import Image
import cv2
import numpy as np

img_path = "/home/llm/bhpeng/github/ControlAny/ControlAny-SD1.5/examples/deepfashion_caption/eval_img/1717759208.062345.png"
save_path = "/home/llm/bhpeng/github/ControlAny/ControlAny-SD1.5/examples/deepfashion_caption/eval_img/warrior_bad.jpg"

select_id = [3, 4]

image = cv2.imread(img_path)
height, width, _ = image.shape
part_width = width // 5

splited_imgs = []
for i in range(5):
    left = i * part_width
    right = (i + 1) * part_width if i < 4 else width  # 确保最后一个分块到图像右边界
    
    split_img = image[:, left:right]
    splited_imgs.append(split_img)

merge_imgs = []
merge_imgs.append(splited_imgs[0])
for i in select_id:
    merge_imgs.append(splited_imgs[i])
merge_imgs = np.concatenate(merge_imgs, axis=1)
print(merge_imgs.shape)
resized_img = cv2.resize(merge_imgs, (merge_imgs.shape[1]//1, merge_imgs.shape[0]//1), interpolation=cv2.INTER_AREA)
print(resized_img.shape)
cv2.imwrite(save_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 85])