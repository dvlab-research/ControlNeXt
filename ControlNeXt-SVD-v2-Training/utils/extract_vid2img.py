

import argparse
import random
import os
from decord import VideoReader
import cv2


"""
python -m utils.extract_vid2img  /home/llm/bhpeng/generation/svd-temporal-controlnet/proj/dataset/cropped/v2/users/肚脐小师妹/7296828856386784548.mp4 /home/llm/bhpeng/generation/svd-temporal-controlnet/validation_demo/test/0
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path",
                        type=str,
                        help="path to the video")
    parser.add_argument("save_dict",
                        type=str,
                        help="path to the save_dict")
    parser.add_argument("--interval_frame",
                        type=int,
                        default=2)
    parser.add_argument("--sample_n_frames",
                        type=int,
                        default=1)
    args = parser.parse_args()


    video_path = args.video_path
    pose_path = video_path.replace("/users", "/pose")

    save_video_path = os.path.join(args.save_dict, "rgb")
    save_pose_path = os.path.join(args.save_dict, "pose")

    if not os.path.exists(save_video_path):
        os.makedirs(save_video_path)
    if not os.path.exists(save_pose_path):
        os.makedirs(save_pose_path)

    vr = VideoReader(video_path)
    length = len(vr)
    segment_length = args.interval_frame * args.sample_n_frames
    assert length >= segment_length, "Too short video..."
    bg_frame_id = random.randint(0, length - segment_length)
    frame_ids = list(range(bg_frame_id, bg_frame_id + segment_length, args.interval_frame))
    for idx, fid in enumerate(frame_ids):
        frame = vr[fid].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.imwrite(os.path.join(save_video_path, "{}.png".format(idx)), frame)
    

    vr = VideoReader(pose_path)
    for idx, fid in enumerate(frame_ids):
        frame = vr[fid].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.imwrite(os.path.join(save_pose_path, "{}.png".format(idx)), frame)
    

