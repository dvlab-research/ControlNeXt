import cv2
import numpy as np
from PIL import Image


def get_extractor(extractor_name):
    if extractor_name is None:
        return None
    if extractor_name not in EXTRACTORS:
        raise ValueError(f"Extractor {extractor_name} is not supported.")
    return EXTRACTORS[extractor_name]


def canny_extractor(image: Image.Image, threshold1=None, threshold2=None) -> Image.Image:
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    v = np.median(gray)

    sigma = 0.33
    threshold1 = threshold1 or int(max(0, (1.0 - sigma) * v))
    threshold2 = threshold2 or int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray, threshold1, threshold2)
    edges = Image.fromarray(edges).convert("RGB")
    return edges


def depth_extractor(image: Image.Image):
    raise NotImplementedError("Depth extractor is not implemented yet.")


def pose_extractor(image: Image.Image):
    raise NotImplementedError("Pose extractor is not implemented yet.")


EXTRACTORS = {
    "canny": canny_extractor,
}
