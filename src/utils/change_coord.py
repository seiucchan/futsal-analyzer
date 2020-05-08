import numpy as np
from PIL import Image

def change_coord(pilimg, img_size):
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0)*(img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0)*(img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    return pad_x, pad_y, unpad_h, unpad_w
