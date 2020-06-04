import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import datasets, preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from utils.change_coord import change_coord


def k_means(tracked_objects, pilimg, img_size, img):
    img = np.array(pilimg)
    print(img.shape[0], img.shape[2])
    bbox_list = []

    # get color data in bbox
    pad_x, pad_y, unpad_h, unpad_w = change_coord(pilimg, img_size)
    for x1, y1, x2, y2, obj_id in tracked_objects:
        print('start iter')
        # change coord
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x2 = int(((x2 - pad_x // 2) / unpad_w) * img.shape[1])
        y2 = int(((y2 - pad_y // 2) / unpad_h) * img.shape[0])

        print(np.sum(img[x1:x2, y1:y2] ,axis = (0,1)))

        bgr_sum = np.sum(img[x1:x2, y1:y2] ,axis = (0,1))
        b_ratio,g_ratio,r_ratio = bgr_sum/np.sum(bgr_sum)        
        bgr_list = [b_ratio,g_ratio,r_ratio]
        bbox_list.append(bgr_list)
    
    bbox_list_n = np.array(bbox_list)
    bbox_list_n = np.nan_to_num(bbox_list_n)
    bbox_list_n = np.float32(bbox_list_n)
    preds = KMeans(n_clusters=3).fit_predict(bbox_list_n)
    return preds
