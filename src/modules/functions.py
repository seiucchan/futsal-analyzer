from utils import *

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from IPython.display import clear_output


def revert_coord(coord: tuple, base: int =416,
                  imsize: tuple =(1280, 720)) -> tuple:
    new_x = coord[0] * imsize[0] // base
    new_y = coord[1] * imsize[1] // base
    return (new_x, new_y)

def detect_image(img, img_size, Tensor, model):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    print("h-w:",max(int((imh-imw)/2),0)) #->0
    print("w-h:", max(int((imw-imh)/2),0)) #->91
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    print("input_img:", input_img.shape)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 0.5, 0.4)
    return detections[0]

class PointList():
    def __init__(self, npoints):
        self.npoints = npoints
        self.ptlist = np.empty((npoints, 2), dtype=int)
        self.pos = 0

    def add(self, x, y):
        if self.pos < self.npoints:
            self.ptlist[self.pos, :] = [x, y]
            self.pos += 1
            return True
        return False


def onMouse(event, x, y, flag, params):
    wname, img, ptlist = params
    if event == cv2.EVENT_MOUSEMOVE:  # マウスが移動したときにx線とy線を更新する
        img2 = np.copy(img)
        h, w = img2.shape[0], img2.shape[1]
        cv2.line(img2, (x, 0), (x, h - 1), (255, 0, 0))
        cv2.line(img2, (0, y), (w - 1, y), (255, 0, 0))
        cv2.imshow(wname, img2)

    if event == cv2.EVENT_LBUTTONDOWN:  # レフトボタンをクリックしたとき、ptlist配列にx,y座標を格納する
        if ptlist.add(x, y):
            print('[%d] ( %d, %d )' % (ptlist.pos - 1, x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), 3)
            cv2.imshow(wname, img)
        else:
            print('All points have selected.  Press ESC-key.')
        if(ptlist.pos == ptlist.npoints):
            print(ptlist.ptlist)
            cv2.line(img, (ptlist.ptlist[0][0], ptlist.ptlist[0][1]),
                     (ptlist.ptlist[1][0], ptlist.ptlist[1][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[1][0], ptlist.ptlist[1][1]),
                     (ptlist.ptlist[2][0], ptlist.ptlist[2][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[2][0], ptlist.ptlist[2][1]),
                     (ptlist.ptlist[3][0], ptlist.ptlist[3][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[3][0], ptlist.ptlist[3][1]),
                     (ptlist.ptlist[4][0], ptlist.ptlist[4][1]), (0, 255, 0), 3)
            cv2.line(img, (ptlist.ptlist[4][0], ptlist.ptlist[4][1]),
                     (ptlist.ptlist[0][0], ptlist.ptlist[0][1]), (0, 255, 0), 3)

def filter_coat(detections, pilimg, img_size, ptlist):
    #exclude result of detection exceot person
    detections_ = np.array(detections)
    out_box_indices = []
    detections = list()
    person_id = 0.0
    for det in detections_:
        if det[6] == person_id:
            detections.append(det)
    detections = np.array(detections)

    pad_x, pad_y, unpad_h, unpad_w = change_coord(pilimg, img_size)
    # chose out_box_indices
    img = np.array(pilimg)

    for i, coord in enumerate(detections):
        x1, y1, x2, y2 = coord[:4]
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        y2 = y1 + box_h
        x2 = x1 + box_w
        pt = ((x1 + x2) // 2, y2 - (box_h // 5))

        result =  cv2.pointPolygonTest(ptlist.ptlist, pt, False)
        if result < 0:
            out_box_indices.insert(0, i)

    # delete out_box
    for i in out_box_indices:
        detections = np.delete(detections, i, 0)

    # count_boxes.append(len(detections))
    detections = torch.tensor(detections)
    return detections


def change_coord(pilimg, img_size):
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0)*(img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0)*(img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    return pad_x, pad_y, unpad_h, unpad_w

def visualization(tracked_objects, pilimg, img_size, img, classes, frame):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    pad_x, pad_y, unpad_h, unpad_w = change_coord(pilimg, img_size)
    for x1, y1, x2, y2, obj_id in tracked_objects:
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        color = colors[int(obj_id) % len(colors)]
        color = [i * 255 for i in color]
        cls = classes[int(0)]
        cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
    cv2.imshow("mot_tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    return key
