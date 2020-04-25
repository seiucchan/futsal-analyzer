from modules.yolo import *
from modules.sort import *
from modules.functions import *
from utils import *
import os, sys, time, datetime, random
sys.path.append(os.pardir)


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from IPython.display import clear_output


config_path='../config/yolov3.cfg'
weights_path='../weights/yolov3.weights'
class_path='../data/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=416).to("cpu")
model.load_darknet_weights(weights_path)
# model.cuda()
model.eval()
classes = utils.load_classes(class_path)
# Tensor = torch.cuda.FloatTensor
Tensor = torch.Tensor


videopath = '../seiucchanvideo.mp4'

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

vid = cv2.VideoCapture(videopath)
ret, frame = vid.read()
img = frame
wname = "MouseEvent"
cv2.namedWindow(wname)
npoints = 5
ptlist = PointList(npoints)
cv2.setMouseCallback(wname, onMouse, [wname, img, ptlist])
cv2.waitKey()
cv2.destroyAllWindows()
mot_tracker = Sort()
count_boxes = []

while(True):
    for ii in range(4000):
        ret, frame = vid.read()

        if type(frame) == type(None):
            print('ratio:', np.mean(count_boxes) / 10)
            sys.exit(0)

        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg, img_size, Tensor, model)
        out_box_indices = []
        detections_ = np.array(detections)
        detections = list()
        person_id = 0.0
        for det in detections_:
            if det[6] == person_id:
                detections.append(det)
        detections = np.array(detections)

        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0)*(img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0)*(img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        for i, coord in enumerate(detections):
            x1, y1, x2, y2 = coord[:4]
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            y2 = y1 + box_h
            x2 = x1 + box_w
            pt = ((x1 + x2) // 2, y2 - (box_h // 5) )
            # pt = revert_coord(pt)

            result =  cv2.pointPolygonTest(ptlist.ptlist, pt, False)
            if result < 0:
                out_box_indices.insert(0, i)
            # else:
            #     print('---', x1, y1, x2, y2)
        # sys.exit(0)
        for i in out_box_indices:
            # sys.exit(0)
            detections = np.delete(detections, i, 0)
        count_boxes.append(len(detections))

        detections = torch.tensor(detections)

        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(0)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                # print('===', x1, y1, x1+box_w, y1+box_h)
        cv2.imshow("mot_tracker", frame)
        key = cv2.waitKey(1) & 0xFF
