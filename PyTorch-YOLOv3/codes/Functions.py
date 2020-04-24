from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *


import os
import sys
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import NullLocator



to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

class YOLO(object):
    def __init__(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = Darknet("config/yolov3.cfg", img_size=416).to(device)
        if "weights/yolov3.weights".endswith(".weights"):
            self.model.load_darknet_weights("weights/yolov3.weights")
        else:
            self.model.load_state_dict(torch.load("weights/yolov3.weights"))
        self.model.eval()
        print("__init__")

    def detect(self, frame):
        frame = Image.fromarray(frame)
        frame = frame.resize((416, 416))
        frame = to_tensor(frame)
        frame = frame.unsqueeze(0)

        with torch.no_grad():
            detections = self.model(frame)
            detections = non_max_suppression(detections, 0.8, 0.4)
        return detections



def draw_bbox(img, detections):
    classes = load_classes("data/coco.names")

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    bbox_coordinate = ()

    for img_i, (path, detections) in enumerate(zip(img, detections)):

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if detections is not None:
            detections = rescale_boxes(detections, 416, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            print(detections)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                print("bboxの確認")
                print(bbox)
                ax.add_patch(bbox)
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
                bbox_coordinate = bbox_coordinate + (round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item()))

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = "test1"
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
        return bbox_coordinate


if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("--image_folder", type=str)
     parser.add_argument("--img_size", type=int, default=416)
     opt = parser.parse_args()
     frame = transforms.ToTensor()(Image.open(opt.image_folder))
     detections = object_detection(frame)
     print(detections)
