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
import click


@click.command()
@click.option('--config_path', default='config/yolov3.cfg')
@click.option('--weights_path', default='weights/yolov3.weights')
@click.option('--class_path', default='data/coco.names')
@click.option('--img_size', default=416)
@click.option('--conf_thres', default=0.8)
@click.option('--nms_thres', default=0.4)
@click.option('--videopath', default='seiucchanvideo.mp4')
@click.option('--npoints', default=5)

def main(config_path,
         weights_path,
         class_path,
         img_size,
         conf_thres,
         nms_thres,
         videopath,
         npoints):

    # Load model and weights
    model = Darknet(config_path, img_size).to("cpu")
    model.load_darknet_weights(weights_path)
    # model.cuda()
    model.eval()
    classes = utils.load_classes(class_path)
    # Tensor = torch.cuda.FloatTensor
    Tensor = torch.Tensor



    vid = cv2.VideoCapture(videopath)
    ret, frame = vid.read()
    img = frame
    wname = "MouseEvent"
    cv2.namedWindow(wname)
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
                # print('ratio:', np.mean(count_boxes) / 10)
                sys.exit(0)

            pilimg = Image.fromarray(frame)
            detections = detect_image(pilimg, img_size, Tensor, model)
            detections_ = np.array(detections)
            detections = filter_coat(detections_, pilimg, img_size, ptlist)
            detections = torch.tensor(detections)

            if detections is not None:
                tracked_objects = mot_tracker.update(detections.cpu())
                visualization(tracked_objects, pilimg, img_size, img, classes, frame)


if __name__ == '__main__':
    main()
