from modules.yolo import Darknet
from modules.yolo import utils
from modules.sort import Sort
from utils.detect import detect_image
from utils.change_coord import change_coord
from utils.visualization import visualization
from utils.filter_court import PointList, onMouse, filter_court
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
@click.option('--config_path', default='weights/configs/yolov3.cfg')
@click.option('--weights_path', default='weights/yolov3.weights')
@click.option('--class_path', default='data/coco.names')
@click.option('--img_size', default=416)
@click.option('--conf_thres', default=0.8)
@click.option('--nms_thres', default=0.4)
@click.option('--videopath', default='data/seiucchanvideo.mp4')
@click.option('--npoints', default=5)
@click.option('--wname', default='MouseEvent')
@click.option('--gpu_id', default=1)
def main(config_path,
         weights_path,
         class_path,
         img_size,
         conf_thres,
         nms_thres,
         videopath,
         npoints,
         wname,
         gpu_id):

    # device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = Darknet(config_path, img_size).to("cpu")
    model.load_darknet_weights(weights_path)
    model.to(device)
    model.eval()

    classes = utils.load_classes(class_path)

    tracker = Sort()

    vid = cv2.VideoCapture(videopath)
    ret, frame = vid.read()
    if not ret:
        raise RuntimeError("Video cant be loaded.")

    cv2.namedWindow(wname)
    ptlist = PointList(npoints)
    cv2.setMouseCallback(wname, onMouse, [wname, frame, ptlist])
    cv2.waitKey()
    cv2.destroyAllWindows()

    cnt = 0
    while(True):
        ret, frame = vid.read()
        if not ret:
            break
        print(f"frame: {cnt}")

        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg, img_size, model, device)
        detections = filter_court(detections, pilimg, img_size, ptlist)
        
        if detections is not None:
            tracked_objects = tracker.update(detections.cpu())
            visualization(tracked_objects, pilimg, img_size, frame, classes, frame)
        
        cnt += 1

    print("End processing.")


if __name__ == '__main__':
    main()
