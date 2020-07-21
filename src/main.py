import datetime
import os
import random
import sys
import time
from typing import NewType, Tuple

import click
import cv2
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from modules.yolo import Darknet
from modules.yolo import utils
# from modules.sort import Sort
from modules.color_classification import team_classifier, team_input, color_mean
from modules.plane_field_conversion import calculate_matrix, generate_plane_field, draw_player_positions
from utils.detect import detect_image
from utils.change_coord import change_coord
from utils.visualization import visualization
from utils.filter_court import PointList, onMouse, filter_court
from utils.paint_black import paint_black
from utils.max_ball_selection import max_ball_selection


VideoRead = NewType('VideoRead', cv2.VideoCapture)
VideoWrite = NewType('VideoWrite', cv2.VideoWriter)


@click.command()
@click.option('--config_path', default='config/yolov3.cfg')
@click.option('--weights_path', default='weights/yolov3.weights')
@click.option('--class_path', default='data/coco.names')
@click.option('--img_size', default=416)
@click.option('--conf_thres', default=0.8)
@click.option('--nms_thres', default=0.4)
@click.option('--input', default='data/seiucchanvideo.mp4')
@click.option('--output', default='data/out.mp4')
@click.option('--npoints', default=5)
@click.option('--wname', default='MouseEvent')
@click.option('--gpu_id', default=1)
@click.option('--is_show', default=True)
@click.option('--write_video', default=True)
def main(config_path,
         weights_path,
         class_path,
         img_size,
         conf_thres,
         nms_thres,
         input,
         output,
         npoints,
         wname,
         gpu_id, 
         is_show,
         write_video):

    # device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = Darknet(config_path, img_size).to("cpu")
    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path)
    else:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.to(device)
    model.eval()

    classes = utils.load_classes(class_path)

    # tracker = Sort()

    # input video
    cap, origin_fps, num_frames, width, height = load_video(input)
    # output video
    if write_video:
        writer = video_writer(output, origin_fps, width, height)

    # user input
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError()
    cv2.namedWindow(wname)
    ptlist = PointList(npoints)
    cv2.setMouseCallback(wname, onMouse, [wname, frame, ptlist])
    cv2.waitKey()
    cv2.destroyAllWindows()

    player_cluster1 = team_input(frame, 1)
    player_cluster2 = team_input(frame, 2)
    player_cluster3 = team_input(frame, 3)
    player_cluster4 = team_input(frame, 4)

    player_cluster1 = color_mean(frame, player_cluster1)
    player_cluster2 = color_mean(frame, player_cluster2)
    player_cluster3 = color_mean(frame, player_cluster3)
    player_cluster4 = color_mean(frame, player_cluster4)

    M = calculate_matrix()

    cnt = 1
    while(cap.isOpened()):
        print('')
        ret, frame = cap.read()
        if not ret:
            break

        print('-----------------------------------------------------')
        print('[INFO] Count: {}/{}'.format(cnt, num_frames))

        pilimg = Image.fromarray(frame)
        bboxes = detect_image(pilimg, img_size, model, device, conf_thres, nms_thres)

        out = frame
        output_img = generate_plane_field()
        if bboxes is not None:
            bboxes = filter_court(bboxes, pilimg, img_size, ptlist)
            bboxes = max_ball_selection(bboxes)
            print(f"[INFO] {len(bboxes)} persons are detected.")
            preds = team_classifier(frame, pilimg, img_size, bboxes, player_cluster1, player_cluster2, player_cluster3, player_cluster4)
            out = visualization(bboxes, pilimg, img_size, frame, classes, frame, is_show, preds)
            output_img = draw_player_positions(frame, bboxes, M, output_img, img_size, preds)

        if write_video:
            writer.write(np.concatenate([out, output_img], axis=0))
        cnt += 1


    print("[INFO] End processing.")
    cap.release()
    writer.release()


def load_video(path: str) -> Tuple[VideoRead, int, int, int, int]:
    print('[INFO] Loading "{}" ...'.format(path))
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))
    return cap, fps, frames, W, H


def video_writer(path: str, fps: int, width: int, height: int,
                 resize_factor: float =None) -> VideoWrite:
    print("[INFO] Save output video in", path)
    if resize_factor is not None:
        width = int(width * resize_factor)
        height = int(height * resize_factor)
        width -= width % 4
        height -= height % 4
    height *= 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 20.0, (width, height))
    return writer


if __name__ == '__main__':
    main()
