import numpy as np
import cv2
import torch
from utils.change_coord import change_coord

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

def filter_court(detections, pilimg, img_size, ptlist):
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
