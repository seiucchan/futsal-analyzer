import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from utils.change_coord import change_coord


def visualization(tracked_objects, pilimg, img_size, img, classes, frame, is_show, preds, track_bbs_ids):
    cmap = plt.get_cmap('tab20b')
    font = cv2.FONT_HERSHEY_COMPLEX
    pad_x, pad_y, unpad_h, unpad_w = change_coord(pilimg, img_size)
    i = 0

    for x1, y1, x2, y2, _, _, cls_id in tracked_objects:
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        # color = colors[int(obj_id) % len(colors)]
        # color = [i * 255 for i in color]
        if cls_id  == 0.0:
            color = [0, 255, 255]
        else:
            color = [0, 0, 0]
        cls = classes[int(0)]
        cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)

        if cls_id == 0:
            if preds[i] == 0:
                cv2.putText(frame, '0', (x1, y1), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif preds[i] == 1:
                cv2.putText(frame, '1', (x1, y1), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            elif preds[i] == 2:
                cv2.putText(frame, '2', (x1, y1), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, '3', (x1, y1), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        i += 1
        
    for x1, y1, x2, y2, tracked_id in track_bbs_ids:
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        tracked_id = str(tracked_id)
        cv2.putText(frame, tracked_id, (int(x1), int(y1+10)), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("mot_tracker", frame)

    if is_show:
        cv2.imshow("mot_tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    return frame
