import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from utils.change_coord import change_coord


def visualization(tracked_objects, pilimg, img_size, img, classes, frame, is_show):
    cmap = plt.get_cmap('tab20b')
    # colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    pad_x, pad_y, unpad_h, unpad_w = change_coord(pilimg, img_size)
    # for x1, y1, x2, y2, obj_id in tracked_objects:
    for x1, y1, x2, y2, _, _, _ in tracked_objects:
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
        # color = colors[int(obj_id) % len(colors)]
        # color = [i * 255 for i in color]
        color = [0, 255, 255]
        cls = classes[int(0)]
        cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
        # pred_id = obj_id - 1
        # pred_id = int(pred_id)
        print(preds, i)
        # print(pred_id.dtype)
        # print(preds[0].dtype)
        if preds[i] == 0:
            cv2.putText(frame, 'team1', (x1, y1), font, 1,(255,255,255),2,cv2.LINE_AA)
        elif preds[i] == 1:
            cv2.putText(frame, 'team2', (x1, y1), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        elif preds[i] == 2:
            cv2.putText(frame, 'GK', (x1, y1), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        i += 1
    cv2.imshow("mot_tracker", frame)
    
    if is_show:
        cv2.imshow("mot_tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    return frame
