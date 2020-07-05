import numpy as np
import cv2

from utils.change_coord import change_coord


def team_input(frame, cluster_num):
    bboxes = []
    bbox_coords = []
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('selectplayer{}'.format(cluster_num), frame)
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[0] + bbox[2])
        y2 = int(bbox[1] + bbox[3])

        bbox_coord = [x1, y1, x2, y2]

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, 2, 1)
        print("Press q to quit selecting this cluster")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            cv2.destroyAllWindows()
            break
    print(bbox_coord)
    return bbox_coord

def color_mean(frame, player):
    print(player)
    x1, y1, x2, y2 = player
    x3 = x1 + (x2-x1) // 3
    x4 = x2 - (x2-x1) // 3
    y3 = y1 + (y2-y1) // 3
    y4 = y2 - (y2-y1) // 3

    frame = frame[y3:y4, x3:x4]
    bgr_sum = np.sum(frame ,axis = (0,1))
    b_ratio,g_ratio,r_ratio = bgr_sum/np.sum(bgr_sum)        
    player_color = np.array([b_ratio,g_ratio,r_ratio])
    
    return player_color

def team_classifier(frame, pilimg, img_size, bboxes, player_cluster1, player_cluster2, player_cluster3, player_cluster4):
    preds = []
    pad_x, pad_y, unpad_h, unpad_w = change_coord(pilimg, img_size)
    for x1, y1, x2, y2, _, _, _ in bboxes:
        player = []
        color_checker = []
        x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])
        x2 = int(((x2 - pad_x // 2) / unpad_w) * frame.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
        y2 = int(((y2 - pad_y // 2) / unpad_h) * frame.shape[0])
        player = [int(x1), int(y1), int(x2), int(y2)]
        player_color = color_mean(frame, player)
        color_checker.append(np.linalg.norm(player_color-player_cluster1))
        color_checker.append(np.linalg.norm(player_color-player_cluster2))
        color_checker.append(np.linalg.norm(player_color-player_cluster3))
        color_checker.append(np.linalg.norm(player_color-player_cluster4))
        preds.append(color_checker.index(min(color_checker)))
        
    return preds