import numpy as np
import cv2


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
        y1 = int(bbox[1] + (bbox[3] - bbox[1]) // 3)
        x2 = int(bbox[0] + bbox[2])
        y2 = int((bbox[1] + bbox[3]) - (bbox[3] - bbox[1]) // 3) 

        bbox_coord = [x1, y1, x2, y2]

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, 2, 1)
        print("Press q to quit selecting this cluster")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            cv2.destroyAllWindows()
            break

    return bbox_coord

def color_mean(frame, player_cluster):
    x1, y1, x2, y2 = player_cluster
    frame = frame[y1:y2, x1:x2]
    player_color = np.mean(frame, axis=(0, 1))

    return player_color

def team_classifier():
    pass