import cv2
import numpy as np

from utils.change_coord import change_coord


def calculate_matrix():
    p_original = np.float32([[25, 268], [593, 1150], [1250, 146], [837, 52]])
    p_trans = np.float32([[140, 110], [140, 610], [1140, 610], [1140, 110]])
    M = cv2.getPerspectiveTransform(p_original, p_trans)

    return M

def generate_plane_field():
    output_img = np.zeros([720, 1280, 3])
    cv2.line(output_img, (140, 110), (140, 610), (255, 255, 255), thickness=5, lineType=cv2.LINE_4)
    cv2.line(output_img, (140, 110), (1140, 110), (255, 255, 255), thickness=5, lineType=cv2.LINE_4)
    cv2.line(output_img, (1140, 110), (1140, 610), (255, 255, 255), thickness=5, lineType=cv2.LINE_4)
    cv2.line(output_img, (140, 610), (1140, 610), (255, 255, 255), thickness=5, lineType=cv2.LINE_4)
    cv2.line(output_img, (640, 110), (640, 610), (255, 255, 255), thickness=5, lineType=cv2.LINE_4)

    return output_img

def draw_player_positions(frame, bboxes, M, output_img, img_size, preds):
    pad_x, pad_y, unpad_h, unpad_w = change_coord(frame, img_size)
    i = 0
    for x1, y1, x2, y2, _, _, cls_id in bboxes:
        x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])
        x2 = int(((x2 - pad_x // 2) / unpad_w) * frame.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
        y2 = int(((y2 - pad_y // 2) / unpad_h) * frame.shape[0])
        x = x1 + (x2 - x1) / 2
        y = y2
        object_position = np.float32([[[x, y]]])
        pts_trans = cv2.perspectiveTransform(object_position, M)
        pts_trans = tuple(pts_trans[0][0])
        if cls_id == 0:
            if preds[i] == 0:
                cv2.circle(output_img, pts_trans, 20, (0, 255, 255), -1)
            elif preds[i] == 1:
                cv2.circle(output_img, pts_trans, 20, (255, 255, 255), -1)
            elif preds[i] == 2:
                cv2.circle(output_img, pts_trans, 20, (0, 0, 255), -1)
            else:
                cv2.circle(output_img, pts_trans, 20, (255, 0, 0), -1)
        else:
            cv2.circle(output_img, pts_trans, 10, (0, 255, 0), -1)

        i += 1


    return output_img