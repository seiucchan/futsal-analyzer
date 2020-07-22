import cv2
import numpy as np

from utils.change_coord import change_coord


def predict_ball_holder(bboxes, preds, M, frame, img_size):
    pad_x, pad_y, unpad_h, unpad_w = change_coord(frame, img_size)
    player_positions = []
    ball_position = []
    distances = []
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

        if cls_id == 0:
            player_positions.append(pts_trans)
        else:
            ball_position.append(pts_trans)
        
    if ball_position:
        player_positions = np.array(player_positions)
        ball_position = np.array(ball_position)
        
        for player_position in player_positions:
            # distances[i] = np.linalg.norm(ball_position - player_position)
            distances.append(np.linalg.norm(ball_position-player_position))
        ball_holder_index = distances.index(min(distances))

        if preds[ball_holder_index] == 0 | preds[ball_holder_index] == 1:
            ball_holder = 1
        else:
            ball_holder = 2
    else:
        ball_holder = 0

    return ball_holder

def calculate_posession(ball_holders):
    team1 = []
    team2 = []
    for ball_holder in ball_holders:
        if ball_holder == 1:
            team1.append(ball_holder)
        elif ball_holder == 2:
            team2.append(ball_holder)
    team1_posession_rate = len(team1) / (len(team1) + len(team2))
    team2_posession_rate =len(team2) / (len(team1) + len(team2))
    posession_rate = [team1_posession_rate, team2_posession_rate]

    return posession_rate