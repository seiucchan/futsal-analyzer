def max_ball_selection(bboxes):
    ball_probability = {}
    new_bboxes = []
    i = 0
    for bbox in bboxes:
        if bbox[6] == 1:
            ball_probability[i] = bbox[4]
        else:
            new_bboxes.append(bbox)
        i += 1
    if ball_probability:
        max_ball_probability_index = max(ball_probability)
        new_bboxes.append(bboxes[max_ball_probability_index])

    return new_bboxes