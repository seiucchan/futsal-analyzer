import cv2
import numpy as np

def paint_black(frame, ptlist):
    img = frame
    black = np.zeros((400, 1280, 3)).astype(img.dtype)
    img = np.concatenate([img, black], axis=0)

    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [ptlist]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(img, stencil)


    return result
