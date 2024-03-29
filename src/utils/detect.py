from modules.yolo.utils import non_max_suppression
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image


def detect_image(img, img_size, model, device, conf_thres, nms_thres, mask, diff_mode):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    # print("h-w:", max(int((imh-imw)/2), 0))  # ->0
    # print("w-h:", max(int((imw-imh)/2), 0))  # ->91
    img_transforms = transforms.Compose([
         transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0),
                         max(int((imw-imh)/2), 0),
                         max(int((imh-imw)/2), 0),
                         max(int((imw-imh)/2), 0)),
                        (128, 128, 128)),
         transforms.ToTensor(),
     ])
    # convert image to Tensor
    if diff_mode == 1:
        img = img_transforms(img).float()
        mask_transforms = transforms.Compose([
         transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0),
                         max(int((imw-imh)/2), 0),
                         max(int((imh-imw)/2), 0),
                         max(int((imw-imh)/2), 0))),
         transforms.ToTensor(),
        ])
        mask = mask_transforms(mask).float()
        inp = torch.cat([img, mask], axis=0)
        inp = inp.unsqueeze(0)
    else:
        inp = img_transforms(img).float().unsqueeze_(0)
    print(inp.shape)
    inp = inp.to(device)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(inp)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        print(detections)
    return detections[0] if detections[0] is not None else []
