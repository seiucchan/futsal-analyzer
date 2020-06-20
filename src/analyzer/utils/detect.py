from modules.yolo.utils import non_max_suppression
import torch
from torchvision import transforms
from torch.autograd import Variable


def detect_image(img, img_size, model, device):
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
    inp = img_transforms(img).float().unsqueeze_(0)
    inp = inp.to(device)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(inp)
        detections = non_max_suppression(detections, 0.5, 0.4)
    return detections[0]
