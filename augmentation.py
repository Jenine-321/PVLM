from albumentations import (
   PadIfNeeded, Compose
)
import numpy as np
from PIL import Image
import cv2
import torch

def strong_aug(p=.5):
    return Compose([
         PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_REPLICATE)
    ], p=p)

def augment(aug, image):
    return aug(image=image)['image']

class Aug(object):
    def __call__(self, img):
        aug = strong_aug(p=0.9)
        return Image.fromarray(augment(aug, np.array(img)))