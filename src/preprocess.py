from __future__ import division

import torch
import numpy as np
import cv2


def resize_padding(img, inp_dim):
    """
    resize image with unchanged aspect ratio using padding
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img,
                               (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2: (h-new_h)//2 + new_h,
           (w-new_w)//2: (w-new_w)//2 + new_w, :] = resized_image

    return canvas


def prep_image(inp_dim, img=None, img_path=None, resize_pad=False):
    """
    Prepare image for inputting to the neural network. 

    Returns resize_padding Variable 
    """
    if img_path:
        origin_img = cv2.imread(img_path)
    else:
        origin_img = img

    assert img is not None, "Image is None"

    dim = origin_img.shape[1], origin_img.shape[0]
    if resize_pad:
        img = (resize_padding(origin_img, (inp_dim, inp_dim)))
    else:
        img = cv2.resize(origin_img, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, origin_img, dim
