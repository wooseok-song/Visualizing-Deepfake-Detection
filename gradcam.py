import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms, utils
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM
import os
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms


def preprocess(img, mean=None, std=None):
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    resizing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    original = resizing(img.copy())
    original = np.array(original)
    return preprocessing(original).unsqueeze(0), original



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str,
                        default='/root/deepfake/12161741/week10/newdata/2.expression/eframe5.jpg',
                        help='Input image path')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()  # use_cuda args에서 상태가 True이고 cuda가 사용가능하면 use_cuda update
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both (CAM + Guided Back Propagation)
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM}


    if args.method not in list(
            methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)  # 4 class Classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,
                         2)
    model.load_state_dict(torch.load('expression.pt'))

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.layer4[-1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    data_path = '/root/deepfake/12161741/week10/newdata/3.real/'
    save_path = '/root/deepfake/12161741/week10/result/expression/'
    file_name=os.listdir(data_path)

    for name in file_name:

        cam = methods[args.method](model=model,
                                   target_layer=target_layer,
                                   use_cuda=args.use_cuda)
        rgb_img = Image.open(data_path+name)

        input_tensor, rgb = preprocess(rgb_img, mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])  #

        rgb = rgb / 255

        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=None)
        cam_image = show_cam_on_image(rgb, grayscale_cam)
        cv2.imwrite(save_path+str('(Real)res_')+name,cam_image)
