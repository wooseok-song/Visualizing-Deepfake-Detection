import argparse

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')

    parser.add_argument('--image-path', type=str, default='/root/deepfake/12161777/newdata/1.swap/sfame401.jpg',
                       help='Input image path')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
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


    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")


    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
    print(model)
    # 우리가 학습시킨 얼굴모델의 가중치를 이용
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)



    model.load_state_dict(torch.load('my_model(efficientnet).pt'))

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # efficient : model._blocks[-1]
    # You can print the model to help chose the layer
    target_layer = model._blocks[-1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

     print(methods[args.method])
    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    target_category = None

    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)


    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam_efficient(attribute).jpg', cam_image)

    plt.imshow(cam_image)
    plt.show()