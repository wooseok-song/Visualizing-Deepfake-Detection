Visualizing Deep Fake Detection
===================================

Inha univ. Capstone Design      


##1.Introduction
최근 Deep Fake를 이용한 범죄들이 증가하면서  Deep Fake Detection의 역할이 중요해 졌다. 
관련 자료들을 찾아보던중 기존의 기법들은 단순히 영상의 Real/Fake만을 판단한다는것을 알게
되었고 왜 이러한 판단을 냈는지에 대한 설명이 부족한 사실을 알게 되었다.
본 프로젝트에서는 Real/Fake를 판단할 뿐만 아니라 왜 그러한 Prediction을 냈는지
에 대해 시각화를 통해 결과를 설명 할 수 있는 방법을 프로젝트를 진행하면서 찾아보았다.



##2. Training

###2.1 Dataset



Public Dataset From Face-Forensics++ 
Video를 Frame단위로 끊어서 데이터 셋 구축


###2.2 Model

###EfficientNet-b0

```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4) #4 class Classification

```
EfficientNet-b0를 이용해 Training 진행

Ref)  https://github.com/lukemelas/EfficientNet-PyTorch

###2.3 Hyperparameter


##3. Visualizing
Explainable AI기법에는 Backpropagation-Based , Approximation-Based method 등등 여러가지 기법이 
존재한다. 본 프로젝트에서는 가장 효율적이고 적용하기 쉬운 Backpropagation based method를 사용한다.
또한 CNN구조를 바꾸지 않아도 되는 Grad-CAM기법을 이용해 설명가능한 output을 낸다.
###Grad-CAM
```python
        from pytorch_grad_cam import GradCAM
        cam = methods[args.method](model=model,
                                   target_layer=target_layer,
                                   use_cuda=args.use_cuda)
        rgb_img = Image.open(data_path+name)

        input_tensor, rgb = preprocess(rgb_img, mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])  
        rgb = rgb / 255
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,target_category=None)
        cam_image = show_cam_on_image(rgb, grayscale_cam)
    
        cv2.imwrite(save_path+str('(Real)res_')+name,cam_image)

```




##4.Result


