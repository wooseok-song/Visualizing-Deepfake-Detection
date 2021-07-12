Visualizing Deep Fake Detection
===================================

Inha univ. Capstone Design      


## 1.Introduction
  최근 Deep Fake를 이용한 범죄들이 증가하면서  Deep Fake Detection의 역할이 중요해 졌다. 
관련 자료들을 찾아보던중 기존의 기법들은 단순히 영상의 Real/Fake만을 판단한다는것을 알게
되었고 왜 이러한 판단을 냈는지에 대한 설명이 부족한 사실을 알게 되었다.
본 프로젝트에서는 Real/Fake를 판단할 뿐만 아니라 왜 그러한 Prediction을 냈는지
에 대해 시각화를 통해 결과를 설명 할 수 있는 방법을 프로젝트를 진행하면서 찾아보았다.





### 1.1 Face Manipulation Method
+ Face Attribute:여러가지 얼굴 속성(머리 색, 안경 착용, 수염 등)을 적용한 기법

+ Face Swap: Source의 얼굴을 Target의 얼굴로 바꾼 기법 우리가 잘 알고 있는 Deep Fake가 이 기법으로 생성된다.
+ Face Expression: Source의 표정을 Target의 표정으로 바꾼 기법. 입모양을 바꿔주는 기법.

### 1.2 System Flow

![image](https://user-images.githubusercontent.com/55542020/125224617-eb308080-e308-11eb-9921-748fbcfe894f.png)


## 2. Training

### 2.1 Dataset

- Public Dataset From Face-Forensics++ videos  해당 데이터셋의 일부를 kaggle에서 다운로드 받아 Frame단위로 끊어서 데이터 셋 구축<br/> 
  ref)https://www.kaggle.com/sorokin/faceforensics
  
  

   

- Face Attribute Case 경우에는 데이터 셋이 존재하지 않아 Face APP을 이용해 데이터 셋 수집<br/>
ref)https://apps.apple.com/us/app/faceapp-ai-face-editor/id1180884341
  


- DataSet Spec

|Label|How|Images|
|:-------:|:----|-----|
|Face Attribute| Face APP을 이용해 직접 만듦 |410장|
|Face Swap| Face Forensics++(Face Swap) |401장|
|Face Expression|Face Forensics++(Neural Texture) |401장|
|Real|Face Forensics++(Original Image) |401장|


### 2.2 Model

### EfficientNet-b0

```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4) #4 class Classification

```
EfficientNet-b0를 이용해 Training 진행

Ref)  https://github.com/lukemelas/EfficientNet-PyTorch

###  2.3 HyperParameter

| Setting | what | with |
|:------:|:------:|------|
|Epoch|200||
|Learning Rate|0.0005||
|Batch size|256||
|Loss| Cross Entropy Loss||
|Optimizer|ADAM||


###  2.4 Train Result
1. Accuracy graph


![image](https://user-images.githubusercontent.com/55542020/125224773-25018700-e309-11eb-8831-db2444a9b7b8.png)


2. Loss graph

![image](https://user-images.githubusercontent.com/55542020/125224778-2763e100-e309-11eb-9222-f91cfadf78b4.png)


##  3. Visualization
Explainable AI기법에는 Backpropagation-Based , Approximation-Based method 등등 여러가지 기법이 
존재한다. 본 프로젝트에서는 가장 효율적이고 적용하기 쉬운 Backpropagation based method를 사용한다.
또한 CNN구조를 바꾸지 않아도 되는 Grad-CAM기법을 이용해 설명가능한 output을 낸다.
### 3.1 Grad-CAM
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
Ref) https://github.com/jacobgil/pytorch-grad-cam



## 4.Result
### 4.1 Result Images
![Result1](https://user-images.githubusercontent.com/55542020/123218381-a0d89280-d506-11eb-91a5-6e3b64a0306d.png)

1. 설명: (a)는 조작되지 않은 원본 이미지, (b)는 Face Attribute 기법으로 조작된 이미지, (c)는 결과를 시각화 한 이미지이다. (a)와 (b)를 비교해보면 (b)
이미지에서는 머리 색이 금색에서 검은색으로 조작된 것을 확인할 수 있다. 이미지를 위 모델에 넣어주게 되면 (c)번과 같은 Heat-Map 형태의
이미지가 출력으로 나온다. 이때 Heat-Map의 색이 빨간색에 가까워 지면 해당 영역이 예측에 가장 영향을 많이 준 영역을 의미한다.
(c)이미지를 보면 현재 머리카락에 Heat-Map이 형성된 것을 확인할 수 있다. 즉 모델이 (b)이미지가 Face Attribute 기법으로
조작되었다고 판단한 뒤 (c)  이미지와 같이 머리 부분이 조작되었다는 사실을 설명해준다.

### 4.2 Confusion Matrix
![Confusion matrix](https://user-images.githubusercontent.com/55542020/123218132-50613500-d506-11eb-80aa-994b33c85e29.png)


위 결과를 통해서 Predicition의 정도를 확인 할 수 있다.


