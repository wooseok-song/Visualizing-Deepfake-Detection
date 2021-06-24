from __future__ import print_function, division
import os
import torch
from skimage import io, transform  # 이미지 I/O와 변형을 위해 필요한 library
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import time
import copy
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import warnings
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_path = '/root/deepfake'

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4) #4 class Classification

data_transforms = {                                                       #data transformation
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),               # PIL 이미지 또는 numpy 이미지를 torch(tensor) 이미지로 변환
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class mdataset(torch.utils.data.Dataset):  # my customize dataset
    def __init__(self, data_path, label, phase, transform=None):  # constructor
        self.label = label  # about csv_file name
        self.data_path = data_path  # about Dataset path
        self.transform = transform  # about Transform
        self.phase = phase
        self.dirlist = os.listdir(self.data_path + str(self.phase))

    def __len__(self):
        return len(self.dirlist)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_path, str(self.phase), self.dirlist[idx])
        image = io.imread(img_name)
        label = self.label
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return image, label

attribute = mdataset(data_path, 0, '0.attribute', transform=data_transforms['train'])
swap = mdataset(data_path, 1, '1.swap', transform=data_transforms['train'])
expression = mdataset(data_path, 2, '2.expression', transform=data_transforms['train'])
real = mdataset(data_path, 3, '3.real', data_transforms['train'])
dataset = attribute + swap + expression + real

train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True)
train_data, valid_data = train_test_split(train_data, test_size=0.3, shuffle=True)

batch_size = 32
dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4
                                                   )

dataloaders['valid'] = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4
                                                   )

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # set gpu


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:              # if phase == 'valid'
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train (train 상태일 때만, 연산 기록을 추적)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)

            # 각 epoch 마다
            if phase == 'train':
                scheduler.step() # learning rate를 조절하기위해 scheduler 사용

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'Efficient.pt')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


model = model.to(device)

criterion = nn.CrossEntropyLoss()

#Choose Adam or SGD
optimizer_ft = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

""" #SGD case
optimizer_ft =optim.SGD(model.parameters(),
                     lr=0.05,
                     momentum=0.9,
                     weight_decay=1e-4)
"""

lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda) # 스케줄러 함수 설정

model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion, optimizer_ft,
                                                                                      exp_lr_scheduler, num_epochs=100)

print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx])) # valid에서 가장 좋았을때의 idx(epoch #)와 acc, loss를 출력




fig, ax1 = plt.subplots()   #Draw graph about acc&loss
ax1.plot(train_acc, 'b-')
ax1.plot(valid_acc, 'r-')
plt.plot(best_idx, valid_acc[best_idx], 'ro')
ax1.set_xlabel('epoch')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('acc', color='k')
ax1.tick_params('y', colors='k')

ax2 = ax1.twinx()
ax2.plot(train_loss, 'g-')
ax2.plot(valid_loss, 'k-')
plt.plot(best_idx, valid_loss[best_idx], 'ro')
ax2.set_ylabel('loss', color='k')
ax2.tick_params('y', colors='k')

fig.tight_layout()
plt.show()
