#!/usr/bin/env python
# coding: utf-8

# # 데이터 불러오기
# 

# In[1]:


import torch 
import random

import torchvision
from torchvision import datasets 
from torchvision import transforms

import glob
import os
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from tqdm import tqdm 

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf


# In[ ]:


# import matplotlib.pyplot as plt
# plt.rc('font', family='NanumBarunGothic')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().system("cp '/content/drive/MyDrive/Project/open.zip' './'")
get_ipython().system('unzip -q open.zip -d remodel/ # remodel이라는 디렉토리 생성성')


# In[2]:


data_dir = './remodel/'


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# # 데이터 전처리 시작

# In[4]:


# origin = df


# In[5]:


filepath = glob.glob('/kaggle/input/remodel/train/*/*.png')


# In[6]:


df = pd.DataFrame(columns=['image','label'])
df['image'] = filepath
df['label']=df['image'].str.split('/').str[-2]


# In[7]:


df['label'].nunique()


# In[8]:


encoding = LabelEncoder()
df['label_encoding'] = encoding.fit_transform(df['label'])
df['label_encoding'].nunique()
# encoding.inverse_transform (encoding 한 코드 다시 원본으로 )


# In[9]:


train_set, valid_set , _,_ = train_test_split(df,df['label_encoding'],test_size=0.2, stratify=df['label_encoding'],random_state=42)
# stratify : class 비율을 일정하게 만들어 준다. Target 값으로 넣어주면 그 값의 비율에 맞게 나누어 주기 때문에 꼭 필요!


# #나만의 데이터셋 만들기

# In[12]:


transform = A.Compose([A.Resize(350, 350), A.Normalize(),
                    A.HorizontalFlip(p=0.3), A.VerticalFlip(p=0.3), ToTensorV2()])


# In[13]:


class MyData(Dataset):
    def __init__(self, image_filepath, label_filepath, transform=None):
        self.image_filepath = image_filepath
        self.label_filepath = label_filepath
        self.transform = transform

    def __len__(self):
        return len(self.image_filepath)

    def __getitem__(self,index):

        image = self.image_filepath[index]
        image = Image.open(image) # .convert('RGB')
        image = np.asarray(image)
        # image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.label_filepath is not None:
            label = self.label_filepath[index]
            return image, label
        else:
            return image


# In[14]:


trainset = MyData(train_set['image'].values, train_set['label_encoding'].values,transform=transform)
validset = MyData(valid_set['image'].values, valid_set['label_encoding'].values,transform=transform)


# In[ ]:


print(type(trainset), len(trainset))
print(type(validset), len(validset))


# In[ ]:


imageset = MyData(train_set['image'].values, train_set['label_encoding'].values, transform=transform)


# In[ ]:


imageset[0][1]


# In[ ]:


# 한국어를 실행을 하지 못함
labels_map = {7 :'몰딩수정', 9:'석고수정', 18:'훼손', 17:'피스', 4:'녹오염', 0:'가구수정', 10:'오염', 5:'들뜸' , 2:'곰팡이',
 14:'창틀,문틀수정', 12:'울음', 11:'오타공', 8:'반점', 6:'면불량', 15:'터짐', 16:'틈새과다' , 1:'걸레받이수정'
 ,13:'이음부불량', 3:'꼬임'}

figure, axes = plt.subplots(nrows=4, ncols=8, figsize=(14, 8))
axes = axes.flatten()

for i in range(32):
  rand_i = np.random.randint(0, 32)
  result = imageset[rand_i]
  image, label = result[0], result[1]
  axes[i].axis('off')
  axes[i].imshow(image.permute(1,2,0))
  axes[i].set_title(labels_map[label])


# # 데이터 적재

# In[16]:


batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)


# In[17]:


print(type(trainloader),len(trainloader))
print(type(validloader), len(validloader))


# In[18]:


images, labels = next(iter(trainloader))
images.size(), labels.size()


# In[ ]:


sample_image = images[0]
type(sample_image), sample_image.shape


# In[ ]:


get_ipython().system('nvidia-smi')


# In[19]:


import torchvision.models as models


# In[ ]:


# # First try
# batchsize = 10
# model = models.resnet50(weights=True)
# model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# model.to(device)
# learning_rate = 0.0001
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# # result: Epoch : 20/55....... Train Loss : 0.054 Valid Loss : 0.625 Valid Accuracy : 0.815


# # First try 
# ## ResNet50 Model
# - batchsize = 10
# - transform = A.Compose([A.Resize(224, 224), A.Normalize(), A.RandomRotate90(), A ToTensorV2()])
# - model = models.resnet50(weights=True)
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# - model.to(device)
# - learning_rate = 0.0001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# 
# #### **result: Epoch : 20/55....... Train Loss : 0.054 Valid Loss : 0.625 Valid Accuracy : 0.815**

# # Second try
# ## ResNet50 model with 
# - batchsize = 5
# - transform = A.Compose([A.Resize(224, 224), A.Normalize(), A.RandomRotate90(), A.HorizontalFlip(p=1), A.VerticalFlip(p=3), ToTensorV2()])
# - model = models.resnet50(weights=True)
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# - +dropout()
# - model.to(device)
# - learning_rate = 0.001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4,  - verbose=True)
# ***Epoch : 38/55....... Train Loss : 1.230 Valid Loss : 1.432 Valid Accuracy : 0.536**

# # Third try
# ## ResNet101 model with 
# - batchsize = 12
# - model = models.resnet101(weights=True)
# - transform = A.Compose([A.Resize(224, 224), A.Normalize(), A.RandomRotate90(), A.HorizontalFlip(p=3), A.VerticalFlip(p=3), ToTensorV2()])
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# 
# - model.to(device)
# - learning_rate = 0.001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4,  - verbose=True)
# ***Epoch : 40/55....... Train Loss : 0.532 Valid Loss : 0.973 Valid Accuracy : 0.689**
# #### best model at the moment

# # Fourth try
# ## ResNet101 model with 
# - batchsize = 16
# - model = models.resnet101(weights=True)
# - transform = A.Compose([A.Resize(224, 224), A.Normalize(), A.HorizontalFlip(p=3), A.VerticalFlip(p=3), ToTensorV2()])
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# 
# - model.to(device)
# - learning_rate = 0.0001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# ***Epoch : 12/55....... Train Loss : 0.044 Valid Loss : 0.758 Valid Accuracy : 0.796**
# 

# # Fifth try
# ## ResNet101 model with 
# - batchsize = 10
# - model = models.resnet101(weights=True)
# - transform = A.Compose([A.Resize(224, 224), A.Normalize(), A.HorizontalFlip(p=3), A.VerticalFlip(p=3), ToTensorV2()])
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# 
# - model.to(device)
# - learning_rate = 0.0001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# ***Epoch : 19/55....... Train Loss : 0.014 Valid Loss : 0.637 Valid Accuracy : 0.834**
# 

# # sixth try
# ## ResNet101 model with 
# - batchsize = 12
# - model = models.resnet101(weights=True)
# - transform = A.Compose([A.Resize(250, 250), A.Normalize(), A.HorizontalFlip(p=3), A.VerticalFlip(p=3), ToTensorV2()])
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# 
# - model.to(device)
# - learning_rate = 0.001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# ***Epoch : 41/55....... Train Loss : 0.487 Valid Loss : 0.904 Valid Accuracy : 0.717**

# # seventh try
# ## ResNet101 model with 
# - batchsize = 10
# - model = models.resnet101(weights=True)
# - transform = A.Compose([A.Resize(400, 400), A.Normalize(), A.HorizontalFlip(p=0.3), A.VerticalFlip(p=0.3), ToTensorV2()])
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# - model.to(device)
# - learning_rate = 0.001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# - epoch = 100
# ***Epoch : 37/100....... Train Loss : 0.789 Valid Loss : 1.131 Valid Accuracy : 0.639**
# 

# # eighth try
# ## ResNet101 model with 
# - batchsize = 32
# - model = models.resnet101(weights=True)
# - transform = A.Compose([A.Resize(350, 350), A.HorizontalFlip(p=0.3), A.VerticalFlip(p=0.3),A.Normalize(), ToTensorV2()])
# - model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# - model.to(device)
# - learning_rate = 0.001
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# - scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
# - epoch = 100
# 

# # 모델 컴파인

# In[20]:


#model = ResNet50()
model = models.resnet101(weights=True)
model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
model.to(device)


# In[ ]:


out = model(images.to(device))
out.shape


# In[ ]:


for name, parameter in model.named_parameters():
    print(name, parameter.size())


# In[21]:


learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)


# In[ ]:


#!pip install torchsummary

from torchsummary import summary
#summary(model, (3, 224, 224))


# # 모델 훈련

# In[22]:


writer = SummaryWriter()

def train_loop(model, trainloader, loss_fn, epochs, optimizer):  
  steps = 0
  steps_per_epoch = len(trainloader) 
  min_loss = 1000000
  max_accuracy = 0
  trigger = 0
  patience = 7 

  for epoch in range(epochs):
    model.train() # 훈련 모드
    train_loss = 0
    for images, labels in trainloader: # 이터레이터로부터 next()가 호출되며 미니배치를 반환(images, labels)
      steps += 1
      # images, labels : (torch.Size([16, 3, 224, 224]), torch.Size([16]))
      # 0. Data를 GPU로 보내기
      images, labels = images.to(device), labels.to(device)

      # 2. 전방향(forward) 예측
      predict = model(images) # 예측 점수
      loss = loss_fn(predict, labels) # 예측 점수와 정답을 CrossEntropyLoss에 넣어 Loss값 반환

      # 3. 역방향(backward) 오차(Gradient) 전파
      optimizer.zero_grad() # Gradient가 누적되지 않게 하기 위해
      loss.backward() # 모델파리미터들의 Gradient 전파

      # 4. 경사 하강법으로 모델 파라미터 업데이트
      optimizer.step() # W <- W -lr*Gradient

      train_loss += loss.item()
      if (steps % steps_per_epoch) == 0 : 
        model.eval() # 평가 모드 : 평가에서 사용하지 않을 계층(배치 정규화, 드롭아웃)들을 수행하지 않게 하기 위해서
        valid_loss, valid_accuracy = validate(model, validloader, loss_fn)

        # tensorboard 시각화를 위한 로그 이벤트 등록
        writer.add_scalar('Train Loss', train_loss/len(trainloader), epoch+1)
        writer.add_scalar('Valid Loss', valid_loss/len(validloader), epoch+1)
        writer.add_scalars('Train Loss and Valid Loss',
                          {'Train' : train_loss/len(trainloader),
                            'Valid' : valid_loss/len(validloader)}, epoch+1)
        writer.add_scalar('Valid Accuracy', valid_accuracy, epoch+1)
        # -------------------------------------------

        print('Epoch : {}/{}.......'.format(epoch+1, epochs),            
              'Train Loss : {:.3f}'.format(train_loss/len(trainloader)), 
              'Valid Loss : {:.3f}'.format(valid_loss/len(validloader)), 
              'Valid Accuracy : {:.3f}'.format(valid_accuracy)            
              )
        
        # Best model 저장    
        # option 1 : valid_loss 모니터링
        # if valid_loss < min_loss: # 바로 이전 epoch의 loss보다 작으면 저장하기
        #   min_loss = valid_loss
        #   best_model_state = deepcopy(model.state_dict())          
        #   torch.save(best_model_state, 'best_checkpoint.pth')     
        
        # option 2 : valid_accuracy 모니터링      
        if valid_accuracy > max_accuracy : # 바로 이전 epoch의 accuracy보다 크면 저장하기
          max_accuracy = valid_accuracy
          best_model_state = deepcopy(model.state_dict())          
          torch.save(best_model_state, 'best_checkpoint.pth')  
        # -------------------------------------------

        # Early Stopping (조기 종료)
        if valid_loss > min_loss: # valid_loss가 min_loss를 갱신하지 못하면
          trigger += 1
          print('trigger : ', trigger)
          if trigger > patience:
            print('Early Stopping !!!')
            print('Training loop is finished !!')
            writer.flush()   
            return
        else:
          trigger = 0
          min_loss = valid_loss
        # -------------------------------------------

        # Learning Rate Scheduler
        scheduler.step(valid_loss)
        # -------------------------------------------
        
  writer.flush()
  return  


# In[23]:


def validate(model, validloader, loss_fn):
  total = 0   
  correct = 0
  valid_loss = 0
  valid_accuracy = 0

  # 전방향 예측을 구할 때는 gradient가 필요가 없음음
  with torch.no_grad():
    for images, labels in validloader: # 이터레이터로부터 next()가 호출되며 미니배치를 반환(images, labels)      
      # images, labels : (torch.Size([16, 3, 224, 224]), torch.Size([16]))
      # 0. Data를 GPU로 보내기
      images, labels = images.to(device), labels.to(device)

      # 1. 입력 데이터 준비
      # not Flatten !!
      # images.resize_(images.size()[0], 784)

      # 2. 전방향(Forward) 예측
      logit = model(images) # 예측 점수
      _, preds = torch.max(logit, 1) # 배치에 대한 최종 예측
      # preds = logit.max(dim=1)[1] 
      correct += int((preds == labels).sum()) # 배치 중 맞은 것의 개수가 correct에 누적
      total += labels.shape[0] # 배치 사이즈만큼씩 total에 누적

      loss = loss_fn(logit, labels)
      valid_loss += loss.item() # tensor에서 값을 꺼내와서, 배치의 loss 평균값을 valid_loss에 누적

    valid_accuracy = correct / total
  
  return valid_loss, valid_accuracy


# In[ ]:


epochs = 100
get_ipython().run_line_magic('time', 'train_loop(model, trainloader, loss_fn, epochs, optimizer)')
writer.close()


# #모델 예측

# In[ ]:


test_set = glob.glob('/kaggle/input/remodel/test/*.png')
test_set.sort()
testset = MyData(test_set, None,transform=transform)


# In[ ]:


testloader = DataLoader(testset,batch_size=batch_size, shuffle=False)


# In[ ]:


images = next(iter(testloader))


# In[ ]:


best_checkpoint_dict = torch.load('/kaggle/working/best_checkpoint.pth')


# In[ ]:


model.load_state_dict(best_checkpoint_dict)


# In[ ]:


def inference(model, testloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(testloader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = encoding.inverse_transform(preds)
    return preds
preds = inference(model, testloader, device)


# In[ ]:


submit = pd.read_csv('/kaggle/input/remodel/sample_submission.csv')


# In[ ]:


submit['label'] = preds
submit['label'].nunique()


# In[ ]:


submit.to_csv('./baseline_submit.csv', index=False)


# In[ ]:




