#!/usr/bin/env python
# coding: utf-8

# # 데이터 가져오기

# In[1]:


import torch 
import random

import torchvision
from torchvision import datasets 
from torchvision import transforms
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import Image

import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[2]:


seed = 50
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)                # 파이썬 난수 생성기 시드 고정
np.random.seed(seed)             # 넘파이 난수 생성기 시드 고정
torch.manual_seed(seed)          # 파이토치 난수 생성기 시드 고정 (CPU 사용 시)
torch.cuda.manual_seed(seed)     # 파이토치 난수 생성기 시드 고정 (GPU 사용 시)
torch.cuda.manual_seed_all(seed) # 파이토치 난수 생성기 시드 고정 (멀티GPU 사용 시)
torch.backends.cudnn.deterministic = True # 확정적 연산 사용
torch.backends.cudnn.benchmark = False    # 벤치마크 기능 해제
torch.backends.cudnn.enabled = False      # cudnn 사용 해제


# In[6]:


# 구글 코랩에서 데이터 가져오기
# from google.colab import drive
# drive.mount('/content/drive')
# !cp '/content/drive/MyDrive/Project/open.zip' './'
# !unzip -q open.zip -d remodel/ # remodel이라는 디렉토리 생성성


# In[3]:


data_dir = './remodel/'


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# # 데이터 전처리 with dataframe

# In[5]:


# for kaggle GPU
filepath = glob.glob('/kaggle/input/remodel/train/*/*.png')

# for google colab
# filepath = glob.glob('/content/remodel/train/*/*.png')


# In[6]:


#df라는 판다스를 생성하여 레이블을 생성
df = pd.DataFrame(columns=['image','label'])
df['image'] = filepath
df['label']=df['image'].str.split('/').str[-2]


# In[7]:


# 한글 레이블을 숫자 레이블로 바꾸기 위해 labelencoder를 사용
encoding = LabelEncoder()
df['label_encoding'] = encoding.fit_transform(df['label'])


# # 데이터 전처리(Transform)

# In[8]:


transform = A.Compose([A.Resize(350, 350),
                       #A.HueSaturationValue(),
                       #A.Emboss(),
                       A.CLAHE(clip_limit=6.0, tile_grid_size=(10,10),p=0.5),
                       #A.Sharpen(alpha=(0.2,0.4)),
                       #A.Emboss(),
                       A.HorizontalFlip(p=0.3), 
                       A.Normalize(), ToTensorV2()])


# In[9]:


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
#         image = cv2.imread(image)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.label_filepath is not None:
            label = self.label_filepath[index]
            return image, label
        else:
            return image


# # trainset과 validset 데이터 프레임 생성

# In[10]:


train_set, valid_set , _,_ = train_test_split(df,df['label_encoding'],test_size=0.1, stratify=df['label_encoding'],random_state=seed)
# stratify : class 비율을 일정하게 만들어 준다. Target 값으로 넣어주면 그 값의 비율에 맞게 나누어 주기 때문에 꼭 필요!


# In[11]:


trainset = MyData(train_set['image'].values, train_set['label_encoding'].values,transform=transform)
validset = MyData(valid_set['image'].values, valid_set['label_encoding'].values,transform=transform)


# ## 데이터 시각화

# In[12]:


imageset = MyData(train_set['image'].values, train_set['label_encoding'].values, transform=transform)


# In[13]:


figure, axes = plt.subplots(nrows=4, ncols=8, figsize=(14, 8))
axes = axes.flatten()

for i in range(32):
    rand_i = np.random.randint(0, 32)
    result = imageset[rand_i]
    image = result[0]
    axes[i].axis('off')
    axes[i].imshow(image.permute(1,2,0))


# ## 데이터 적재

# In[14]:


batch_size = 9
trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
validloader = DataLoader(validset, batch_size=batch_size,shuffle=False)


# ## Efficientnet 모델을 생성하는 함수

# In[15]:


pip install efficientnet_pytorch


# In[16]:


from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=19)
model.to(device)


# ## Resnet 모델 활용하는 함수

# In[17]:


# model = models.resnet101(weights=True)
# model.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
# model.to(device)


# ## 하이퍼파리미터 설정

# In[18]:


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)


# # 모델 훈련

# In[19]:


def train_loop(model, trainloader, loss_fn, epochs, optimizer):
    steps = 0
    steps_per_epoch = len(trainloader)
    min_loss = 1000000
    max_f1_score = 0
    trigger = 0
    patience = 7

    for epoch in range(epochs):
        model.train() # training mode
        train_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            predict = model(images)
            loss = loss_fn(predict, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if (steps % steps_per_epoch) == 0:
                model.eval() # evaluation mode
                valid_loss, weighted_f1_score = validate(model, validloader, loss_fn)
                
                print('Epoch : {}/{}.......'.format(epoch+1, epochs),            
                      'Train Loss : {:.3f}'.format(train_loss/len(trainloader)),
                      'Valid Loss : {:.3f}'.format(valid_loss/len(validloader)), 
                      'Weighted F1 Score : {:.3f}'.format(weighted_f1_score)            
                      )

                # Best model saving    
                if weighted_f1_score > max_f1_score:
                    max_f1_score = weighted_f1_score
                    best_model_state = deepcopy(model.state_dict())
                    torch.save(best_model_state, 'best_checkpoint.pth')

                # Early Stopping
                if valid_loss > min_loss: 
                    trigger += 1
                    print('trigger : ', trigger)
                    if trigger > patience:
                        print('Early Stopping !!!')
                        print('Training loop is finished !!') 
                        return
                else:
                    trigger = 0
                    min_loss = valid_loss

            # Learning Rate Scheduler
                scheduler.step(loss)


# In[20]:


from sklearn.metrics import f1_score

def validate(model, validloader, loss_fn):
    total = 0   
    valid_loss = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for images, labels in validloader: 
            images, labels = images.to(device), labels.to(device)

            pred = model(images) # 예측 점수 
            
            loss = loss_fn(pred, labels)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

            
            valid_loss += loss.item() 

        weighted_f1_score = f1_score(true_labels, preds, average='weighted')
      
    return valid_loss, weighted_f1_score


# In[21]:


epochs = 60
get_ipython().run_line_magic('time', 'train_loop(model, trainloader, loss_fn, epochs, optimizer)')


# # 테스트 셋 파일 가져오기

# In[22]:


# for google colab
# test_set = glob.glob('/content/remodel/test/*.png')
# test_set.sort()
# testset = MyData(test_set, None,transform=transform)

# for kaggle
test_set = glob.glob('/kaggle/input/remodel/test/*.png')
test_set.sort()
testset = MyData(test_set, None,transform=transform)


# In[23]:


testloader = DataLoader(testset,batch_size=batch_size, shuffle=False)


# In[24]:


# for kaggle 
best_checkpoint_dict = torch.load('/kaggle/working/best_checkpoint.pth')

#for google colab
# best_checkpoint_dict = torch.load('//content/best_checkpoint.pth')


# In[25]:


model.load_state_dict(best_checkpoint_dict)


# In[26]:


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


# In[27]:


#for kaggle
submit = pd.read_csv('/kaggle/input/remodel/sample_submission.csv')

#for google colab
# submit = pd.read_csv('/content/remodel/sample_submission.csv')


# In[28]:


submit['label'] = preds


# In[ ]:


submit.head(10)


# In[30]:


submit['label'].value_counts(), submit['label'].nunique()


# In[31]:


submit.to_csv('./batch_size=8changeclahe_submit.csv', index=False)


# In[ ]:




