import random
import pandas as pd
import numpy as np
import os, re, glob, cv2
from PIL import Image 
import matplotlib.pyplot as plt
# plt.rc('font', family='NanumBarunGothic')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Subset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from copy import deepcopy

import warnings
warnings.filterwarnings(action='ignore') 

class myOverfitting():
    def __init__(self, data_path, cutline=30):
        self.data_path = data_path
        self.cutline = cutline
        self.choose_small_data()
    
    def choose_small_data(self):
        df = pd.DataFrame(data={'img_path':glob.glob(self.data_path+'/train/*/*')})
        df['label'] = df['img_path'].apply(lambda x: str(x).split('/')[-2])
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        self.overs_dic = {}
        for i, name in zip(range(len(le.classes_)), le.classes_):
            if df['label'].value_counts()[i]<=self.cutline:
                print('{0:<3} {1:<10}: {2}'.format(i, name, df['label'].value_counts()[i]))
                self.overs_dic[name] = glob.glob(self.data_path+'/train/'+name+'/*')
        return self
                
    def increase_data(self):
        for k,v in self.overs_dic.items():
            increase_num = self.cutline - len(v)
            self.overs_dic[k] = v*(increase_num//len(v)) + v[:increase_num%len(v)]
            for i, img_path in enumerate(self.overs_dic[k]):
                img = np.array(Image.open(img_path).convert('RGB'))
                flip, rnd_n = np.random.randint(0,2), np.random.randint(0,7)
                if flip: img = cv2.flip(img, 1)

                h,w = img.shape[:2]
                dst_pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
                if rnd_n==0: img_pts = np.array([[15,15],[w-15,15],[w-15,h-15],[15,h-15]], dtype=np.float32)
                elif rnd_n==1: img_pts = np.array([[30,0],[w-1,0],[w-1,h-1],[30,h-1]], dtype=np.float32)
                elif rnd_n==2: img_pts = np.array([[0,25],[w-1,25],[w-1,h-1],[0,h-1]], dtype=np.float32)
                elif rnd_n==3: img_pts = np.array([[0,0],[w-30,0],[w-30,h-1],[0,h-1]], dtype=np.float32)
                elif rnd_n==4: img_pts = np.array([[0,0],[w-1,0],[w-1,h-25],[0,h-25]], dtype=np.float32)
                elif rnd_n==5: img_pts = np.array([[30,0],[w-20,40],[w-10,h-10],[5,h-20]], dtype=np.float32)
                elif rnd_n==6: img_pts = np.array([[40,10],[w-15,20],[w-30,h-20],[15,h-10]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(img_pts, dst_pts)
                dst = cv2.warpPerspective(img, M, dsize=(0,0))
                self.saveImg(self.data_path+'/more_data/'+k, '{}.png'.format(i), dst)    
        
    def saveImg(self, directory, img_name, img):
        if not os.path.exists(directory):
            os.makedirs(directory)
        Image.fromarray(img).save(directory+'/'+img_name)

class mySetting():
    def __init__(self, seed):
        self.seed = seed
                
    def seedSetting(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        return

    def getDataframe(self, data_path, valid_s=0.2):
        self.data_path = data_path
        df = pd.DataFrame(data={'img_path':glob.glob(data_path+'/train/*/*')})
        if os.path.exists(data_path+'/more_data'):
            df = pd.concat([df, pd.DataFrame(data={'img_path':glob.glob(data_path+'/more_data/*/*')})], axis=0)
        df['label'] = df['img_path'].apply(lambda x: str(x).split('/')[-2])
        self.label_li = df['label'].unique().tolist()
        self.le = LabelEncoder()
        if valid_s!=0:
            self.train_df, self.valid_df, _, _ = train_test_split(df, df['label'], test_size=valid_s, 
                                                    stratify=df['label'], random_state=self.seed)
            self.train_df['label'] = self.le.fit_transform(self.train_df['label'])
            self.valid_df['label'] = self.le.transform(self.valid_df['label'])
        else:
            self.train_df = df.copy()
            self.train_df['label'] = self.le.fit_transform(self.train_df['label'])
            self.valid_df = []
        self.test_df = pd.DataFrame(data={'img_path':sorted(glob.glob(data_path+'/test/*'))})
        return self
        
    def getDataset(self, transform, test_transform, using='transforms'):
        self.trainset = mycreateDataset(self.train_df['img_path'].values, self.train_df['label'].values, transform, using)
        if len(self.valid_df)!=0:
            self.validset = mycreateDataset(self.valid_df['img_path'].values, self.valid_df['label'].values, transform, using)
        else: self.validset = []
        self.testset = mycreateDataset(self.test_df['img_path'].values, None, test_transform, using)
        print('train, valid, test:', len(self.trainset), len(self.validset), len(self.testset))
        return self
    
    def getDataloader(self, batch_s=16):
        self.trainloader = DataLoader(self.trainset, batch_size=batch_s, shuffle=False, num_workers=0)
        if len(self.validset)!=0:
            self.validloader = DataLoader(self.validset, batch_size=batch_s, shuffle=False, num_workers=0)
        else: self.validloader = []
        self.testloader = DataLoader(self.testset, batch_size=batch_s, shuffle=False, num_workers=0)
        print('train, valid, test:', len(self.trainloader), len(self.validloader), len(self.testloader))
        train_iter = iter(self.trainloader)
        imgs, labels = train_iter.__next__()
        print('trainloader shape', imgs.shape, labels.shape)
        return self
    
    def showimg(self):
        self.labels_map = {}
        for k,v in zip(self.le.transform(self.label_li), self.label_li):
            self.labels_map[k] = v
        fig, ax = plt.subplots(2,4, figsize=(10,6))
        ax = ax.flatten()
        for i in range(8):
            item = self.trainset[np.random.randint(0, len(self.trainset))]
            img, label = item[0].permute(1,2,0), item[1]
            img.mul_(torch.tensor([0.229, 0.224, 0.225])) # std
            img.add_(torch.tensor([0.485, 0.456, 0.406])) # mean
            ax[i].axis('off'); ax[i].imshow(img)
            ax[i].set_title(str(label))  # (self.labels_map[label])
        plt.show()
        return self

class mycreateDataset(Dataset):
    def __init__(self, filepaths, labels, transform, using='transforms'):
        self.filepaths, self.labels = filepaths, labels
        self.transform = transform
        self.using = using

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.labels is not None: img_filepath = self.filepaths[idx]
        else: img_filepath = self.filepaths[idx]
        img = Image.open(img_filepath).convert('RGB')
        if self.using=='A': 
            img = np.array(img)
            transed_img = self.transform(image=img)['image']
        else: 
            transed_img = self.transform(img)
        if self.labels is not None:
            return transed_img, self.labels[idx]
        else: return transed_img        
        
class myTrain():
    def __init__(self, model, loss_fn, optimizer, trainloader, validloader, testloader, 
                 scheduler, device, le, epochs=30, patience=5, batch_s=16):
        self.model, self.loss_fn, self.optimizer = model, loss_fn, optimizer
        self.trainloader, self.validloader, self.testloader = trainloader, validloader, testloader
        self.scheduler, self.device = scheduler, device
        self.epochs, self.patience, self.batch_s = epochs, patience, batch_s
        self.le = le
        self.train_loss_li, self.valid_loss_li, self.valid_f1_li = [], [], []
        
    def train_loop(self):
        # self.loss_fn = self.loss_fn.to(device)

        best_f1 = 0; trigger = 0
        for epoch in range(1,self.epochs+1):
            self.model.train() 
            train_loss = 0
            for imgs, labels in tqdm(self.trainloader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                loss = self.loss_fn(self.model(imgs), labels) 
                self.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step() 
                train_loss += loss.item()

            if len(self.validloader)!=0:
                valid_loss, valid_f1 = self.validate()
                print('Epoch : {}/{}.......'.format(epoch+1, self.epochs),            
                      'Train Loss : {:.3f}'.format(train_loss/len(self.trainloader)), 
                      'Valid Loss : {:.3f}'.format(valid_loss),
                      'Valid F1 Score : {:.3f}'.format(valid_f1))
                self.train_loss_li.append(train_loss/len(self.trainloader))
                self.valid_loss_li.append(valid_loss)
                self.valid_f1_li.append(valid_f1)

                if valid_f1<best_f1:
                    trigger += 1
                    if trigger > self.patience:
                        print('\nEarly Stopping!! epoch/epochs: {}/{}'.format(epoch, self.epochs))
                        break
                else:
                    trigger = 0
                    best_f1 = valid_f1
                    best_model_state = deepcopy(self.model.state_dict())
                    torch.save(best_model_state, 'best_checkpoint.pth') 
                self.scheduler.step(valid_f1)
            
            else:
                print('Epoch : {}/{}.......'.format(epoch+1, self.epochs),            
                      'Train Loss : {:.3f}'.format(train_loss/len(self.trainloader)))
                self.train_loss_li.append(train_loss/len(self.trainloader))
                self.scheduler.step(train_loss/len(self.trainloader))
        return

    def validate(self):
        self.model.eval()
        loss, preds, true_labels = 0, [], []
        with torch.no_grad():
            for imgs, labels in self.validloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                pred = self.model(imgs)
                # _, preds = torch.max(pred, 1)
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += labels.detach().cpu().numpy().tolist()
                loss += self.loss_fn(pred, labels).item()
        return loss/len(self.validloader), f1_score(true_labels, preds, average='weighted')

    def result_plot(self):
        x = np.arange(len(self.train_loss_li))
        if len(self.validloader)!=0:
            fig, ax = plt.subplots(1,2, figsize=(10,3))
            ax[0].plot(x, self.valid_f1_li, label='valid_F1_score')
            ax[0].set_title('valid F1 score')
            ax[1].plot(x, self.train_loss_li, label='train loss')
            ax[1].plot(x, self.valid_loss_li, label='valid loss')
            ax[1].set_title('loss')
        else:
            plt.plot(x, self.train_loss_li, label='train loss')
        plt.xlabel('epochs'); plt.legend(loc='best')
        plt.show()
        
    def evaluate(self, model, testloader, loss_fn):
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs in tqdm(testloader):
                imgs = imgs.to(self.device)
                pred = model(imgs)
                # _, preds = torch.max(logit, 1)
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
        return self.le.inverse_transform(preds)

    def load_model(self, dirct, model, testloader, loss_fn):
        state_dict = torch.load(dirct)
        load_model = model
        load_model.load_state_dict(state_dict)
        return self.evaluate(load_model, testloader, loss_fn)
    
    def save_preds(self, submit, output_fp, date_n, preds_li):
        self.test_submit, self.last_submit, self.best_submit = submit.copy(), submit.copy(), submit.copy()
        self.test_submit['label'] = preds_li[0]
        self.test_submit.to_csv(output_fp+date_n+'_test_submit.csv', index=False)
        self.last_submit['label'] = preds_li[1]
        self.last_submit.to_csv(output_fp+date_n+'_last_submit.csv', index=False)
        if len(preds_li)==3:
            self.best_submit['label'] = preds_li[2]
            self.best_submit.to_csv(output_fp+date_n+'_best_submit.csv', index=False)
        return self
        
        