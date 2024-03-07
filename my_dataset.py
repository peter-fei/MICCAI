import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
IMAGE_FOLDER = "/public_bme/data/jianght/datas/Pathology"
IMAGE_FOLDER2 = "/public_bme/data/jianght/datas/Pathology_256"
IMAGE_FOLDER3 = "/public_bme/data/jianght/datas/Pathology_256_pt"

import os  
from PIL import Image  
import torch  
import torchvision.transforms as transforms  
from torch.utils.data import Dataset  
from glob import glob
from einops.layers.torch import Rearrange
  
        



class MultiDataSetMoE(Dataset):
    def __init__(self, data, head_idx=None, age=False, img_batch=25,tasks=['fungus','label'],need_patch=False,patch_size=256,task_id=None):
        if isinstance(data,str) and  os.path.isfile(data):
            data = pd.read_csv(data)
        # print(data.columns,'pppppppppppppppppp')
        if task_id:
            self.data = data[data['task_id'].isin(task_id)]
        else:
            self.data = data 
        # print('ssdsad',len(data),len(self.data))
        self.head_idx = head_idx
        self.age = age
        self.img_batch = img_batch
        self.tasks = tasks
        self.patch_size = patch_size
        self.need_patch = need_patch
        
        # columns = self.data.columns[2:].to_list()
        # if 'task_id' in columns:
        #     columns.remove('task_id')
        # label	highlabel	degree	fungus	cluecell	microbe

        # columns = ['label','highlabel','degree','fungus','cluecell','microbe']    
        columns = ['label','bumanyi','degree','fungus','cluecell','microbe']    
        self.columns = columns
        print(self.columns)


        if isinstance(self.tasks,str):
            self.tasks = [tasks]
        # print(self.data.columns.array)
        for i in self.tasks:
            assert i in self.columns, f'task names wrong : get {i} -------- '

    def __getitem__(self, index):
        # Initialize transform and normalize

        #label 0	highlabel/manyidu 1	degree 2	fungus 3	cluecell 4	microbe 5

        # Read images
        images_tensor = self.get_imgtensor(index)
        # images_tensor = 1
        label_dict={}
        

        for i in self.tasks:
            column_index = self.data.columns.get_loc(i)
            idx = self.columns.index(i)
            label_dict[f'label_{idx}'] = self.data.iloc[index,column_index]
            # print(idx,i,self.data.iloc[index,column_index],'wdwdwad')
            
        if 'code' in self.data.columns:
            idx = self.data.columns.get_loc('code')
            label_dict['code'] = self.data.iloc[index,idx]
        else:
            label_dict['code'] = self.data.iloc[index, 0].split('/')[-1]
        if 'highlabel' in self.data.columns:
            multilabel_index = self.data.columns.get_loc('highlabel')
        else:
            multilabel_index = self.data.columns.get_loc('multilabel')
        label_dict['highlabel'] = self.data.iloc[index, multilabel_index]
        if 'task_id' in self.data.columns:
            idx = self.data.columns.get_loc('task_id')
            label_dict['task_id'] = self.data.iloc[index,idx]  
            label_dict[f'label_single'] = self.data.iloc[index, label_dict['task_id']+2]        
        else:
            label_dict['task_id'] = 'None'
        label_dict['tasks'] = self.columns
        labels = label_dict
        # print(folder_path,images_tensor.size())
        # print(labels)
        return images_tensor, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    def get_imgtensor(self,index):
        
        folder_path = get_folder_path(self.data.iloc[index, 0],IMAGE_FOLDER3)
        if folder_path and os.path.isfile(os.path.join(folder_path,'images.pt')):
            images_tensor = torch.load(os.path.join(folder_path,'images.pt'))
            images_tensor = images_tensor[:self.img_batch]
            return images_tensor

        folder_path = get_folder_path(self.data.iloc[index, 0],IMAGE_FOLDER2)
        if folder_path is None:
            folder_path = get_folder_path(self.data.iloc[index, 0])
        assert os.path.isdir(folder_path),f'get wrong name {self.data.iloc[index, 0]}'
        
        image_filenames = sorted(glob(f'{folder_path}/*.jpg'), key=lambda x: os.path.getsize(x), reverse=True)[:self.img_batch]
        # if len(image_filenames) != 50:
        
        # print('sss',folder_path)
        images = []
        for img_name in image_filenames:
            image_path = img_name
            image = Image.open(image_path)
            
            if self.need_patch:
                transform = transforms.Compose([
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Rearrange('c (h p1) (w p2) -> (h w) c p1 p2 ', p1=self.patch_size, p2=self.patch_size),
                ])
                image = transform(image)
                images.extend(image)
            else:
                transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #Rearrange('c (h p1) (w p2) -> (h w) c p1 p2 ', p1=self.patch_size, p2=self.patch_size),
                ])
                image = transform(image)
                images.append(image)

            # Stack images
        images_tensor = torch.stack(images)
        return images_tensor
    

def get_folder_path(name,floder=IMAGE_FOLDER):

    folder_path = None
    if os.path.isdir(os.path.join(floder, name, 'torch')):
        folder_path = os.path.join(floder, name, 'torch')
    elif os.path.isdir(os.path.join(floder,'yangxing','Torch',name,'torch')):
        folder_path = os.path.join(floder,'yangxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder,'yinxing','Torch',name,'torch')):
        folder_path = os.path.join(floder,'yinxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(floder, name)):
        folder_path = os.path.join(floder, name)
    
    if folder_path is None or (len(os.listdir(folder_path)) < 50 and len(os.listdir(folder_path)) != 1) :
        # print(f'{name} not in {floder}')
        folder_path = None
    # assert os.path.isdir(folder_path),f'get wrong name {name}'
    
    return folder_path
if __name__=='__main__':
    # data1=Gongjing('D:\\Datas\\bingli')
    # data2=Fungus('E:\\fungus')
    data3 = MultiDataSetMoE(data='/public_bme/data/jianght/datas/Pathology/class2/select_train.csv',img_batch=50)
    # data=data1+data2
    data = data3
    loader=torch.utils.data.DataLoader(data,shuffle=False,batch_size=16)

    for i,l in loader:
        print(i.size())


