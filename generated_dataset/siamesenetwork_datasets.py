import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
import torch 
import numpy as np
import sys
from torch.utils.data import Dataset

sys.path.append('..')
from transformation_utils import full_transformations

## Train and Validation dataset for ContrastiveLoss
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.tracker = {}
        self.preenumerate = self.preenumerate_function()

    def __getitem__(self,index):
        return self.preenumerate[index]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def preenumerate_function(self):
        enum_dict = {}
        for i in tqdm(range(len(self.imageFolderDataset.imgs))):
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            img0 = Image.open(img0_tuple[0]).convert('L')

            choices = [0, 1, 2] # 0: transform, 1: same class, 2: different class
            weights = [0.2, 0.4, 0.4]

            data_prep_choice = random.choices(choices, weights, k=1)[0]
            if data_prep_choice == 0:
                img0 = Image.open(img0_tuple[0]).convert('L')
                img1 = full_transformations(img0).convert('L')
                img0 = self.transform(img0)
                img1 = self.transform(img1)
                enum_dict[i] = (img0, img1, torch.from_numpy(np.array([1], dtype=np.float32)))
            elif data_prep_choice == 1:
                while True: # Find a different image
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if ((img0_tuple[0] != img1_tuple[0] and img1_tuple[0] not in self.tracker) or \
                    (img0_tuple[0] != img1_tuple[0] and img1_tuple[0] in self.tracker and self.tracker[img1_tuple[0]] != img0_tuple[0])) and \
                    (img0_tuple[1] == img1_tuple[1]): # make same class
                        img0 = Image.open(img0_tuple[0]).convert('L')
                        img1 = Image.open(img1_tuple[0]).convert('L')
                        img0 = self.transform(img0)
                        img1 = self.transform(img1)
                        self.tracker[img0_tuple[0]] = img1_tuple[0]
                        break
                enum_dict[i] = (img0, img1, torch.from_numpy(np.array([0], dtype=np.float32)))
            else:
                while True: # Find a different image
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if ((img0_tuple[0] != img1_tuple[0] and img1_tuple[0] not in self.tracker) or \
                    (img0_tuple[0] != img1_tuple[0] and img1_tuple[0] in self.tracker and self.tracker[img1_tuple[0]] != img0_tuple[0])) and \
                    (img0_tuple[1] != img1_tuple[1]): # make different class
                        img0 = Image.open(img0_tuple[0]).convert('L')
                        img1 = Image.open(img1_tuple[0]).convert('L')
                        img0 = self.transform(img0)
                        img1 = self.transform(img1)
                        self.tracker[img0_tuple[0]] = img1_tuple[0]
                        break

                enum_dict[i] = (img0, img1, torch.from_numpy(np.array([0], dtype=np.float32)))
        return enum_dict

## Train and Validation Dataset for TripletLoss
class SiameseTripletNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.tracker = {}
        self.preenumerate = self.preenumerate_function()

    def __getitem__(self,index):
        return self.preenumerate[index]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def preenumerate_function(self):
        enum_dict = {}
        for i in tqdm(range(len(self.imageFolderDataset.imgs))):
            #anchor
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            img0 = Image.open(img0_tuple[0]).convert('L')

            #positive
            img1 = full_transformations(img0).convert('L')
            img1 = self.transform(img1)

            #negative
            while True: # ensure that negative image is not the same as anchor
                img2_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[0] != img2_tuple[0]:
                    img2 = Image.open(img2_tuple[0]).convert('L')
                    img2 = self.transform(img2)
                    break

            #Transform img0
            img0 = self.transform(img0)

            enum_dict[i] = (img0, img1,img2)

        return enum_dict

## Test Dataset
class SiameseNetworkTestDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.tracker = {}
        self.preenumerate, self.preenumerate1, self.preenumerate2 = self.preenumerate_function()

    def __getitem__(self,index):
        return self.preenumerate[index],self.preenumerate1[index],self.preenumerate2[index] 

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def preenumerate_function(self):
        enum_dict = {}
        enum_dict1 = {}
        enum_dict2 = {}
        for i in tqdm(range(len(self.imageFolderDataset.imgs))):
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            img0 = Image.open(img0_tuple[0]).convert('L')
            
            
            img0 = Image.open(img0_tuple[0]).convert('L')
            img1 = full_transformations(img0).convert('L')
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            enum_dict[i] = (img0, img1, torch.from_numpy(np.array([1], dtype=np.float32)))

        
            while True: # Find a different image
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if ((img0_tuple[0] != img1_tuple[0] and img1_tuple[0] not in self.tracker) or \
                (img0_tuple[0] != img1_tuple[0] and img1_tuple[0] in self.tracker and self.tracker[img1_tuple[0]] != img0_tuple[0])) and \
                (img0_tuple[1] == img1_tuple[1]): # make same class
                    img0 = Image.open(img0_tuple[0]).convert('L')
                    img1 = Image.open(img1_tuple[0]).convert('L')
                    img0 = self.transform(img0)
                    img1 = self.transform(img1)
                    self.tracker[img0_tuple[0]] = img1_tuple[0]
                    break
            enum_dict1[i] = (img0, img1, torch.from_numpy(np.array([0], dtype=np.float32)))
        
            while True: # Find a different image
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if ((img0_tuple[0] != img1_tuple[0] and img1_tuple[0] not in self.tracker) or \
                (img0_tuple[0] != img1_tuple[0] and img1_tuple[0] in self.tracker and self.tracker[img1_tuple[0]] != img0_tuple[0])) and \
                (img0_tuple[1] != img1_tuple[1]): # make different class
                    img0 = Image.open(img0_tuple[0]).convert('L')
                    img1 = Image.open(img1_tuple[0]).convert('L')
                    img0 = self.transform(img0)
                    img1 = self.transform(img1)
                    self.tracker[img0_tuple[0]] = img1_tuple[0]
                    break

            enum_dict2[i] = (img0, img1, torch.from_numpy(np.array([0], dtype=np.float32)))
        return enum_dict, enum_dict1, enum_dict2
    
## Used to pick out classes for visualization
class SiameseNetworkDatasetVisualization(Dataset): # 0: transform (label 1), 1: same class (label 0), 2: different class (label 0)
    def __init__(self,imageFolderDataset,choice,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.choice = choice
        self.tracker = {}

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0 = Image.open(img0_tuple[0]).convert('L')

        choices = [0, 1, 2] # 0: transform, 1: same class, 2: different class
        weights = [0.2, 0.4, 0.4]

        # data_prep_choice = random.choices(choices, weights, k=1)[0]
        if self.choice == 0:
            img0 = Image.open(img0_tuple[0]).convert('L')
            img1 = full_transformations(img0).convert('L')
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            return img0, img1, torch.from_numpy(np.array([1], dtype=np.float32))
        elif self.choice == 1:
            while True: # Find a different image
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if ((img0_tuple[0] != img1_tuple[0] and img1_tuple[0] not in self.tracker) or \
                 (img0_tuple[0] != img1_tuple[0] and img1_tuple[0] in self.tracker and self.tracker[img1_tuple[0]] != img0_tuple[0])) and \
                 (img0_tuple[1] == img1_tuple[1]): # make same class
                    img0 = Image.open(img0_tuple[0]).convert('L')
                    img1 = Image.open(img1_tuple[0]).convert('L')
                    img0 = self.transform(img0)
                    img1 = self.transform(img1)
                    self.tracker[img0_tuple[0]] = img1_tuple[0]
                    break
            return img0, img1, torch.from_numpy(np.array([0], dtype=np.float32))
        else:
            while True: # Find a different image
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if ((img0_tuple[0] != img1_tuple[0] and img1_tuple[0] not in self.tracker) or \
                 (img0_tuple[0] != img1_tuple[0] and img1_tuple[0] in self.tracker and self.tracker[img1_tuple[0]] != img0_tuple[0])) and \
                 (img0_tuple[1] != img1_tuple[1]): # make different class
                    img0 = Image.open(img0_tuple[0]).convert('L')
                    img1 = Image.open(img1_tuple[0]).convert('L')
                    img0 = self.transform(img0)
                    img1 = self.transform(img1)
                    self.tracker[img0_tuple[0]] = img1_tuple[0]
                    break

            return img0, img1, torch.from_numpy(np.array([0], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
