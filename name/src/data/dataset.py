""" データのロード・データセットの作成 """
import os
# import sys
# sys.path.append('name/src')
import cv2
from natsort import natsorted
from glob import glob
from itertools import product
from typing import Tuple
import random
import numpy as np
from einops import repeat
import torch
from torch.utils.data import Dataset, random_split, Subset
from build_features import change_brightness, add_noise
from utils.utils import (EarlyStopping, get_device, save_hyperparameters,
                             seed_worker, torch_fix_seed)

class DatasetForMultimodal(Dataset):
    """
    視覚、触覚、関節角度の情報を用いて、次の時刻の関節角度を予測
    訓練用:32個, 
    評価用:4個,
    テスト用:4個,
    jnt:(bs, len, dim)
    img:(bs, len, h, w, c)
    tac:(bs, len, h, w, c)
    """
    def __init__(self, mode='train'):

        joints, imgs, tacs = make_dataset()

        self.mode = mode
        self.data_num = 40
        self.train_num = 32 + 4  
        self.test_num = self.data_num - self.train_num #4
        
        self.joint_inputs = joints[:self.train_num, :-1] #0 ~ len-1
        self.test_joint_inputs = joints[self.train_num:, :-1]

        self.train_targets = joints[:self.train_num, 1:] #1 ~ len
        self.test_targets = joints[self.train_num:, 1:]

        #うまく取得できていない最初の方のタイムステップの触覚情報を差し替える
        tactile_head_1 = repeat(tacs[:self.train_num, 6], "b c h w -> b l c h w", l=7)
        tactile_head_2 = repeat(tacs[self.train_num:, 6], "b c h w -> b l c h w", l=7)
        tactile_tail_1 = tacs[:self.train_num, 7:-1]
        tactile_tail_2 = tacs[self.train_num:, 7:-1]
        self.tactile_inputs = torch.cat([tactile_head_1, tactile_tail_1], dim=1)
        self.test_tactile_inputs = torch.cat([tactile_head_2, tactile_tail_2], dim=1)

        self.image_inputs = imgs[:self.train_num, :-1]
        self.test_image_inputs = imgs[self.train_num:, :-1]

    def __len__(self):
        if self.mode == 'train':
            return self.train_num
        else:
            return self.test_num

    def __getitem__(self, idx):
        if self.mode == 'train':

            joint_inputs = self.joint_inputs[idx]
            image_inputs = self.image_inputs[idx]
            tactile_inputs = self.tactile_inputs[idx]

            # #ノイズ付与等
            # joint_inputs = add_noise(self.joint_inputs[idx], noise_std=0.2)
            # image_inputs = change_brightness(self.image_inputs[idx], alpha=np.random.normal(0, 0.2))
            # tactile_inputs = change_brightness(self.tactile_inputs[idx], beta=np.random.normal(1, 0.1))

            targets = self.train_targets[idx]

        else:
            #test_data
            joint_inputs = self.test_joint_inputs[idx]
            image_inputs = self.test_image_inputs[idx]
            tactile_inputs = self.test_tactile_inputs[idx]
            targets = self.test_targets[idx]

        data = [joint_inputs, image_inputs, tactile_inputs, targets]

        return data
    

class DatasetForVisual(Dataset):
    """
    視覚情報の事前学習（再構成）
    訓練用:32個, 
    評価用:4個,
    テスト用:4個,
    img:(bs, len, h, w, c)
    """
    def __init__(self, mode='train'):

        _, imgs, _ = make_dataset()

        self.mode = mode
        self.data_num = 40
        self.train_num = 32 + 4  
        self.test_num = self.data_num - self.train_num #4

        self.image_inputs = imgs[:self.train_num, :-1]
        self.test_image_inputs = imgs[self.train_num:, :-1]

        self.train_targets = imgs[:self.train_num, 1:] #1 ~ len
        self.test_targets = imgs[self.train_num:, 1:]


    def __len__(self):
        if self.mode == 'train':
            return self.train_num
        else:
            return self.test_num

    def __getitem__(self, idx):
        if self.mode == 'train':

            image_inputs = self.image_inputs[idx]

            # #ノイズ付与等
            # image_inputs = change_brightness(self.image_inputs[idx], alpha=np.random.normal(0, 0.2))

            targets = self.train_targets[idx]

        else:
            #test_data
            image_inputs = self.test_image_inputs[idx]
            targets = self.test_targets[idx]

        data = [image_inputs, targets]

        return data
    

class DatasetForTactile(Dataset):
    """
    触覚情報の事前学習（再構成）
    訓練用:32個, 
    評価用:4個,
    テスト用:4個,
    tac:(bs, len, h, w, c)
    """
    def __init__(self, mode='train'):

        _, _, tacs = make_dataset()

        self.mode = mode
        self.data_num = 40
        self.train_num = 32 + 4  
        self.test_num = self.data_num - self.train_num #4

        #うまく取得できていない最初の方のタイムステップの触覚情報を差し替える
        tactile_head_1 = repeat(tacs[:self.train_num, 6], "b c h w -> b l c h w", l=7)
        tactile_head_2 = repeat(tacs[self.train_num:, 6], "b c h w -> b l c h w", l=7)
        tactile_tail_1 = tacs[:self.train_num, 7:]
        tactile_tail_2 = tacs[self.train_num:, 7:]
        self.tactile_data = torch.cat([tactile_head_1, tactile_tail_1], dim=1)
        self.test_tactile_data = torch.cat([tactile_head_2, tactile_tail_2], dim=1)

        self.tactile_inputs = self.tactile_data[:, :-1]
        self.test_tactile_inputs = self.test_tactile_data[:, :-1]

        self.train_targets = self.tactile_data[:, 1:]
        self.test_targets = self.test_tactile_data[:, 1:]


    def __len__(self):
        if self.mode == 'train':
            return self.train_num
        else:
            return self.test_num

    def __getitem__(self, idx):
        if self.mode == 'train':

            tactile_inputs = self.tactile_inputs[idx]

            # #ノイズ付与等
            # tactile_inputs = change_brightness(self.tactile_inputs[idx], beta=np.random.normal(1, 0.1))

            targets = self.train_targets[idx]

        else:
            #test_data
            tactile_inputs = self.test_tactile_inputs[idx]
            targets = self.test_targets[idx]

        data = [tactile_inputs, targets]

        return data
    

def split_train_validation(dataset: Dataset, validation_split: float) \
        -> Tuple[Dataset, Dataset]:
    """ DatasetをTrainとValidationに分割する"""

    num_data = len(dataset)
    val_size = int(num_data * validation_split)
    train_size = num_data - val_size
    indices = np.arange(len(dataset))

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:])

    print(f'Train: {len(train_dataset)}', end=' ')
    print(f'Validation: {len(val_dataset)}')

    return train_dataset, val_dataset


def make_dataset() -> torch.Tensor:

    username = 'name'

    data_num = 40
    data_len = 80

    train_idx = []
    val_idx = []
    test_idx = []
    goal_list = ['green', 'blue']
    obj_list = ['r_sponge', 'r_plastic']
    modality_list = ['joint', 'image', 'right_tactile']

    joint_data = np.empty([data_num, data_len, 7])
    image_data = np.empty([data_num, data_len, 32, 32, 3])
    right_tactile_data = np.empty([data_num, data_len, 80, 80, 3])

    for goalcolor, obj in product(goal_list, obj_list):
        print('goalcolor:', goalcolor, 'obj:', obj)

        dirpath = f'{username}/data/processed_data/{goalcolor}/{obj}'

        if os.path.exists(dirpath) == True:
            for modality in modality_list:

                path_list1 = glob(f'{dirpath}/{modality}/*4_[0-9].npy') 
                path_list1 = natsorted(path_list1)

                # path_list2 = glob(f'{dirpath}/{modality}/*4_[1][0-9].npy')
                # path_list2 = natsorted(path_list2)

                # path_list1.extend(path_list2)

                # print('modality:', modality)

                if obj == 'r_sponge':

                    for i, tr_idx in enumerate(train_idx):
                        if modality == 'joint':
                            joint_data[i] = np.load(path_list1[tr_idx])
                        elif modality == 'image':
                            image_data[i] = np.load(path_list1[tr_idx])
                        elif modality == 'right_tactile':
                            right_tactile_data[i] = np.load(path_list1[tr_idx])

                    for i, v_idx in enumerate(val_idx): 
                        if modality == 'joint':
                            joint_data[i+32] = np.load(path_list1[v_idx])
                        elif modality == 'image':
                            image_data[i+32] = np.load(path_list1[v_idx])
                        elif modality == 'right_tactile':
                            right_tactile_data[i+32] = np.load(path_list1[v_idx])

                    for i, ts_idx in enumerate(test_idx): 
                        if modality == 'joint':
                            joint_data[i+36] = np.load(path_list1[ts_idx])
                        elif modality == 'image':
                            image_data[i+36] = np.load(path_list1[ts_idx])
                        elif modality == 'right_tactile':
                            right_tactile_data[i+36] = np.load(path_list1[ts_idx])

                elif obj == 'r_plastic':

                    for i, tr_idx in enumerate(train_idx):
                        if modality == 'joint':
                            joint_data[i+16] = np.load(path_list1[tr_idx])
                        elif modality == 'image':
                            image_data[i+16] = np.load(path_list1[tr_idx])
                        elif modality == 'right_tactile':
                            right_tactile_data[i+16] = np.load(path_list1[tr_idx])

                    for i, v_idx in enumerate(val_idx): 
                        if modality == 'joint':
                            joint_data[i+34] = np.load(path_list1[v_idx])
                        elif modality == 'image':
                            image_data[i+34] = np.load(path_list1[v_idx])
                        elif modality == 'right_tactile':
                            right_tactile_data[i+34] = np.load(path_list1[v_idx])

                    for i, ts_idx in enumerate(test_idx): 
                        if modality == 'joint':
                            joint_data[i+38] = np.load(path_list1[ts_idx])
                        elif modality == 'image':
                            image_data[i+38] = np.load(path_list1[ts_idx])
                        elif modality == 'right_tactile':
                            right_tactile_data[i+38] = np.load(path_list1[ts_idx])

    joint_data = torch.from_numpy(joint_data).to(torch.float32)
    image_data = torch.from_numpy(image_data).to(torch.float32).permute(0, 1, 4, 2, 3) #conv2dに合ったshapeに
    right_tactile_data = torch.from_numpy(right_tactile_data).to(torch.float32).permute(0, 1, 4, 2, 3)

    return joint_data, image_data, right_tactile_data

                

if __name__ == '__main__':
    device = get_device(gpu_id=0)
    mode = 'train'
    dataset = DatasetForMultimodal(device=device, mode=mode) 
    if mode == 'train':
        train_data, val_data = split_train_validation(
            dataset=dataset, validation_split=0.12)
    if mode == 'test':
        train_data, val_data = split_train_validation(
            dataset=dataset, validation_split=0.00)

