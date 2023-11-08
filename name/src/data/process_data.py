""" データサイズの加工 """
import os
# import sys
from glob import glob
import cv2
import numpy as np
from itertools import product
import natsort

def joint_resize(data):
    #データセット1
    # # JOINT_MAX_ARRAY = [0.453, 1.919, 0.449, 0.498, 0.206, 2.496, 0.901] #それぞれの関節角度データ（学習用）の最大
    # # JOINT_MIN_ARRAY = [-1.202, 1.238, -0.379, -1.194, -0.428, 1.422, -0.901] #それぞれの関節角度データ（学習用）の最小   
    
    # #データセット2
    # JOINT_MAX_ARRAY = [0.372, 1.920, 0.475, 0.077, 0.189, 2.496, 0.901] #それぞれの関節角度データ（学習用）の最大
    # JOINT_MIN_ARRAY = [-0.639, 1.327, -0.356, -1.327, -0.212, 1.532, -0.901] #それぞれの関節角度データ（学習用）の最小   

    JOINT_MAX_ARRAY = []
    JOINT_MIN_ARRAY = []
    length, dim = data.shape 
    
    #必要な関節データを取り出す
    assert data.ndim == 2

    #最大・最小のlistを作る
    for i in dim:
        i_max = data[:, i].max()
        i_min = data[:, i].min()
        JOINT_MAX_ARRAY.append(i_max)
        JOINT_MIN_ARRAY.append(i_min)

    print(f'max:', JOINT_MAX_ARRAY)
    print(f'min:', JOINT_MIN_ARRAY)

    #関節ごとにデータの正規化, -1.0~1.0
    for t, d in product(range(length), range(dim)):
        data[t, d] -= JOINT_MIN_ARRAY[d]
        data[t, d] /= JOINT_MAX_ARRAY[d] - JOINT_MIN_ARRAY[d]
        data[t, d] *= 2
        data[t, d] -= 1

    return data


def image_resize(data):

    l, h, w, c = data.shape

    if (h==32 and w==32):
        return data
    
    else:

        data_re = np.ndarray((l, h, w, c))

        for t in range(l):
            data_t = data[t]
            data_t = cv2.resize(data_t, dsize=(32, 32))
            data_re[t] = data_t

    return data_re

def tactile_resize(data):

    l, h, w, c = data.shape

    re_h = 64
    re_w = 64

    data_re = np.ndarray((l, re_h, re_w, c))

    for t in range(l):
        data_t = data[t]
        data_t = cv2.resize(data_t, dsize=(re_h, re_w))
        data_re[t] = data_t

    return data_re

def normalize_tac(data):
    data=data.astype("float32")/255 #これを省いてはいけない。つまり、raw_dataは正規化する必要がある。
    data=data[:, :, :, [2, 1, 0]] #BGR->RGB
    return data

def data_resize(filepath, savepath, modality):
    rawdata = np.load(filepath)

    if modality == 'image':
        resized_data = image_resize(rawdata)
    elif modality == 'right_tactile':
        resized_data = tactile_resize(rawdata)
        resized_data = normalize_tac(resized_data)
    elif modality == 'joint':
        #必要な関節の情報だけ取り出す＆正規化を施す
        resized_data = joint_resize(rawdata)
    else:
        pass

    basename1 = os.path.basename(filepath)
    basename2 = os.path.basename(basename1.replace('raw_', ''))
    filename = basename2.replace('_state', '')
    print('filename:', filename)

    np.save(os.path.join(savepath, filename), resized_data)
    

def main():

    username = 'name'

    modality_list = ['joint', 'image', 'right_tactile']

    dirpath = f'{username}/data/raw_data/'

    if os.path.exists(dirpath) == True:
        for modality in modality_list:
            arrangement1_list = glob(f'{dirpath}/raw_{modality}*1_[0-9].npy') 
            # arrangement2_list = glob(f'{dirpath}/{modality}*4_[1][0-9].npy')
            arrangement1_list = natsort.natsorted(arrangement1_list)
            # arrangement2_list = natsort.natsorted(arrangement2_list)

            savepath = f'{username}/data/processed_data/{modality}'
            os.makedirs(savepath, exist_ok=True)

            for arr1pth in arrangement1_list:
                data_resize(arr1pth, savepath, modality)

            # for arr2pth in arrangement2_list:
            #     print('modality:', modality)
            #     data_resize(arr2pth, savepath, modality)

    else:
        pass

if __name__ == '__main__':
    main()
