""" データの加工を行うメソッド """

from itertools import product
from typing import Union

import numpy as np
import torch
import cv2

JOINT_MAX_ARRAY = [0.799, 0.759, 0.681, 1.500, -0.00992] #それぞれの関節角度データ（学習用）の最大
JOINT_MIN_ARRAY = [-0.828, -0.169, -0.701, 0.954, -0.00996] #それぞれの関節角度データ（学習用）の最小

def norm_joint_data(data: np.ndarray) -> None:
    """ 関節角度をとりうる最大・最小値で正規化する

    Args:
        data(np.ndarray): (data_num, length, dim)のデータ

    Returns:
        None: 参照されているdataに影響を与える

    """

    assert data.ndim == 3

    num, leng, dim = data.shape

    for n, l, d in product(range(num), range(leng), range(dim)):
        data[n, l, d] -= JOINT_MIN_ARRAY[d]
        data[n, l, d] /= JOINT_MAX_ARRAY[d] - JOINT_MIN_ARRAY[d]
        data[n, l, d] *= 2
        data[n, l, d] -= 1


def denorm_joint_data(data: torch.Tensor) -> torch.Tensor:
    """ 正規化された値を元に戻す

    Args:
        data(Union[np.ndarray, torch.Tensor]):
            (data_num, length, dim)のデータ

    Returns:
        torch.Tensor:
            (data_num, length, dim)のデータ

    """

    if type(data) == torch.Tensor:
        data = data.cpu().clone().detach().numpy()

    num, leng, dim = data.shape
    for n, l, d in product(range(num), range(leng), range(dim)):
        data[n, l, d] += 1
        data[n, l, d] /= 2
        data[n, l, d] *= JOINT_MAX_ARRAY[d] - JOINT_MIN_ARRAY[d]
        data[n, l, d] += JOINT_MIN_ARRAY[d]

    return torch.from_numpy(data).float()


def np_image_totensor(data: np.ndarray, permute: bool = True) -> torch.Tensor:
    """ numpy array image/movieをtensorに変換する

    4,5次元のnumpy arrayを0~255から0.~1.のtensorにする．
    また， permute==Trueのとき，torch.nn.conv2dに合わせて軸の順序を変更する．

    Args:
        data(np.ndarray): 4・5次元のnumpy array
        permute(bool): 軸の順序を変更するかどうか

    Returns:
        torch.Tensor: 4・5次元のtensor

    """

    data = data.copy()

    # データが0~1にされていない場合
    if data.max() > 1.0:
        data = data.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(data).float()

    if not permute:
        return img_tensor
    # データをpytorchのconv2dの引数の型に合わせる
    if img_tensor.ndim == 4:
        #print(4)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
    elif img_tensor.ndim == 5:
        #print(5)
        img_tensor = img_tensor.permute(0, 1, 4, 2, 3)
    else:
        raise Exception(
            f'data.build_features.np_image_totensor allows 4 or 5 dim \
              np.ndarray. given dim:{img_tensor.ndim}'
        )

    return img_tensor

def np_tactile_totensor(data: np.ndarray, permute: bool = True) -> torch.Tensor:
    """ numpy array tactileをtensorに変換する

    4,5次元のnumpy arrayを0~255から0.~1.のtensorにする．
    また， permute==Trueのとき，torch.nn.conv2dに合わせて軸の順序を変更する．

    Args:
        data(np.ndarray): 4・5次元のnumpy array
        permute(bool): 軸の順序を変更するかどうか

    Returns:
        torch.Tensor: 4・5次元のtensor

    """

    data = data.copy()
    data = cv2.resize(data, dsize=(80, 80))
    data=data[:, :, [2, 1, 0]]#BGR->RGB

    # データが0~1にされていない場合
    if data.max() > 1.0:
        data = data.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(data).float()

    if not permute:
        return img_tensor
    # データをpytorchのconv2dの引数の型に合わせる

    h = img_tensor.size()[0]
    w = img_tensor.size()[1]
    c = img_tensor.size()[2]
    img_tensor = img_tensor.permute(2, 0, 1)

    return img_tensor




def change_brightness(
    img: torch.Tensor, 
    alpha: float = 0, 
    beta: float = np.random.normal(1, 0.2), 
) -> torch.Tensor:
    """

    画像(tensor)明るさを変更する関数
    M1のコードそのまま

    Args:
        img(torch.Tensor): 元画像
        alpha(float): コントラスト
        beta(float): 明るさ
    Returns:
        torch.Tensor

    """
    bright_img = alpha + beta * img.cpu().detach().numpy()
    clip_img = np.clip(bright_img, 0, 1)
    return torch.tensor(clip_img).float()

def add_noise(
    joint_states: torch.Tensor, 
    noise_mean: float=0, 
    noise_std: float=0.2
) -> torch.Tensor:
    noise = torch.normal(noise_mean, noise_std, size=joint_states.shape).to(joint_states.device)
    noised_joint_states = joint_states + noise
    return torch.clamp(noised_joint_states, min=-1.0, max=1.0)