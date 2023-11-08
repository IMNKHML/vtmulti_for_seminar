import time
import datetime
from typing import Literal
import os

import numpy as np
import rospy
import torch

from src.data.build_features import (denorm_joint_data, norm_joint_data,
                                     np_image_totensor, np_tactile_totensor)
from src.data.dataset import DatasetForMultimodal
from src.models.multimodal_model import MultimodalFusionModel
from src.utils.camera_utils import get_image, get_pipeline
from src.utils.ros_utils import Replay, move_to_home_position, torque_on
from src.utils.utils import get_device
from src.utils.visualize import visualize_joint_prediction
from digit_interface import Digit
from src.utils.digit_utils import get_tactile_image
# from watch_image import show_image

@torch.no_grad()
def main():
    #自身のディレクトリ名
    username = "name"

    #モデルを選択
    model_name = 'model_00'

    dirpath = f'{username}/models/{model_name}'
    os.makedirs(dirpath, exist_ok=True)

    # 初期設定
    generation_length = 80
    device = get_device(gpu_id=-1)
    model = MultimodalFusionModel(device=device)
    model.load_state_dict(torch.load(f'{username}/models/{model_name}.pth'))
    model.eval()
    model.batch_size = 1
    model.init_hidden(batch_size=1)

    datamode = 'test'
    dataset = DatasetForMultimodal(device=device, mode=datamode) 

    #パイプラインを生成する。
    pipe = get_pipeline()
    process = torque_on(port=0)
    time.sleep(2)

    replay = Replay(rate=9.0, path_time=0.05)
    replay.home()
    time.sleep(2)

    d_r = Digit("D20542") # right tactile data
    d_r.connect()

    #ホームポジションの関節角度データ
    joint_predictions = torch.from_numpy(
        np.array([[[-0.0191, 0.00782, 0.481, 1.074, -0.9]]])
    ).float()

    norm_joint_data(joint_predictions.cpu().detach().numpy())
    
    image_data = torch.zeros(generation_length, 3, 32, 32)
    tactile_data = torch.zeros(generation_length, 3, 320, 240)

    #毎時刻、closedで関節角度の予測を求め、ロボットに指令を与える。
    for i in range(generation_length-1):
        #joint_states = dataset[DATA_IDX: DATA_IDX+1][0][0][:, i:i+1].to(device)  # open joint
        #image_states = dataset[DATA_IDX: DATA_IDX+1][0][1][:, i:i+1].to(device)  # open image
        
        image_states = np_image_totensor(get_image(pipeline=pipe)).unsqueeze(0)
        image_data[i, :, :, :] = image_states.squeeze()

        right_tactile_states = np_tactile_totensor(get_tactile_image(d_r))
        tactile_data[i, :, :, :] = right_tactile_states.squeeze()
        
        inputs = [joint_predictions[-1:], image_states, right_tactile_states]
        
        inputs = [i.to(device) for i in inputs]

        joint_prediction, *_ = model(inputs)
        
        joint_predictions = torch.cat([joint_predictions.to(device), joint_prediction], dim=0)

        joint_prediction = denorm_joint_data(joint_prediction).to(device)
        joint_call = joint_prediction.squeeze().cpu().detach().numpy()

        replay.call_service(joint_call)


    # #カメラより取得した視覚情報の可視化    
    # image_data = image_data.permute(0, 2, 3, 1).cpu().detach().numpy()
    # show_image(
    #     image_data=image_data, 
    #     save_path=f'{username}/outputs/robottest_results/{model_name}/real_vis_image.png'
    # )

    # #触覚センサより取得した触覚情報の可視化    
    # image_data = image_data.permute(0, 2, 3, 1).cpu().detach().numpy()
    # show_image(
    #     image_data=tactile_data, 
    #     save_path=f'{username}/outputs/robottest_results/{model_name}/real_tac_image.png'
    # )
    
    # #予測した関節角度の可視化
    
    
    # ホームポジションに戻り，トルクをOFFにする


    time.sleep(1)
    replay.home()
    d_r.disconnect()
    
    time.sleep(1)
    process.terminate()

    time.sleep(1)
    process.kill()


if __name__ == '__main__':
    main()
