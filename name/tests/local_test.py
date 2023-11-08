from typing import Literal
import os
import torch

from src.data.dataset import DatasetForMultimodal
from src.models.multimodal_model import MultimodalFusionModel
from src.utils.utils import get_device
from src.utils.visualize import array2gif, visualize_joint_prediction

def predict(
    model, dataset, in_idx: int,
    joint: Literal['open', 'closed'], image: Literal['open', 'closed'],
    tactile: Literal['open', 'closed'],
    pred_mode: Literal['joint', 'movie'], model_name: str, dirpath: str
):
    model.eval()
    model.batch_size = 1
    model.init_hidden(1)

    initial_inputs, _ = dataset[in_idx:in_idx + 1, 0:1]
    initial_inputs = [i.to(model.device) for i in initial_inputs]
    predictions = model(initial_inputs)
    joint_predictions = predictions[0].squeeze(0)  # (1, 5)
    image_predictions = predictions[1]  # (1, 3, 32, 32)
    tactile_predictions = predictions[2] # (1, 3, 32, 32)

    for i in range(dataset.length - 1):
        i_th_input, _ = dataset[in_idx:in_idx + 1, i:i + 1]

        if joint == 'open':
            joint_input = i_th_input[0].to(model.device)
        else:
            joint_input = joint_predictions[-1:].unsqueeze(0)

        if image == 'open':
            image_input = i_th_input[1].to(model.device)
        # else:
        #     image_input = image_predictions[-1:].unsqueeze(0)

        if tactile == 'open':
            tactile_input = i_th_input[1].to(model.device)
        # else:
        #     tactile_input = tactile_predictions[-1:].unsqueeze(0) 


        inputs = (joint_input, image_input, tactile_input)
        outputs = model(inputs) # joint : (1, 5)


        joint_predictions = torch.cat(
            (joint_predictions, outputs[0].squeeze(0)), dim=0)
        
        # image_predictions = torch.cat(
        #     (image_predictions, outputs[1]), dim=0)
        # tactile_predictions = torch.cat(
        #     (tactile_predictions, outputs[2]), dim=0)


    if pred_mode == 'joint':
        visualize_joint_prediction(
            joint_predictions,
            dataset,
            f'{dirpath}/joint_pred.png',
            in_idx
        )
    else: 
        pass

def main():

    #自身のディレクトリ名
    username = "name"

    #モデルを選択
    model_name = 'model_00'

    dirpath = f'{username}/outputs/localtest_reults/{model_name}'
    os.makedirs(dirpath, exist_ok=True)

    path_to_model = f'{username}/models/{model_name}.pth'

    #デバイスの指定
    device = get_device(gpu_id=-1) 

    datamode = 'test'
    
    dataset = DatasetForMultimodal(device=device, mode=datamode) 

    model = MultimodalFusionModel(device=device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))

    for i in range(4):
        predict(
            model,
            dataset,
            in_idx=i,
            joint='closed',
            image='open',
            tactile='open',
            pred_mode='joint',
            model_name=model_name,
            dirpath=dirpath,
        )

if __name__ == '__main__':
    main()
