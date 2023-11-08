import os
import torch
from multimodal_model import MultimodalFusionModel
from visual_model import VisionVAE
from models.tactile_model import TactileVAE
from fusion_unit import ConcatUnit
from utils.fit import fit_multimodal, fit_enc_recon
from utils.loss import LossLogger
from utils.utils import EarlyStopping, save_hyperparameters

def train_model(
        username, 
        model_name, 
        model_type, 
        vision_model_name,
        tactile_model_name,
        device,
        train_loader,
        val_loader,
        optimizer,
        lr,
        loss,
        patience,
        ):
    
    #モデルの情報や損失等の保存先を作成
    os.makedirs(f'{username}/models/{model_name}', exist_ok=True)
    os.makedirs(f'{username}/outputs/learning_results/losses/{model_name}', exist_ok=True)

    #モデルの構築
    if model_type == "vis_enc":
        model = VisionVAE(device=device)

    elif model_type ==  "tac_enc":
        model = TactileVAE(device=device)

    elif model_type ==  "multimodal":
        vis_vae = VisionVAE(device=device)
        if vision_model_name != "None":
            path_to_vis_enc = f'{username}/models/{vision_model_name}.pth'
            vis_vae.load_state_dict(torch.load(path_to_vis_enc, map_location=device))
            #vis_vaeの重みを更新しない
            for param in vis_vae.parameters():
                param.requires_grad = False
        vis_enc = vis_vae.encoder

        tac_vae = TactileVAE(device=device)
        if tactile_model_name != "None":
            path_to_tac_enc = f'{username}/models/{tactile_model_name}.pth'
            tac_vae.load_state_dict(torch.load(path_to_tac_enc, map_location=device))
            #tac_vaeの重みを更新しない
            for param in tac_vae.parameters():
                param.requires_grad = False
        tac_enc = tac_vae.encoder

        fusion_unit = ConcatUnit(device=device)

        model = MultimodalFusionModel(device=device, vision_encoder=vis_enc, tactile_encoder=tac_enc, fusion_unit=fusion_unit)

    # model = model.to(device)

    optimizer(params=model.parameters(), lr=lr, amsgrad=True)

    scheduler = None

    early_stopping = EarlyStopping(username=username, model_name=model_name, patience=patience)
    loss_logger = LossLogger(model_name=model_name)

    print('model_ok')


    #学習を回す
    #パラメータ情報の保存
    save_hyperparameters(
        username=username,
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        loss_fn=loss,
        optimizer=optimizer,
        val_loader=val_loader,
        early_stopping=early_stopping,
        scheduler=scheduler
    )

    if model_type == "multimodal":
        #視覚・触覚・関節角度から次の時刻の関節角度を予測する学習
        fit_multimodal(
            username=username,
            model=model,
            train_loader=train_loader,
            loss_fn=loss,
            optimizer=optimizer,
            val_loader=val_loader,
            early_stopping=early_stopping,
            loss_logger=loss_logger,
            scheduler=scheduler
        )
        print('train_ok')

    elif "enc" in model_type:
        #画像の再構成を通したエンコーダの事前学習
        fit_enc_recon(
            username=username,
            model=model,
            train_loader=train_loader,
            loss_fn=loss,
            optimizer=optimizer,
            val_loader=val_loader,
            early_stopping=early_stopping,
            loss_logger=loss_logger,
            scheduler=scheduler
        )
        print('train_ok')

    else:
        print("training does not work.")