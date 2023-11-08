import torch

# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.data.dataset import DatasetForMultimodal, DatasetForVisual, DatasetForTactile, split_train_validation
from src.models.train import train_model
from src.utils.loss import recon_kl, mseloss, recon_kl, cross_ent, charbo, w_loss
from src.utils.utils import get_device, save_hyperparameters, seed_worker, torch_fix_seed

def main():
    """
    パラメータの設定とモデルの学習

    """

    # 自身のディレクトリ名
    username = "name"

    
    # 学習対象であるモデル名の設定
    model_name = "model_00"

    # 学習対象であるモデルの種類を選択（モデルのアーキテクチャそのものを変更したい場合は、"username/src/models"以下のコードに手を加える）
    # model_type =  "multimodal"
    # model_type = "vis_enc"
    model_type = "tac_enc" 

    # multimodal モデルを学習する際は、vision_encoder および tactile_encoder として用いる事前学習のモデル名を入力。そうでない場合は"None"
    vision_model_name = "None" # e.g. "vis_model_00"
    tactile_model_name = "None" # e.g. "tac_model_00"


    #シードの選択
    seed = 42

    #シードの固定
    torch_fix_seed(seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)


    #デバイスを指定（GPU / CPU）
    device = get_device(gpu_id=0) 


    #学習に用いるデータセットを選択
    # dataset = DatasetForMultimodal(device=device) # 視覚、触覚、関節角度の情報を用いて、次の時刻の関節角度を予測
    # dataset = DatasetForVisual(device=device) # 視覚情報の事前学習（再構成）
    dataset = DatasetForTactile(device=device) # 触覚情報の事前学習（再構成）


    #訓練データと評価データの作成（バッチサイズ等の設定）
    train_data, val_data = split_train_validation(dataset=dataset, validation_split=0.12)
    train_loader = DataLoader(
        train_data, batch_size=24, shuffle=True,
        worker_init_fn=seed_worker, generator=g, 
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_data, batch_size=12, shuffle=True,
        worker_init_fn=seed_worker, generator=g
    )
    print('data_ok')


    #最適化手法の選択
    optimizer = torch.optim.AdamW()

    #学習率の選択
    lr = 0.0001


    #損失関数の選択
    # loss = mseloss # MSE / 関節角度の学習など
    loss = recon_kl # 再構成誤差＋KLD / 再構成の学習など
    # loss = cross_ent # CrossEntropy / 分類タスクの学習など
    # loss = charbo # charbonnier（差分の2乗に微少量加えて√をとった平均） / 関節角度の学習など
    # loss = w_loss # 関節の最大最小に基づいた、関節ごとの重み付け損失 / 関節角度の学習など


    #EarlyStoppingのpatience
    patience = 1000


    #学習の実行
    train_model(
        username, 
        model_name, 
        model_type, 
        vision_model_name,
        tactile_model_name,
        seed,
        device,
        train_loader,
        val_loader,
        optimizer,
        lr, 
        loss,
        patience,
        )


if __name__ == '__main__':
    main()

