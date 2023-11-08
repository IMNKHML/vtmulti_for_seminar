from itertools import product
import matplotlib.pyplot as plt

# import sklearn #機械学習のライブラリ
# from sklearn.decomposition import PCA #主成分分析器
# from sklearn.preprocessing import scale
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

def visualize_joint_data(dataset, in_idx):
    """ データセット内の関節角度のtargetの可視化

    Args:
        dataset: データセット
        in_idx(int): 予測したターゲットデータのindex

    Returns:
        axes(List[subplot]):
            各関節次元のサブプロットが入ったリスト

    """
    fig = plt.figure()
    fig.tight_layout()

    axes = [fig.add_subplot(dataset.dim, 1, i + 1)
            for i in range(dataset.dim)]

    for i, d in product(range(2), range(dataset.dim)):
        axes[d].plot(
            dataset.joint_targets[in_idx % 10 + i * 10, :, d],
            color='gray',
            linestyle='solid' if in_idx == (in_idx % 10 + i * 10) else 'dashed'
        )
        axes[d].set_title(f'dim {d}')

    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95, hspace=1)
    return axes


def visualize_joint_prediction(prediction, dataset, save_path, in_idx):
    """ 関節角度の予測結果の可視化

    Args:
        prediction(torch.Tensor): 予測結果
        dataset: データセット
        save_path(str): 保存する画像の相対パス
        in_idx(int): 予測したターゲットデータのindex

    Returns:
        None

    """

    axes = visualize_joint_data(dataset, in_idx)
    prediction = prediction.cpu().detach().numpy()
    for d, axe in enumerate(axes):
        axe.plot(prediction[:, d], color='blue')
    plt.savefig(save_path)
    plt.clf()
    plt.close()

def save_loss(username, logger_1):
    """ ロスを可視化する

    Args:
        logger(LossLogger)

    """

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    plt.suptitle(f'loss_and_acc / model:{logger_1.model_name}')
    ax1 = fig.add_subplot(111)
    ax1.plot(logger_1.train.loss,
            linestyle="solid", label='train_loss', color='red')

    ax1.plot(logger_1.val.loss,
            linestyle="solid", label='val_loss', color='blue')
    ax1.set_yscale('log')
    ax1.legend()
    plt.rcParams["font.size"] = 18

    plt.savefig(f'{username}/outputs/learning_results/losses/{logger_1.model_name}/loss.png')
    plt.clf()
    plt.close()


