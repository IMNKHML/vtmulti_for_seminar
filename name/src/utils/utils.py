import random
import numpy as np
import torch


def torch_fix_seed(seed=42):
    """乱数を固定する関数

    各行でやっていることは
        https://qiita.com/north_redwing/items/1e153139125d37829d2d
    などに詳細あり

    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """early stopping を行うクラス

    監視する値が　!最小値!を更新した時，そのモデルをstate_dict形式で保存する

    Args:
        save_dir(str): モデルを保存する !ディレクトリ名!
        patience(int): 何エポック更新されなかったら終了するか
        verbose(bool): ログを出力するかどうか．オンにすると毎エポック何かしら出力してくれる．True推奨

    Examples:
        Training前に以下みたいに定義して

        early_stopping = EarlyStopping(patience=30, save_dir='hoge')

        Trainingのiterationで

        early_stopping(loss=validation_loss, model=model, epoch=epoch)
        if early_stopping.early_stop:
            break

        とすることで， hoge/model.pth に最良のモデルが保存されていく．

    """

    def __init__(self, username, model_name, patience, verbose=True):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.loss_min = np.Inf
        self.username = username
        self.model_name = model_name
        self.verbose = verbose

    def __call__(self, loss, model):
        """

        Args:
            loss: 監視する値
            model: 保存するモデル

        """
        if loss > self.loss_min:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

            if self.verbose:
                print(f'(Counter: {self.counter} / {self.patience})')

        else:
            self._save_model(loss, model)
            self.counter = 0

    def _save_model(self, loss, model):
        if self.verbose:
            print(f'(Decreased {self.loss_min-loss})')

        torch.save(model.state_dict(), f'{self.username}/models/{self.model_name}/model.pth')
        self.loss_min = loss

    def __repr__(self) -> str:
        return f'EarlyStopping(patience={self.patience})'


def seed_worker(worker_id):
    """

    DataLoaderのworkerの固定
    Dataloaderの乱数固定にはgeneratorの固定も必要らしい

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(gpu_id=-1):
    """

    使えるならGPUを使う

    """
    if gpu_id >= 0 and torch.cuda.is_available():
        print('Using GPU')
        return torch.device("cuda", gpu_id)
    else:
        print('Using CPU')
        return torch.device("cpu")


def save_hyperparameters(
        username, model_name, model, train_loader,
        loss_fn, optimizer, val_loader, early_stopping, scheduler):
    """

    モデルの構造とかハイパーパラメータとかを
    save_dir/hparams.mdにmarkdown形式で片っ端から記録していく

    """

    with open(f'{username}/models/{model_name}/hparams.md', 'w+') as f:
        # model
        f.write('# Model\n\n```\n')
        f.write(repr(model))
        f.write('\n```\n')

        # loss_fn
        f.write('\n# Loss function\n\n```')
        f.write(f'{loss_fn.__doc__}\n```\n')

        # optimizer
        f.write(f'\n# Optimizer\n\n```\n{optimizer}\n```\n')

        # train_loader
        f.write('\n# train dataloader\n\n```\n')
        for i in [i for i in dir(train_loader) if not i.startswith('_')]:
            f.write(f'{i}: {getattr(train_loader, i)}\n')
        f.write('```\n')

        # validation loader
        f.write('\n# validation dataloader\n\n```\n')
        for i in [i for i in dir(val_loader) if not i.startswith('_')]:
            f.write(f'{i}: {getattr(val_loader, i)}\n')
        f.write('```\n')

        # early stopping
        f.write('\n# early stopping\n\n```\n')
        f.write(f'{early_stopping}\n```\n')

        # scheduler
        f.write('\n# lr scheduler\n\n```\n')
        f.write(f'{scheduler}\n```\n')

    print(f'saved hyperparamerters on models/{model_name}/hparams.md')
