from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

def recon_kl(predictions, targets):
    """
        事前学習用。画像の予測再構成誤差*1e10+KLD。

    """
    Img_hat, mu, logvar = predictions

    I_target = targets

    image_prediction_loss = nn.MSELoss()(
        torch.flatten(Img_hat), torch.flatten(I_target))
    
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return image_prediction_loss * 1e8 + kld, \
        image_prediction_loss * 1e8, kld

def mseloss(predictions, targets):
    """
        関節角度学習用。

    """
    loss = nn.MSELoss()(predictions, targets)
    return loss


def cross_ent(predictions, targets):
    """

    Args:
        predictions:(BS, n)
        targets:(BS, n) / onehot vector

    Reruens:
        CrossEntropyLossを計算して返す

    """
    # print('pred:', predictions)
    # print('tar:', targets)
    loss = nn.CrossEntropyLoss()
    output = loss(predictions, targets)
    # print('loss:', output)
    return output


def charbo(predictions, targets):
    """
    charbonnier
    """
    loss = torch.mean(torch.sum(torch.sqrt((predictions - targets)**2+10.0**(-6)), dim=[-2, -1]))
    return loss

def w_loss(predictions, targets):
    #損失関数の重みづけ
    # w = torch.tensor([1.0/1.50, 1.0/0.50, 1.0/0.73, 1.0/1.31, 1.0/0.32, 1.0/1.05, 1.0/1.8]).to(predictions.device) # w : (7,)
    w = torch.tensor([1.50, 0.50, 0.73, 1.31, 0.32, 1.05, 1.8]).to(predictions.device)

    loss = torch.mean(torch.sum((torch.mul((predictions - targets)**2, w)), dim=[-2, -1]))

    return loss

class LossHolder:
    def __init__(self):
        self.__loss = []
        self.__image_loss = []
        self.__kld = []

        self.loss = []
        self.image_loss = []
        self.kld = []

    def __iadd__(self, packed_loss):

        if len(packed_loss) == 1:
            self.__loss.append(packed_loss.item())

        elif len(packed_loss) == 3:
            self.__loss.append(packed_loss[0].item())
            self.__image_loss.append(packed_loss[1].item())
            self.__kld.append(packed_loss[2].item())

        return self

    def calc(self):
        self.loss.append(np.mean(self.__loss))
        self.image_loss.append(np.mean(self.__image_loss))
        self.kld.append(np.mean(self.__kld))

        self.__loss.clear()
        self.__image_loss.clear()
        self.__kld.clear()


@dataclass
class LossLogger:
    model_name: str
    train: LossHolder = LossHolder()
    val: LossHolder = LossHolder()

# if __name__ == '__main__':
#     predictions = torch.ones(2, 5, 7)
#     targets = torch.zeros(2, 5, 7)
#     loss = loss_fnc5(predictions, targets)
#     print(loss)