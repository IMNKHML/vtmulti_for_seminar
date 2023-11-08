import torch
import torch.nn as nn

class ConcatUnit(nn.Module):
    """
    concatによる特徴量の結合およびマルチモーダル表現の形成
    """
    def __init__(self, 
            device, 
            joint_dim=5, 
            vis_dim=4, 
            tac_dim=4, 
            hidden_dim=64,
            hs_t_out_dim = 4
            ):
        super().__init__()

        self.joint_dim = joint_dim
        self.vis_dim = vis_dim
        self.tac_dim = tac_dim
        self.hidden_dim = hidden_dim
        self.hs_t_out_dim = hs_t_out_dim
        self.mlt_dim = self.joint_dim + self.vis_dim + self.tac_dim + self.hs_t_out_dim

        self.fc_1 = nn.Linear(self.hidden_dim, self.hs_t_out_dim)
        self.mish = nn.Mish()
        self.tanh = nn.Tanh()

        self.to(device)

    def forward(self, j_t, vis_t, tac_t, hs_t):

        #gru隠れ層の次元を落とす
        h = self.tanh(self.fc_1(hs_t))

        #concatによりマルチモーダル表現を形成
        multi_rep = torch.cat([j_t, vis_t, tac_t, h.permute(1, 0, 2)], dim=2)

        return multi_rep
    