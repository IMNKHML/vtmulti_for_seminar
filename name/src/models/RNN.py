""" RNN系のモデル """


import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        device: torch.device
    ):

        super().__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(in_size, self.hidden_size, self.num_layers, batch_first=True)
        self.device = device
        self.is_rnn = True
        self.to(device)

    def forward(self, in_states):
        hidden_states, self.hidden_state = \
            self.rnn(in_states, self.hidden_state) #出力：（入力の系列データに対応する隠れ状態, 最後の隠れ状態）

        hs = hidden_states

        return hs

    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size
        ).to(self.device)


class LSTM(nn.Module):
    
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        # out_size: int,
        device: torch.device
    ):

        super().__init__()
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(in_size, self.hidden_size, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, out_size)
        self.act = nn.Tanh()

        self.to(device)
        self.device = device
        self.is_rnn = True

    def forward(self, in_states):
        # print('is:', in_states.shape)
        # print('h0:', self.hidden_state[0].shape)
        # print('c0:', self.hidden_state[1].shape)

        hidden_states, self.hidden_state = \
            self.rnn(in_states, self.hidden_state)
        
        hs = hidden_states
        # predictions = self.act(self.fc(hidden_states))
        hs_act = self.act(hs)

        return hs_act

    def init_hidden(self, batch_size):
        self.hidden_state = (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size
            ).to(self.device),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_size
            ).to(self.device)
        )
