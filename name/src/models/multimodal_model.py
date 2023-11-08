import torch
import torch.nn as nn
from RNN import GRU
from RNN import LSTM
from tactile_model import TactileEncoder
from visual_model import VisionEncoder
from data.build_features import change_brightness, add_noise

# Define the multimodal fusion model
class MultimodalFusionModel(nn.Module):
    def __init__(
            self, 
            device,
            vision_encoder, 
            tactile_encoder, 
            fusion_unit,
            hidden_dim=364,
            middle_dim_1=32,
            middle_dim_2=16,
            target_dim=5,
            ):
        super().__init__()

        self.to(device)

        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.fusion_unit = fusion_unit

        self.mish = nn.Mish()
        self.tanh = nn.Tanh()

        self.gru_layer = GRU(
            in_size = self.fusion_unit.mlt_dim,
            hidden_size = hidden_dim,
            device = device
        )

        self.pred_layer1 = nn.Linear(
            in_features = hidden_dim, 
            out_features = middle_dim_1,
            bias=True,
            device=device
            )
        
        self.pred_layer2 = nn.Linear(
            in_features = middle_dim_1, 
            out_features = middle_dim_2,
            bias=True,
            device=device
            )
        
        self.pred_layer3 = nn.Linear(
            in_features = middle_dim_2, 
            out_features = target_dim,
            bias=True,
            device=device
            )
        

    def forward(self, inputs):

        joint_inputs, vision_inputs, tactile_inputs = inputs

        batch_size, length, _ = joint_inputs.size()

        vision_outputs, _ = self.vision_encoder(vision_inputs)
        tactile_outputs, _ = self.tactile_encoder(tactile_inputs)

        j_hats, mlt_reps, hs_nexts, out1s, out2s = [], [], [], [], []

        for t in range(length):

            vis_t = vision_outputs[:, t:t+1]
            #備考：.unsqueeze(1)必要?
            tac_t = tactile_outputs[:, t:t+1]

            hs_t = self.gru_layer.hidden_state.clone().detach()

            if t == 0:
                j_t = joint_inputs[:, t:t+1]
            else:
                j_t = j_hats[t-1]

            mlt_rep = self.fusion_unit(j_t, vis_t, tac_t, hs_t)

            hs = self.gru_layer(mlt_rep)
            hs_next = hs[:, -1, :] 

            out1 = self.mish(self.pred_layer1(hs_next))
            out2 = self.mish(self.pred_layer2(out1))
            j_hat = self.tanh(self.pred_layer3(out2))
            
            # Append results to the lists
            j_hats.append(j_hat)
            mlt_reps.append(mlt_rep)
            hs_nexts.append(hs_next)
            out1s.append(out1)
            out2s.append(out2)

        # Concatenate the lists into tensors
        j_hats = torch.cat(j_hats, dim=1)
        mlt_reps = torch.cat(mlt_reps, dim=1)
        hs_nexts = torch.cat(hs_nexts, dim=1)
        out1s = torch.cat(out1s, dim=1)
        out2s = torch.cat(out2s, dim=1)

        return j_hats, mlt_reps, hs_nexts, out1s, out2s, vision_outputs, tactile_outputs

    def init_hidden(self, batch_size):

        self.gru_layer.init_hidden(batch_size)
