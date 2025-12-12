import torch
import torch.nn as nn
from code_models.stanhop import STanHopNet
from code_models.fusion import FIATSFusion

class STanHopFIATS(nn.Module):
    def __init__(self,
                 data_dim,
                 in_len,
                 out_len,
                 seg_len,
                 win_size=4,
                 factor=10,
                 d_model=512,
                 d_ff=1024,
                 n_heads=8,
                 e_layers=3,
                 dropout=0.0,
                 baseline=False,
                 device=torch.device("cuda:0"),
                 inter_dim=384,
                 fusion_mode="crossattn"):
        super(STanHopFIATS, self).__init__()

        self.backbone = STanHopNet(
            data_dim=data_dim,
            in_len=in_len,
            out_len=out_len,
            seg_len=seg_len,
            win_size=win_size,
            factor=factor,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            baseline=baseline,
            device=device,
        )

        self.fusion = FIATSFusion(d_model=d_model, fusion_mode=fusion_mode, dropout=dropout)
        self.inter_proj = nn.Linear(inter_dim, d_model)

    def forward(self, x_seq, inter_vec=None):
        """
        x_seq: shape (B, T_in, C)
        inter_vec: shape (B, D) - daily intervention vector (Sentence-BERT or其他)
        """
        pred = self.backbone.forward(x_seq)  # (B, T_out, C)

        if inter_vec is not None:
            inter_vec = self.inter_proj(inter_vec)  # (B, d_model)
            pred = self.fusion(pred, inter_vec)     # (B, T_out, C)

        return pred
