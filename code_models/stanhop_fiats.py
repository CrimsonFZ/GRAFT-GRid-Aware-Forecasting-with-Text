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
                 inter_dims=None,  # dict: {source_name: dim}
                 fusion_mode="crossattn"):
        super(STanHopFIATS, self).__init__()

        if inter_dims is None:
            inter_dims = {}  # 默认空字典，兼容不带外部信息的情况

        # 主干预测模型
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

        # 输出维度统一映射至 d_model
        self.backbone_output_proj = nn.Linear(data_dim, d_model)

        # 外部干预源独立投影模块（如 news, reddit, policy）
        self.inter_proj = nn.ModuleDict()
        for source, dim in inter_dims.items():
            if dim and dim > 0:
                self.inter_proj[source] = nn.Linear(dim, d_model)

        # 多源融合器
        self.fusion = FIATSFusion(
            d_model=d_model,
            fusion_mode=fusion_mode,
            dropout=dropout,
            multi_source=True
        )

        # 输出还原层
        self.output_proj = nn.Linear(d_model, data_dim)

    def forward(self, x_seq, **inter_vecs):
        """
        参数:
            x_seq: (B, T_in, data_dim) - 主输入序列
            inter_vecs: dict, 如 {'news': Tensor, 'reddit': Tensor}
                        每个维度为 (B, inter_dim)
        返回:
            pred: (B, T_out, data_dim)
        """
        # 主干网络前向传播
        hidden = self.backbone(x_seq)               # (B, T_out, data_dim)
        hidden = self.backbone_output_proj(hidden)  # (B, T_out, d_model)

        # 处理外部信息
        ext_embeds = []
        for key, proj in self.inter_proj.items():
            if key in inter_vecs and inter_vecs[key] is not None:
                ext = inter_vecs[key]               # (B, inter_dim)
                ext_proj = proj(ext)                # (B, d_model)
                ext_embeds.append(ext_proj)

        # 多源融合
        if ext_embeds:
            hidden = self.fusion(hidden, *ext_embeds)

        # 输出还原
        pred = self.output_proj(hidden)             # (B, T_out, data_dim)
        return pred
