import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query: (B, T, d_model)
        key_value: (B, d_model)
        """
        B, T, d_model = query.shape
        query_proj = self.query_proj(query)                      # (B, T, d_model)
        key_proj = self.key_proj(key_value).unsqueeze(1)         # (B, 1, d_model)
        value_proj = self.value_proj(key_value).unsqueeze(1)     # (B, 1, d_model)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(query_proj, key_proj.transpose(-2, -1)) / (d_model ** 0.5)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, value_proj)        # (B, T, d_model)
        output = self.out_proj(attended)                         # (B, T, d_model)
        return output


class FIATSFusion(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, fusion_mode="crossattn", multi_source=True):
        super(FIATSFusion, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.fusion_mode = fusion_mode
        self.multi_source = multi_source

        if fusion_mode == "crossattn":
            self.attn_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, dropout)
                for _ in range(3)  # 最多支持3个 source（如 news、reddit、policy）
            ])
        elif fusion_mode == "concat":
            self.concat_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, backbone_output, *ext_vecs):
        """
        backbone_output: (B, T, d_model)
        ext_vecs: List[(B, d_model)], 可变数量的外部信息源
        返回: (B, T, d_model)
        """
        if not ext_vecs:
            return backbone_output

        if self.fusion_mode == "add":
            # 先广播扩展维度后逐元素加
            ext_sum = sum(vec.unsqueeze(1).expand_as(backbone_output) for vec in ext_vecs)
            return backbone_output + ext_sum

        elif self.fusion_mode == "concat":
            # 仅使用第一个外部向量
            ext = ext_vecs[0].unsqueeze(1).expand_as(backbone_output)
            concat = torch.cat([backbone_output, ext], dim=-1)
            return self.concat_proj(concat)

        elif self.fusion_mode == "crossattn":
            # 每个 source 分别进行 cross-attention，再加权合并
            outputs = []
            for i, ext in enumerate(ext_vecs):
                if i >= len(self.attn_blocks):
                    break
                out = self.attn_blocks[i](backbone_output, ext)
                outputs.append(out)
            fusion = backbone_output + sum(outputs)  # 残差连接
            return fusion

        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
