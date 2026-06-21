from torch import nn, optim, Tensor
import torch
import math
from torch.nn import functional as F
from typing import Callable, Optional

class ResBlock(nn.Module):
    def __init__(
        self,
        layer_fn: Callable[[int, int], nn.Module],
        in_channels: int,
        out_channels: Optional[int] = None,
        activation: Callable[..., nn.Module] = nn.LeakyReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.layer1 = layer_fn(in_channels, out_channels)
        self.layer2 = layer_fn(out_channels, out_channels)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.layer1(x)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.layer2(out)
        skip = identity
        out = out + skip
        out = self.activation(out)
        return out

class ADAResBlock(nn.Module):
    def __init__(
        self,
        layer_fn: Callable[[int, int], nn.Module],
        time_embed_dim : int,
        in_channels: int,
        emb_dim: int,
        out_channels: Optional[int] = None,
        activation: Callable[..., nn.Module] = nn.LeakyReLU,
        skip_fn: Callable[[], nn.Module] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.layer1 = layer_fn(in_channels, out_channels)
        self.layer2 = layer_fn(out_channels, out_channels)
        self.skip_fn = skip_fn() if skip_fn is not None else None

        self.activation = activation()
        self.norm = AdaLayerNorm(embed_dim=emb_dim, cond_dim=time_embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    def forward(self, x: Tensor, timeEmbedding : Tensor) -> Tensor:
        identity = x

        out = self.layer1(x)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.layer2(out)
        out = self.norm(out, timeEmbedding)
        if self.skip_fn != None:
            skip = self.skip_fn(identity)
        else:
            skip = identity

        out = out + skip
        out = self.activation(out)
        return out

class TransformerBlock(nn.Module):
    """Transformer block with time conditioning for wavelet coefficient processing. Inspired by WaveletDIFF https://arxiv.org/abs/2510.11839"""
    
    def __init__(self, emb_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, time_embed_dim=64):
        super().__init__()
        self.dim = emb_dim
        self.num_heads = num_heads
        
        # Self-attention
        self.norm1 = AdaLayerNorm(emb_dim, time_embed_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.norm2 = AdaLayerNorm(emb_dim, time_embed_dim)
        hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_embed, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, dim, seq_leng)
            time_embed: Time embedding of shape (batch_size, time_embed_dim)
            mask: Optional attention mask
        
        Returns:
            Transformed tensor of same shape as input
        """
        # Self-attention with time conditioning
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, dim)
        x_norm1 = self.norm1(x, time_embed)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with time conditioning
        x_norm2 = self.norm2(x, time_embed)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        return x.permute(0, 2, 1)  # (batch_size, dim, seq_len)


#SinusoidalTimeEmbedding adapted from WaveletDIFF https://arxiv.org/abs/2510.11839
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        if t.dim() == 2:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim

    def forward(self, x):
        position = torch.arange(self.seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=position.device) * -(math.log(10000.0) / self.dim))
        pe = torch.zeros(self.seq_len, self.dim, device=position.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe


#AdalayerNorm implementation adapted from WaveletDIFF https://arxiv.org/abs/2510.11839
class AdaLayerNorm(nn.Module):
    
    def __init__(self, embed_dim, cond_dim, multichannel=False):
        super().__init__()
        self.multichannel = multichannel
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        
        self.ada_lin = nn.Linear(cond_dim, 2 * embed_dim)
        
        if multichannel:
            self.pool = nn.Sequential(
                nn.Conv1d(cond_dim, cond_dim, 1),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1)
            )

        with torch.no_grad():
            self.ada_lin.weight.zero_()
            self.ada_lin.bias.zero_()
            self.ada_lin.bias[:embed_dim] = 1.0
    
    def forward(self, x, cond_embed):
        if self.multichannel:
            cond_embed = self.pool(cond_embed.transpose(1,2)).squeeze(-1)

        assert cond_embed.dim() == 2 ,"Cond embedding dim must be 2"
        x_norm = self.norm(x)
        
        ada_params = self.ada_lin(cond_embed)
        
        scale, shift = ada_params.chunk(2, dim=-1)
        
        if scale.dim() != x_norm.dim():
            scale = scale.unsqueeze(1)
        if shift.dim() != x_norm.dim():
            shift = shift.unsqueeze(1)
        return scale * x_norm + shift


class FILM(nn.Module):
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        print(embed_dim, cond_dim)
        self.film_lin = nn.Linear(cond_dim, 2 * embed_dim)

        with torch.no_grad():
            self.film_lin.weight.zero_()
            self.film_lin.bias.zero_()
            self.film_lin.bias[:embed_dim] = 1.0
    
    def forward(self, x, cond_embed):
        #if cond_embed is 3D and its sequence length doesn't match x's, interpolate it to match x's sequence length
        if cond_embed.dim() == 3 and x.dim() == 3 and cond_embed.shape[1] != x.shape[1]:
            cond_embed = cond_embed.transpose(1, 2)
            cond_embed = F.interpolate(cond_embed, size=x.shape[1], mode="linear", align_corners=False)
            cond_embed = cond_embed.transpose(1, 2)

        
        film_params = self.film_lin(cond_embed)
        
        scale, shift = film_params.chunk(2, dim=-1)
        
        if scale.dim() != x.dim():
            scale = scale.unsqueeze(1)
        if shift.dim() != x.dim():
            shift = shift.unsqueeze(1)
        return scale * x + shift



class GroupAdaLayerNorm(nn.Module):
    def __init__(self, embed_dim, cond_dim, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, embed_dim, affine=False)
        self.ada_lin = nn.Linear(cond_dim, 2 * embed_dim)

        with torch.no_grad():
            self.ada_lin.weight.zero_()
            self.ada_lin.bias.zero_()
            self.ada_lin.bias[:embed_dim] = 1.0
    
    def forward(self, x, cond_embed):
        x_norm = self.norm(x)
        
        ada_params = self.ada_lin(cond_embed)  # (batch_size, 2 * embed_dim)
        
        scale, shift = ada_params.chunk(2, dim=-1)  # Each: (batch_size, embed_dim)
        
        if scale.dim() != x_norm.dim():
            scale = scale.unsqueeze(-1)
        if shift.dim() != x_norm.dim():
            shift = shift.unsqueeze(-1)
        return scale * x_norm + shift

class AdaConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_emb_dim,
        kernel_size=3,
        padding=1,
        num_groups=8
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = GroupAdaLayerNorm(out_channels, cond_emb_dim, num_groups=num_groups)
        self.norm2 = GroupAdaLayerNorm(out_channels, cond_emb_dim, num_groups=num_groups)
        self.activation = nn.SiLU()
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, cond_emb):
        h = self.conv1(x)
        h = self.norm1(h,cond_emb)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h,cond_emb)
        h = self.activation(h)
        return h + self.skip_conv(x)

class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, emb_dim: int, time_dim: int, cond_dim = None, expansion: int = 4):
        super().__init__()
        self.token_norm = AdaLayerNorm(emb_dim, time_dim)
        if cond_dim is not None:
            self.cond_dim = FILM(emb_dim, cond_dim)
        self.channel_norm = AdaLayerNorm(emb_dim, time_dim)

        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion),
            nn.SiLU(),
            nn.Linear(seq_len * expansion, seq_len),
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.SiLU(),
            nn.Linear(emb_dim * expansion, emb_dim),
        )

    def forward(self, x, t_emb, c=None):
        y = self.token_norm(x, t_emb)
        if c is not None:
            y = self.cond_dim(y, c)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.channel_norm(x, t_emb)
        y = self.channel_mlp(y)
        x = x + y

        return x
