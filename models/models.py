import torch
import numpy as np

from torch.nn.modules import transformer
from data.data_loader import RICODataset, get_sine_loader
from tqdm import tqdm
from torch import nn, optim, Tensor
from models.model_componets import FILM, AdaConvBlock, AdaLayerNorm, MixerBlock, ResBlock, ADAResBlock, SinusoidalPositionalEmbedding, SinusoidalTimeEmbedding
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from utils.logger import Logger, MlFlowLogger
from utils.stats import compute_channel_stats


class NamedODE():
    def __init__(self, name : str):
        if name == "":
            raise ValueError("Name cant be empty")
            
        self._name = name

    @property
    def name(self) -> str:
        """The name of the ODE."""
        return self._name




class CONVODE(nn.Module, NamedODE):
    def __init__(self, in_channels=3, output_channels=1, output_lenght: int | None = None):
        self._name = "CONVODE"
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels + 1, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 3, padding=1, stride=2),
            nn.LeakyReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2,
                               padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2,
                               padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(64, output_channels, 3, padding=1),
        )

        if output_lenght != None:
            self.length_adjuster = nn.AdaptiveAvgPool1d(output_lenght)
        else:
            self.length_adjuster = None

    def forward(self, x, t):
        # t: (B,) scalar → expand to match spatial dims
        t = t.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        xt = torch.cat([x, t], dim=1)
        h = self.encoder(xt)
        h = self.middle(h)
        v = self.decoder(h)
        if self.length_adjuster:
            v = self.length_adjuster(v)
        return v

class MLPODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor) -> None:
        self._name = "MLPODE"
        super().__init__()
        B, C, L = input_template.shape
        self.down_lin = nn.Sequential(
            nn.Linear(C + 1, C),
            nn.SiLU(),
            nn.Linear(C, 1),
        )
        self.up_lin = nn.Sequential(
            nn.Linear(1, C),
            nn.SiLU(),
            nn.Linear(C, C),
        )
        self.layers = nn.Sequential(
            nn.Linear(L, L * 2),
            nn.SiLU(),
            nn.Linear(L*2, L * 2),
            nn.SiLU(),
            nn.Linear(L*2, L * 2),
            nn.SiLU(),
            nn.Linear(L*2, L * 2),
            nn.SiLU(),
            nn.Linear(L*2, L * 2),
            # nn.SiLU(),
            # nn.Linear(L*2, L * 2),
            nn.SiLU(),
            nn.Linear(L*2, L)
        )
    def forward(self, x, t):
        t = t.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        xt = torch.cat([x, t], dim=1)
        out = self.down_lin(xt.permute(0,2,1))
        out = self.layers(out.permute(0,2,1))
        # out = self.up_lin(out.permute(0,2,1))
        return out
        return out.permute(0,2,1)

class MLPODE2(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor) -> None:
        self._name = "MLPODE2"
        self.hidden_dim = 1024
        super().__init__()
        B, self.C, self.L = input_template.shape
        self.layers = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
    def forward(self, x, t):
        t = t.view(-1, 1, 1).expand(x.shape[0], 1, x.shape[2])
        xt = torch.cat([x, t], dim=1)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        out = self.layers(x)
        out = out.view(out.shape[0], self.C, self.L)
        return out

class MLPODE2_3(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor) -> None:
        self._name = "MLPODE2"
        self.hidden_dim = 1024
        super().__init__()
        B, self.C, self.L = input_template.shape
        self.layers = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2,self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
    def forward(self, x, t):
        t = t.view(-1, 1, 1).expand(x.shape[0], 1, x.shape[2])
        xt = torch.cat([x, t], dim=1)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        out = self.layers(x)
        out = out.view(out.shape[0], self.C, self.L)
        return out

class MLPODE3(nn.Module, NamedODE):

    def __init__(self, input_template : Tensor) -> None:
        self._name = "MLPODE3"
        self.hidden_dim = 512
        super().__init__()
        B, self.C, self.L = input_template.shape
        self.time_embbing = SinusoidalTimeEmbedding(self.L)
        self.upscale = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   
        self.resblocks = nn.Sequential(
            *[ResBlock(layer_fn=lambda i,o : nn.Linear(self.hidden_dim, self.hidden_dim),
                        in_channels=1,
                        activation=nn.SiLU,) for _ in range(3)]
        )
        self.downscale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
        
    def forward(self, x, t):
        t = self.time_embbing(t).unsqueeze(1).expand(x.shape[0], 1, self.L)
        xt = torch.cat([x, t], dim=1)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        x = self.upscale(x)
        x = self.resblocks(x)
        out = self.downscale(x)
        out = out.view(out.shape[0], self.C, self.L)
        return out

class MLPODE4(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        self.hidden_dim = 1024
        self._name = f"MLPODE4_hidden_dim{self.hidden_dim}"
        self.time_embbing = SinusoidalTimeEmbedding(self.L)
        self.upscale = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   
        self.resblocks = nn.Sequential(
            *[ResBlock(layer_fn=lambda i,o : nn.Linear(self.hidden_dim, self.hidden_dim),
                        in_channels=1,
                        activation=nn.SiLU) for _ in range(3)]
        )
        self.downscale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
        
    def forward(self, x, t):
        t = self.time_embbing(t).unsqueeze(1).expand(x.shape[0], 1, self.L)
        xt = torch.cat([x, t], dim=1)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        x = self.upscale(x)
        x = self.resblocks(x)
        out = self.downscale(x)
        out = out.view(out.shape[0], self.C, self.L)
        return out


class SimpleMLPODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor | None =None) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        cond_B, self.cond_C, self.cond_L = cond_template.shape if cond_template is not None else (None, None, None)
        dynamic = False
        self.hidden_depth = 0
        self.downscale_factor = 1
        # if dynamic:
        #     self.hidden_dim = max(2048, self.L * self.C)
        #     self._name = f"SimpleMLPODE_hidden_dim_dynamic{self.hidden_dim}"
        # else:
        # self.hidden_dim = int(1024 * 1.5)
        self.hidden_dim = 1024
        self._name = f"SimpleMLPODE_hidden_dim{self.hidden_dim}_resblocks{self.hidden_depth}_downscale{self.downscale_factor}"

        self.time_dim = 256
        self.cond_dim = 512
        self.pos_emb = True
        self.in_dim = self.C * self.L

        self.time_embbing = SinusoidalTimeEmbedding(self.time_dim)
        if self.pos_emb:
            self.posemb = nn.Parameter(torch.zeros(1, 1, self.L))

        if self.hidden_depth == 0:
            self.time_film = FILM(self.hidden_dim, self.time_dim)
        self.upscale = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   

        self.dense = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_depth)]
        )
        self.dense2 = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_depth)]
        )
        self.conditioning = nn.ModuleList(
            [AdaLayerNorm(self.hidden_dim, self.time_dim) for _ in range(self.hidden_depth)]
        )
        self.conditioning2 = nn.ModuleList(
            [AdaLayerNorm(self.hidden_dim, self.time_dim) for _ in range(self.hidden_depth)]
        )
        self.silu = nn.SiLU()
        if self.cond_C is not None and self.cond_L is not None:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_C*self.cond_L, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.cond_dim),
            )
            self.cond_film = FILM(self.hidden_dim, self.cond_dim)
            self.cond_films = nn.ModuleList([FILM(self.hidden_dim, self.cond_dim) for _ in range(self.hidden_depth)])
        self.downscale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.downscale_factor),  
            nn.SiLU(),
            nn.Linear(self.hidden_dim * self.downscale_factor, self.L*self.C),
            
        )
        
    def forward(self, x, t, c=None):
        t_emb = self.time_embbing(t)
        if self.pos_emb:
            x = x + self.posemb
        x = x.view(x.shape[0], self.C*self.L)
        x = self.upscale(x)
        if c is not None:
            c = self.cond_proj(c.view(c.shape[0], -1))
            x = self.cond_film(x, c)
        if self.hidden_depth == 0:
            x = self.time_film(x, t_emb)

        for i, (l1, l2, adaln1, adaln2) in enumerate(zip(self.dense,self.dense2, self.conditioning,self.conditioning2)):
            ori = x
            x = l1(x)
            x = adaln1(x, t_emb)
            if c is not None:
                x = self.cond_films[i](x, c)
            x = self.silu(x)
            x = l2(x)
            x = adaln2(x, t_emb)
            x = x + ori
            x = self.silu(x)

        out = self.downscale(x)
        out = out.view(out.shape[0], self.C, self.L)

        return out

class IdentityODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor) -> None:
        super().__init__()
        self._name = "IdentityODE"
    def forward(self, x, t, c=None):
        return torch.zeros_like(x)
    


class ChannelMLPODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor | None =None) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        cond_B, self.cond_C, self.cond_L = cond_template.shape if cond_template is not None else (None, None, None)

        self.channel_dim = 128
        self.hidden_dim = self.channel_dim * self.C
        self._name = f"ChannelMLPODE_hidden_dim{self.hidden_dim}"

        self.time_embbing_dim = 256
        self.cond_dim = 512  if cond_template is not None else 0
        self.in_dim = self.C * self.L + self.cond_dim

        self.adadim = self.time_embbing_dim + self.cond_dim

        self.hidden_depth = 2
        self.time_embbing = SinusoidalTimeEmbedding(self.time_embbing_dim)
        self.upscale = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   

        self.dense = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_depth)]
        )
        self.dense2 = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_depth)]
        )
        self.conditioning = nn.ModuleList(
            [AdaLayerNorm(self.hidden_dim, self.adadim) for _ in range(self.hidden_depth)]
        )
        self.conditioning2 = nn.ModuleList(
            [AdaLayerNorm(self.hidden_dim, self.adadim) for _ in range(self.hidden_depth)]
        )
        self.silu = nn.SiLU()
        if self.cond_C is not None and self.cond_L is not None:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_C*self.cond_L, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.cond_dim),
            )

        self.channel_layers = nn.Sequential(
            nn.Linear(self.channel_dim, self.channel_dim),
            nn.SiLU(),
            nn.Linear(self.channel_dim, self.L),
        )
        
    def forward(self, x, t, c=None):
        t = self.time_embbing(t)
        x = x.reshape(x.shape[0], self.C*self.L)
        # if c is not None:
        #     cond_emb = self.film(c.view(c.shape[0], -1))
        #     g, b = cond_emb.chunk(2, dim=1)
        #     x = x * g  + b

        if c is not None:
            c = self.cond_proj(c.view(c.shape[0], -1))
            t = t.expand(c.shape[0], self.time_embbing_dim)
            cond_emb = torch.cat([t, c], dim=1)
            x = torch.cat([x, c], dim=1)
        else:
            cond_emb = t
        x = self.upscale(x)

        for l1, l2, adaln1, adaln2 in zip(self.dense,self.dense2, self.conditioning,self.conditioning2):
            ori = x
            x = l1(x)
            x = adaln1(x, cond_emb)
            x = self.silu(x)
            x = l2(x)
            x = adaln2(x, cond_emb)
            x = x + ori
            x = self.silu(x)

        out = x.reshape(x.shape[0], self.C, self.channel_dim)
        out = self.channel_layers(out)
        return out


class ShapedPriorModel(nn.Module, NamedODE):
    def __init__(self, base_model : nn.Module, dataloader, transformer) -> None:
        super().__init__()
        self._name = f"ShapedPriorModel_{base_model.name}"
        self.base_model = base_model
        self.channels_stats = torch.as_tensor(compute_channel_stats(dataloader, transformer, type="z"))
        # self.channel_stats = compute_channel_stats_per_split(dataloader, transformer, splits, norm)
        
    def channel_means(self, dataloader, transformer):
        all_coeffs = []
        out = []
        for _, y in dataloader:
            coeffs = transformer.transform(y)
            all_coeffs.append(coeffs)
        coeffs = torch.cat(all_coeffs, dim=0)
        for c in range(coeffs.shape[1]):
            out.append(coeffs[:, c, :].mean(dim=0))
        return out
    def forward(self, x, t, c=None):
        return self.base_model(x, t, c)

    def sample_prior_like(self, input_example, device="cpu"):
        out = torch.randn_like(input_example, device=device)
        channels_stats = self.channels_stats.to(device)
        return out * channels_stats[:, 1].view(1, -1, 1) + channels_stats[:, 0].view(1, -1, 1)
        # out = []
        # idk = torch.randn(batch_size, self.C, self.L, device=device) 
        # priorsplits = torch.split(idk, self.splits, dim=2)
        # for split, norm_stats in zip(priorsplits, self.channel_stats):
        #     channels=[]
        #     for c, (m, s) in enumerate(norm_stats):
        #         idk = split * s + m     
        #         channels.append(idk)
        #     out.append(torch.cat(channels, dim=1))
        # return torch.cat(out, dim=2)


class UnetODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor | None =None) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        cond_B, self.cond_C, self.cond_L = cond_template.shape if cond_template is not None else (None, None, None)


        self.time_embbing_dim = 256
        self.cond_dim = 1024  if cond_template is not None else 0

        self.adadim = self.time_embbing_dim + self.cond_dim

        self.depth = 1
        self.mult = 2**self.depth #lowest safe length for coeff with self.depth downsamples

        self.basedim = 16
        self._name = f"UnetODE_basedim{self.basedim}"
        self.upconv = nn.Conv1d(self.C, self.basedim, 3, padding=1) 
        self.encs = nn.ModuleList([
            AdaConvBlock(self.basedim*(2**i), self.basedim*(2**(i+1)), self.adadim) for i in range(self.depth)])
        self.decs = nn.ModuleList([
            AdaConvBlock(
                self.basedim * (2 ** (self.depth - i)) * 2,
                self.basedim * (2 ** (self.depth - i - 1)),
                self.adadim
            )
            for i in range(self.depth)
        ])
        self.pool = nn.MaxPool1d(2)

        bottleneck_dim = self.basedim * (2 ** self.depth)
        self.bottleneck = AdaConvBlock(bottleneck_dim, bottleneck_dim, self.adadim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.downconv = nn.Conv1d(self.basedim, self.C, 3, padding=1)
        self.time_embbing = SinusoidalTimeEmbedding(self.time_embbing_dim)

        if self.cond_C is not None and self.cond_L is not None:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_C*self.cond_L, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.SiLU(),
            )
        _, pad_len = self.pad_to_mult(input_template, self.mult)
        print("Pad length for UNetODE:", pad_len)



    def pad_to_mult(self, x, mult, mode="reflect"):
        pad_len = (mult - (x.shape[2] % mult)) % mult
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode=mode)
        return x, pad_len
    def unpad(self, x, original_len):
        return x[:,:,:original_len]

        
    def forward(self, x, t, c=None):
        t = self.time_embbing(t)
        x,_ = self.pad_to_mult(x, self.mult)

        if c is not None:
            c = self.cond_proj(c.view(c.shape[0], -1))
            t = t.expand(c.shape[0], self.time_embbing_dim)
            cond_emb = torch.cat([t, c], dim=1)
        else:
            cond_emb = t

        x = self.upconv(x)
        states = []
        for enc in self.encs:
            x = enc(x, cond_emb)
            states.append(x)
            x = self.pool(x)
        b = self.bottleneck(x, cond_emb)
        for dec, s in zip(self.decs, reversed(states)):
            b = self.upsample(b)
            cat = torch.cat([b, s], dim=1)
            b = dec(cat, cond_emb)
        out = self.downconv(b)
        return self.unpad(out, self.L)

class MLPODE7(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor | None = None, dynamic_scaling=False, hidden_dim = None) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        cond_B, self.cond_C, self.cond_L = cond_template.shape if cond_template is not None else (None, None, None)

        if dynamic_scaling:
            self.hidden_dim = self.L * 10
        elif hidden_dim is not None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = 1024
        self.cond_dim = 256
        self.time_emb_dim = 256

        self.resblock_dim = self.hidden_dim + self.cond_dim if cond_template is not None else self.hidden_dim
        if dynamic_scaling:
            self._name = f"MLPODE7_hidden_dim{self.hidden_dim}_dynamic_scaling"
        else:
            self._name = f"MLPODE7_hidden_dim{self.hidden_dim}"
        self.time_embbing = SinusoidalTimeEmbedding(self.time_emb_dim)
        self.time_proj = nn.Linear(self.time_emb_dim, self.L)
        self.upscale = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   
        if cond_template is not None:
            self.cond_upscale = nn.Sequential(
                nn.Linear(self.cond_C*self.cond_L, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim))

        self.resblocks = nn.ModuleList(
            [ADAResBlock(layer_fn= lambda i,o : nn.Linear(i,o),
                         activation=nn.SiLU,
                         emb_dim=self.resblock_dim,
                         in_channels=self.resblock_dim,
                         out_channels=self.resblock_dim,
                         time_embed_dim=self.L) for _ in range(3)]
        )
        self.downscale = nn.Sequential(
            nn.Linear(self.resblock_dim, self.L*self.C),
        )
        
    def forward(self, x, t, c=None):
        t = self.time_embbing(t)
        t = self.time_proj(t).unsqueeze(1).expand(x.shape[0], 1, self.L)
        xt = torch.cat([x, t], dim=1)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        x = self.upscale(x)
        if c is not None:
            cond_emb = self.cond_upscale(c.view(c.shape[0], -1))
            x = torch.cat([x, cond_emb], dim=1)
        t = t.squeeze(1)
        for block in self.resblocks:
            x = block(x, t)
        out = self.downscale(x)
        out = out.view(out.shape[0], self.C, self.L)
        return out

class TransformerODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor | None = None) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        # self.emb_dim = 128
        self.emb_dim = 224
        self._name = f"TransformerODE_emb_dim{self.emb_dim}"
        
        self.time_emb_dim = self.emb_dim
        self.time_embedding = SinusoidalTimeEmbedding(self.time_emb_dim)

        self.pos_emb = SinusoidalPositionalEmbedding(self.L, self.emb_dim)

        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.multiHeadAttention = nn.MultiheadAttention(self.emb_dim, num_heads=8, batch_first=True)

        self.upscale = nn.Sequential(
            nn.Linear(self.C, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim))

        if cond_template is not None:
            _ , cond_C, _ = cond_template.shape
            self.cond_upscale = nn.Sequential(
                nn.Linear(cond_C, self.emb_dim),
                nn.SiLU(),
                nn.Linear(self.emb_dim, self.emb_dim))
            self.cond_film = FILM(self.emb_dim, self.emb_dim)

        self.dense = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
        )
        self.downscale = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim//2),
            nn.SiLU(),
            nn.Linear(self.emb_dim // 2, self.C),
        )

    def forward(self, x, t, c=None):
        x = x.permute(0,2,1)
        x = self.upscale(x)

        t = self.time_embedding(t).unsqueeze(1)
        x = x + t
        
        if c is not None:
            cond_out = self.cond_upscale(c.permute(0,2,1))
            x = self.cond_film(x, cond_out)

        x = self.pos_emb(x)
        attn_out, _ = self.multiHeadAttention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.dense(self.norm2(x))
        out = self.downscale(x)
        out = out.permute(0,2,1)
        return out

class MixerODE(nn.Module, NamedODE):
    def __init__(self, input_template: Tensor, cond_template: Tensor | None = None):
        super().__init__()
        B, self.C, self.L = input_template.shape

        self.emb_dim = 128
        self.time_dim = 128
        self.cond_dim = 128

        self.depth = 2
        self._name = f"MixerODE_emb_dim{self.emb_dim}_depth{self.depth}"

        self.time_embedding = SinusoidalTimeEmbedding(self.time_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(self.C, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        if cond_template is not None:
            _, cond_C, _ = cond_template.shape
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_C, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
            self.film = FILM(self.emb_dim, self.cond_dim)



        self.pos_emb = SinusoidalPositionalEmbedding(self.L, self.emb_dim)

        self.blocks = nn.ModuleList([
            MixerBlock(
                seq_len=self.L,
                emb_dim=self.emb_dim,
                time_dim=self.time_dim,
                cond_dim=self.cond_dim,
                expansion=2,
            )
            for _ in range(self.depth)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.SiLU(),
            nn.Linear(self.emb_dim // 2, self.C),
        )

    def forward(self, x, t, c=None):
        time_emb = self.time_embedding(t)

        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_emb(x)

        if c is not None:
            c = self.cond_proj(c.permute(0, 2, 1))
            x = self.film(x, c)
        for block in self.blocks:
            x = block(x, time_emb)

        x = self.output_proj(x)
        return x.permute(0, 2, 1)

class MLPODE6(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        self.hidden_dim = 1024
        # self.conv_channels = self.C
        self._name = f"MLPODE6_hidden_dim{self.hidden_dim}"
        self.time_embbing = SinusoidalTimeEmbedding(self.L)
        self.upscale = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   
        self.resblocks = nn.Sequential(
            *[ResBlock(layer_fn=lambda i,o : nn.Linear(self.hidden_dim, self.hidden_dim),
                        in_channels=1,
                        activation=nn.SiLU) for _ in range(3)]
        )
        self.inconv = nn.Sequential(
            nn.Conv1d(self.C + 1, self.C, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(self.C, self.C, 5, padding=2),
            nn.SiLU(),
        )
        self.outconv = nn.Sequential(
            nn.Conv1d(self.C, self.C, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(self.C, self.C, 5, padding=2),
        )
        self.downscale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
        
    def forward(self, x, t):
        t = self.time_embbing(t).unsqueeze(1).expand(x.shape[0], 1, self.L)
        xt = torch.cat([x, t], dim=1)
        x = self.inconv(xt)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        x = self.upscale(x)
        x = self.resblocks(x)
        out = self.downscale(x)
        out = out.view(out.shape[0], self.C, self.L)
        out = self.outconv(out)
        return out


class CondMLPODE4(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, cond_template : Tensor) -> None:
        super().__init__()
        B, self.C, self.L = input_template.shape
        _, cond_C, cond_l = cond_template.shape
        self.hidden_dim = 1024
        self._name = f"Cond_MLPODE4_hidden_dim{self.hidden_dim}"
        self.time_embbing = SinusoidalTimeEmbedding(self.L)
        self.upscale = nn.Sequential(
            nn.Linear(self.L*(self.C + 2), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))   
        self.resblocks = nn.Sequential(
            *[ResBlock(layer_fn=lambda i,o : nn.Linear(i, o),
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        activation=nn.SiLU,
                        ) for _ in range(3)]
        )
        self.downscale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
        self.conditional_compaction = nn.Sequential(
            nn.Linear(cond_C, self.C),
            nn.SiLU(),
            nn.Conv1d(self.C, 1, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.SiLU(),
            nn.Linear(cond_l, self.L)
        )

        
    def forward(self, x, t, c):
        t = self.time_embbing(t).unsqueeze(1).expand(x.shape[0], 1, self.L)
        c = self.conditional_compaction(c)
        xt = torch.cat([x, t, c], dim=1)
        x = xt.view(xt.shape[0], (self.C + 2)*self.L)
        x = self.upscale(x)
        x = self.resblocks(x)
        out = self.downscale(x)
        out = out.view(out.shape[0], self.C, self.L)
        return out



class WaveletLayerdODE(nn.Module, NamedODE):
    def __init__(self, model1, model2, split) -> None:
        self._name = f"WaveletLayerdODE_{model1.name}_{model2.name}_split{split}" 
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.split = split
    
    def forward(self, x, t):
        x1 = x[:,:,:self.split]
        x2 = x[:,:,self.split:]
        out1 = self.model1(x1, t)
        out2 = self.model2(x2, t)
        return torch.cat([out1, out2], dim=2)

class MLPODE3_multichannel(nn.Module, NamedODE):
    def __init__(self, input_template: Tensor) -> None:
        super().__init__()
        self._name = "MLPODE3_multichannel"
        self.hidden_dim = 256
        B, self.C, self.L = input_template.shape

        self.upscale = nn.Sequential(
            nn.Linear(self.L * (self.C + 1), self.hidden_dim * self.C),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * self.C, self.hidden_dim * self.C)
        )

        self.resblocks = nn.ModuleList([
            nn.Sequential(
                *[ResBlock(
                    layer_fn=lambda i, o: nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                    in_channels=1,
                    activation=nn.SiLU,
                ) for _ in range(3)]
            ) for _ in range(self.C)
        ])

        self.downscaler = nn.ModuleList([
            nn.Linear(self.hidden_dim * 2, self.L) for _ in range(self.C)]
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_1 = t.view(-1, 1, 1).expand(x.shape[0], 1, x.shape[2])  
        xt = torch.cat([x, t_1], dim=1)  
        x_flat = xt.view(xt.shape[0], (self.C + 1) * self.L)
        x_up = self.upscale(x_flat)  

        x_up = x_up.view(x_up.shape[0], self.C, self.hidden_dim)  
        outs = []
        for c in range(self.C):
            #re add time dim for each channel
            t_2 = t.view(-1, 1, 1).expand(x_up.shape[0], 1, x_up.shape[2])
            x_c = torch.cat([x_up[:, c:c+1, :], t_2], dim=1)
            x_c_flat = x_c.view(x_c.shape[0], (2) * self.hidden_dim)
            out_c = self.resblocks[c](x_c_flat)  
            out_down_scaled = self.downscaler[c](out_c)
            outs.append(out_down_scaled.unsqueeze(1)) 
        out = torch.cat(outs, dim=1)
        return out

class MLPODE2_1(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor) -> None:
        self._name = "MLPODE2_1"
        self.hidden_dim = 1024
        super().__init__()
        B, self.C, self.L = input_template.shape
        self.layers = nn.Sequential(
            nn.Linear(self.L*(self.C + 1), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.L*self.C),
        )
        self.channel_wise = nn.Sequential(
            nn.Linear(self.C + 1, self.C),
            nn.SiLU(),
            nn.Linear(self.C, self.C),
            nn.SiLU(),
            nn.Linear(self.C, self.C),
        )
    def forward(self, x, t):
        t = t.view(-1, 1, 1).expand(x.shape[0], 1, x.shape[2])
        xt = torch.cat([x, t], dim=1)
        x = xt.view(xt.shape[0], (self.C + 1)*self.L)
        x = self.layers(x)
        x = x.view(x.shape[0], self.C, self.L)
        xt = torch.cat([x, t], dim=1)
        out = self.channel_wise(xt.permute(0,2,1)).permute(0,2,1)
        return out





class MultiLayerWaveletODE(nn.Module, NamedODE):
    def __init__(self, input_template : Tensor, indexes : list[int]):
        self._name = "MultiLayerWaveletODE"
        super().__init__()
        B, C, L = input_template.shape
        self.conv = nn.Conv1d(C + 1, C*16, 5, 1, 2)
        self.upconv = nn.Conv1d(1, C, 3,1,1)
        self.flatten = nn.Flatten()
        self.dropout = 0.2
        norm = nn.BatchNorm1d
        aktiv = nn.SiLU
        self.resblocks = nn.Sequential(
            *[ResBlock(layer_fn=lambda i,o : nn.Conv1d(i,o,3,1,1),
                       in_channels=C*16,
                       activation=aktiv,
                       norm=norm, dropout=0.1) for _ in range(10)]
        )
        self.dense = nn.Sequential(
            nn.Linear(L*C*16, L*C*8),
            # norm(L*C*8),
            nn.Dropout(self.dropout),
            aktiv(),
            nn.Linear(L*C*8, L*C*8),
            nn.Dropout(self.dropout),
            # norm(L*C*8),
            aktiv(),
            nn.Linear(L*C*8, L*C),
            nn.Dropout(self.dropout),
            # norm(L*C),
            aktiv(),
            nn.Linear(L*C, L),
        )

    def forward(self, x, t):
        t = t.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        xt = torch.cat([x, t], dim=1)
        c = self.conv(xt)
        c = self.resblocks(c)
        f = self.flatten(c)
        out = self.dense(f)
        out = out.unsqueeze(1)
        out = self.upconv(out)
        return out 

