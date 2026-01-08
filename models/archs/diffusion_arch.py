import torch
from torch import nn 

from einops import repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from diff_utils.model_utils import LayerNorm, RelPosBias, Attention, SinusoidalPosEmb, MLP, FeedForward, default, exists
from models.archs.encoders.conv_pointnet import ConvPointnet

from typing import Optional, Tuple

class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim, 
        depth,
        dim_in_out=None,
        cross_attn=False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True, 
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True, 
        normformer = False,
        rotary_emb = True, 
        **kwargs
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)
        point_feature_dim = kwargs.get('point_feature_dim', dim)

        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
        else:
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, out_dim=dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.cross_attn = cross_attn

    def forward(self, x, context: Optional[torch.Tensor] = None):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        if self.cross_attn:
            assert context is not None, "Context must be provided for cross attention"

            for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now 
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x 
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now 
                #print("x2 shape, context shape: ", x.shape, context.shape)
                
                #print("x3 shape, context shape: ", x.shape, context.shape)
                x = ff(x) + x
        
        else:
            for idx, (attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = attn(x, attn_bias = attn_bias)
                else:
                    x = attn(x, attn_bias = attn_bias) + x
                #print("x2 shape: ", x.shape)
                x = ff(x) + x
                #print("x3 shape: ", x.shape)

        out = self.norm(x)
        return self.project_out(out)

class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_in_out: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        num_time_embeds: int = 1,
        cond: Optional[bool] = None,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds: int = num_time_embeds
        self.dim: int = dim        # 768
        self.cond: bool = cond     # True
        self.cross_attn: bool = kwargs.get('cross_attn', False)               # True
        self.point_feature_dim: int = kwargs.get('point_feature_dim', dim)    # 128

        self.dim_in_out: int = default(dim_in_out, dim)     # 768

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer = CausalTransformer(dim = dim, dim_in_out=self.dim_in_out, **kwargs)

        if cond:
            # output dim of pointnet needs to match model dim; unless add additional linear layer
            self.pointnet = ConvPointnet(c_dim=self.point_feature_dim)

    def forward(
        self,
        data: Tuple[torch.Tensor, torch.Tensor], 
        diffusion_timesteps: torch.Tensor,
    ):

        if self.cond:
            assert type(data) is tuple

            # data: latent          (B, 768)
            # cond: partial PCD     (B, N, 3)
            data, cond = data # adding noise to cond_feature so doing this in diffusion.py
            cond_feature = self.pointnet(p=cond, query=cond)


        batch = data.shape[0]

        time_embed = self.to_time_embeds(diffusion_timesteps)
        data = data.unsqueeze(1)
        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)
        tokens = torch.cat([time_embed, data, learned_queries], dim = 1) # (B, 3, 768) 

        if self.cross_attn:
            tokens = self.causal_transformer(tokens, context=cond_feature)
        else:
            tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :]

        return pred

