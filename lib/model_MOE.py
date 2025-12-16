import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
# from lib.pos_embed import get_3d_sincos_pos_embed
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=32):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Adapter_Layer(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        # self.non_linearity = args.non_linearity  # use ReLU by default

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.adapter = Adapter_Layer(dim, dim)

    def forward(self, x):
        # _x = x[:,1:, :] + x[:,1:, :] * mask
        # x = torch.cat((x[:,:1,:], _x), dim=1)
        
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.adapter(self.mlp(self.norm2(x))))
        # x = x + x *mask
        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, nx, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        n_state = nx
        # self.c_attn = Linear(
        #     nx, n_state, 
        #     r=8
        # )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # x = self.c_attn(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT_ct(nn.Module):
    def __init__(self, *, image_size=256, image_patch_size=16, frames=16, frame_patch_size=2, num_classes=4, dim=1024, depth=6, heads=8, mlp_dim=2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 decoder_embed_dim=512, 
            decoder_depth=8, 
            decoder_num_heads=16,
            norm_layer=nn.LayerNorm,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None, 
            drop_rate=0., 
            attn_drop_rate=0.,
            drop_path_rate=0.,
            mask_ratio = 0.75):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        self.patch_dim = image_patch_size
        self.frame_patch_size = frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=image_patch_size, in_chans=1, embed_dim=dim, num_frames=frames, tubelet_size=frame_patch_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim)
        # self.text_proj = nn.Linear(4096, dim)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embedding.shape[-1], self.num_h, self.num_c, cls_token=True)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_h, self.num_c, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, input):
        input = input
        skip_list = []
        x = self.patch_embed(input)
        # mask = self.patch_embed(rois)
        b, n, _ = x.shape
        x += self.pos_embedding[:, 1:,:]
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # mask = torch.cat((cls_tokens, mask), dim=1)
        for blocks in self.blocks:
            skip_list.append(x)
            x = blocks(x)
        x = self.norm(x)
        return x
    
    def forward(self, input):
        latent = self.forward_encoder(input)
        return latent
        # x = latent.mean(dim = 1) if self.pool == 'mean' else latent[:, 0]
        # x = self.to_latent(x)
        # return self.mlp_head(x)

class ViT_fu(nn.Module):
    def __init__(self, *, image_size=256, image_patch_size=16, frames=16, frame_patch_size=2, num_classes=4, dim=1024, depth=6, heads=8, mlp_dim=2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 decoder_embed_dim=512, 
            decoder_depth=8, 
            decoder_num_heads=16,
            norm_layer=nn.LayerNorm,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None, 
            drop_rate=0., 
            attn_drop_rate=0.,
            drop_path_rate=0.,
            mask_ratio = 0.75):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        self.patch_dim = image_patch_size
        self.frame_patch_size = frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=image_patch_size, in_chans=1, embed_dim=dim, num_frames=frames, tubelet_size=frame_patch_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embedding.shape[-1], self.num_h, self.num_c, cls_token=True)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_h, self.num_c, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, input):
        input = input
        skip_list = []
        x = self.patch_embed(input)
        # mask = self.patch_embed(rois)
        b, n, _ = x.shape
        x += self.pos_embedding[:, 1:,:]
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # mask = torch.cat((cls_tokens, mask), dim=1)
        for blocks in self.blocks:
            skip_list.append(x)
            x = blocks(x)
        x = self.norm(x)

        return x
    
    def forward(self, input):
        latent = self.forward_encoder(input)
        return latent
        # x = latent.mean(dim = 1) if self.pool == 'mean' else latent[:, 0]
        # x = self.to_latent(x)

        # return self.mlp_head(x)

class FusionModel(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, classes):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim_a + input_dim_b, 512)
        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim_a + input_dim_b, input_dim_a,),
            nn.ReLU(),
            nn.Linear(input_dim_a, 2),  # 两个模型权重
        )

    def forward(self, a, b):
        cls_token_features1, all_tokens_features1 = a[:, 0], a[:, 1:]
        cls_token_features2, all_tokens_features2 = b[:, 0], b[:, 1:]
        fused_pooled = 0.5 * cls_token_features1 + 0.5 * cls_token_features2
        return fused_pooled

class WeightNet(nn.Module):
    def __init__(self, feat_dim1, feat_dim2, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim1 + feat_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 输出两个权重
    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        w = F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)  # 确保和为1
        return w
    
class FusionPipeline(nn.Module):
    def __init__(self, model_a, model_b, fusion_module,num_classes):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.fusion = fusion_module
        self.weight_net = WeightNet(1024, 1024)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 展平成 [B, C]
            nn.Linear(1024, num_classes) # 全连接分类层
        )
    
    def forward(self, x, flag=0):
        feat_a = self.model_a(x)
        feat_b = self.model_b(x)
        w = self.weight_net(feat_a[:, 0], feat_b[:, 0])
        fused_feat = w[:,0:1]*feat_a[:, 0] + w[:,1:2]*feat_b[:, 0]
        logits = self.classifier(fused_feat)   # 分类输出 [B, num_classes]
        if flag:
            return logits, fused_feat.cpu().numpy(), w
        else:
            return logits

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x1, x2):
        """
        x1: 来自第一个 Transformer 输出的特征 (seq_len, batch_size, embed_dim)
        x2: 来自第二个 Transformer 输出的特征 (seq_len, batch_size, embed_dim)
        
        返回: 融合后的特征 (seq_len, batch_size, embed_dim)
        """
        # 计算交叉注意力：x1 为查询，x2 为键和值
        attn_output, _ = self.attn(x1, x2, x2)  # 注意这里是用 x2 作为键和值
        return attn_output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.depth = embed_dim // num_heads  # 每个注意力头的维度
        
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        
        self.output_dense = nn.Linear(embed_dim, embed_dim)
    
    def split_heads(self, x, batch_size):
        """将最后一个维度（embedding维度）分成多个头"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)  # [batch_size, seq_len, num_heads, depth]
        return x.permute(0, 2, 1, 3)  # 转换为 [batch_size, num_heads, seq_len, depth]

    def forward(self, x):
        batch_size = x.size(0)
        
        # 获取Q, K, V
        query = self.query_dense(x)  # [batch_size, seq_len, embed_dim]
        key = self.key_dense(x)  # [batch_size, seq_len, embed_dim]
        value = self.value_dense(x)  # [batch_size, seq_len, embed_dim]

        # 将Q, K, V分成多个头
        query = self.split_heads(query, batch_size)  # [batch_size, num_heads, seq_len, depth]
        key = self.split_heads(key, batch_size)  # [batch_size, num_heads, seq_len, depth]
        value = self.split_heads(value, batch_size)  # [batch_size, num_heads, seq_len, depth]

        # 计算注意力得分
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        # 使用Softmax计算注意力权重
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 加权求和
        output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, depth]
        
        # 将多个头拼接
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, depth]
        output = output.view(batch_size, -1, self.embed_dim)  # [batch_size, seq_len, embed_dim]
        
        # 线性变换得到最终输出
        output = self.output_dense(output)
        return output
        
# class FusionModel(nn.Module):
#     def __init__(self, input_dim_a, input_dim_b, classes, num_layers):
#         super(FusionModel, self).__init__()
    
#         self.Cross_block = nn.ModuleList()
#         for cross_layer in range(num_layers):  # VSS Block
#             clayer = CrossAttentionFusion(
#                 embed_dim=1024,
#                 num_heads=8,
#             )
#             self.Cross_block.append(clayer)
        
#         self.layers_up = nn.ModuleList()
#         for i_layer in range(num_layers):
#             layer = SelfAttention(
#                 embed_dim=1024,
#                 num_heads=8,
#             )
#             self.layers_up.append(layer)

#         self.fc1 = nn.Linear(input_dim_a, 512)  # 全连接层 1
#         self.fc2 = nn.Linear(512, classes)  # 全连接层 2
#     def Fusion_network(self, skip_list1, skip_list2):
#         fused_skip_list = []
#         for Cross_layer, skip1, skip2 in zip(self.Cross_block, skip_list1, skip_list2):
#             fused_skip = Cross_layer(skip1, skip2)
#             fused_skip_list.append(fused_skip)
#         return fused_skip_list

#     def forward_features_up(self, x, skip_list):
#         for inx, layer_up in enumerate(self.layers_up):
#             if inx == 0:
#                 x = layer_up(x)
#             else:
#                 x = layer_up(x + skip_list[-inx])

#         return x

#     def forward_final(self, x):
#         # x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x[:,0]))
#         x = self.fc2(x)
#         return x

#     def forward(self, x_a, x_b):
#         x1, skip_list1 = x_a[0], x_a[1]
#         x2, skip_list2 = x_b[0], x_b[1]
#         x = x1 + x2
#         skip_list = self.Fusion_network(skip_list1,skip_list2)
#         x = self.forward_features_up(x, skip_list)
#         x = self.forward_final(x)
  
#         return x
                                                                                     
# class FusionModel(nn.Module):
#     def __init__(self, input_dim_a, input_dim_b, classes):
#         super(FusionModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim_a + input_dim_b, 256)
#         self.fc2 = nn.Linear(256, classes)  
#     def forward(self, x_a, x_b):
#         fused_features = torch.cat((x_a, x_b), dim=1)  
#         x = torch.relu(self.fc1(fused_features))  
#         x = self.fc2(x) 
#         return x

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint
if __name__ == '__main__':
    model = ViT(
    image_size = 256,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1)

    video = torch.randn(4, 2, 16, 256, 256) # (batch, channels, frames, height, width)

    preds = model(video) # (4, 1000)
    print(preds.size())