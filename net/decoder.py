from net.modules import *
import torch
import torch.nn as nn
from datetime import datetime
from net.encoder import SwinTransformerBlock # BasicLayer 등은 encoder에서 가져오거나 여기에 정의

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()
        # Global Average Pooling: (B, C, L) -> (B, C, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.gap(x).flatten(1) # (B, C)
        return self.fc(x)
    
class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class SwinJSCC_Decoder(nn.Module):
    def __init__(self, 
                 img_size,       # Final resolution of the reconstructed image (e.g., (256, 256))
                 embed_dims,     # List of channel dimensions for each stage in the decoder (e.g., [384, 192, 96])
                 depths,         # List of Swin Transformer block depths for each stage (e.g., [2, 2, 6])
                 num_heads,      # List of attention heads for each stage
                 C,              # [Input Channel] The channel dimension of the latent feature coming from the channel (H/16 x W/16 x C)
                 num_classes=0,  # Number of classes for the auxiliary classification task (0 means no classification)
                 window_size=8,  # Window size for Window Multi-head Self Attention (W-MSA)
                 mlp_ratio=4.,   # Expansion ratio for the MLP feed-forward layer
                 qkv_bias=True,  # If True, add a learnable bias to query, key, value
                 qk_scale=None,  # Override default qk scale of head_dim ** -0.5 if set
                 norm_layer=nn.LayerNorm, # Normalization layer used in blocks
                 ape=False,      # (Unused) Absolute Position Embedding
                 patch_norm=True,# If True, add normalization after patch merging
                 bottleneck_dim=16, # (Unused) Legacy parameter for bottleneck dimension
                 model=None,     # Model type identifier (e.g., 'E2E')
                 patch_size=2,   # Patch size for initial embedding (kept for kwargs compatibility)
                 in_chans=3      # Number of output channels for the reconstructed image (usually 3 for RGB)
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))

        # 1. Reconstruction Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())

        # Input Projection (C -> embed_dims[0])
        if C is not None:
            self.head_list = nn.Linear(C, embed_dims[0])
            clf_input_dim = C # C가 있으면 그걸 입력 차원으로 사용
        else:
            self.head_list = nn.Identity()
            clf_input_dim = embed_dims[0] # C가 None이면 embed_dims[0] 사용

        # 2. Classification Head (Optional)
        self.num_classes = num_classes
        if num_classes > 0:
            # Classification은 Latent Feature(C)에서 바로 수행한다고 가정
            self.classifier = ClassificationHead(clf_input_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, L, C) - Latent Feature from Channel
        
        # Task 1: Classification (Latent에서 바로 수행)
        logits = None
        if self.num_classes > 0:
            logits = self.classifier(x)

        # Task 2: Reconstruction
        x_recon = self.head_list(x)
        for layer in self.layers:
            x_recon = layer(x_recon)
            
        # (B, L, 3) -> (B, 3, H, W)
        B, L, Ch = x_recon.shape
        x_recon = x_recon.view(B, self.H, self.W, Ch).permute(0, 3, 1, 2)
        
        return x_recon, logits
    
    def update_resolution(self, H, W):
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        # self.patches_resolution 업데이트 필요
        self.patches_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

def create_decoder(**kwargs):
    model = SwinJSCC_Decoder(**kwargs)
    return model

def build_model(config):
    input_image = torch.ones([1, 1536, 256]).to(config.device)
    model = create_decoder(**config.encoder_kwargs).to(config.device)
    t0 = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            features = model(input_image, SNR=15)
        t1 = datetime.datetime.now()
        delta_t = t1 - t0
        print("Decoding Time per img {}s".format((delta_t.seconds + 1e-6 * delta_t.microseconds) / 100))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))

