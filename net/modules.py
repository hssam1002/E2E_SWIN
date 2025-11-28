import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(
                0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        H, W = input_resolution
        self.patch_merging1 = PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_merging2 = PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H//2, W//2)
        return x


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # print(x.shape)
        x = x.flatten(2).permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchReverseMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H*2, W*2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class FeatureToBitMapper(nn.Module):
    """
    [User Request Implementation]
    Input: (B, L, C)
    Process: 
      1. Transpose & Reshape: (B, L, C) -> (B, C, L) -> (B*C, 1, H, W)
      2. Conv2d Layers: Extract features per channel
      3. Flatten & Linear: Map to target bit sequence length
      4. Reshape: (B*C, Target) -> (B, C, Target)
    """
    def __init__(self, src_shape, target_bits_per_channel):
        super().__init__()
        self.H_feat, self.W_feat = src_shape
        self.target_bits = target_bits_per_channel
        
        mid_channels = 4
        
        # 1. Conv2d Layers
        # Input: (B*C, 1, H, W)
        self.conv1 = nn.Conv2d(1, mid_channels, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, padding = 1)
        
        # 2. Projection Layer
        # Flatten size: mid_channels * H * W
        flat_dim = mid_channels * self.H_feat * self.W_feat
        self.proj = nn.Linear(flat_dim, target_bits_per_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # Shape Check (Optional but recommended)
        if L != self.H_feat * self.W_feat:
            # Fallback logic for variable resolution
            size = int(np.sqrt(L))
            H, W = size, size
        else:
            H, W = self.H_feat, self.W_feat
            
        # 1. Transpose (B, L, C) -> (B, C, L) -> (B*C, 1, H, W)
        x = x.transpose(1, 2).reshape(B * C, 1, H, W)
        
        # 2. Conv2d -> ReLU -> Conv2d
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out) # (B*C, mid, H, W)
        
        # 3. Flatten & Projection
        out = out.flatten(1)  # (B*C, mid*H*W)
        out = self.proj(out)  # (B*C, target)
        
        # 4. Sigmoid
        out = self.sigmoid(out)
        
        # 5. Reshape back to (B, C, target)
        out = out.view(B, C, self.target_bits)
        
        return out

# Constellation 을 매번 update할지, 아니면 epoch가 끝나고 update를 할지에 대해서는 생각해보자
class DigitalMapper(nn.Module):
    def __init__(self, in_dim, feature_map_shape, M = 16, target_bits_per_channel=16, temp=1.0):
        super().__init__()
        self.M = M
        self.bits_per_symbol = int(np.log2(M)) # Q
        self.temp = temp
        self.target_bits_per_channel = target_bits_per_channel
        
        # target_bits_per_channel: 각 채널(Feature) 하나가 변환될 비트 수 (Target)
        self.bit_mapper = FeatureToBitMapper(feature_map_shape, target_bits_per_channel) # (B, C, target)
        
        # Constellation & Binary Map (Trainable/Fixed)
        self.constellation_param = nn.Parameter(self._build_initial_constellation(M))
        self.register_buffer('bin_map', self._build_binary_map(M, self.bits_per_symbol))

    def _build_initial_constellation(self, M):
        s = int(np.sqrt(M))
        x, y = np.meshgrid(np.linspace(-1,1,s), np.linspace(-1,1,s))
        points = (x + 1j*y).flatten()
        points = torch.from_numpy(points).cfloat()
        return torch.stack([points.real, points.imag], dim=1).float()

    def _build_binary_map(self, M, k):
        bin_map = np.zeros((M, k), dtype=np.float32)
        for i in range(M):
            bin_map[i, :] = [int(b) for b in format(i, f'0{k}b')]
        return torch.from_numpy(bin_map)

    def get_normalized_constellation(self):
        c = self.constellation_param
        c = c - c.mean(0, keepdim=True)
        return c / c.pow(2).sum(1).mean().sqrt()

    def forward(self, x):
        # 1. Generate Bits: (B, C, Target)
        p_bits = self.bit_mapper(x) # (B, C, target)

        # 2. No Padding (Assuming Target % Q == 0)
        Q = self.bits_per_symbol

        # 3. Modulation coding
        B, C, T = p_bits.shape
        half_Q = self.bits_per_symbol // 2

        # 1. 채널을 짝수(2l)와 홀수(2l-1)로 분리 (논문에 맞춰 순서 확인 필요)
        # 논문: 2l-1(MSB), 2l(LSB) -> 코드 인덱스로는 짝수(0,2..)가 MSB, 홀수(1,3..)가 LSB라고 가정
        p_bits_msb = p_bits[:, 0::2, :] # (B, C/2, T)
        p_bits_lsb = p_bits[:, 1::2, :] # (B, C/2, T)

        # 2. 비트 차원(T)에서 심볼 단위로 쪼개기 위해 Reshape
        # (B, C/2, num_symbols, half_Q)
        p_msb_grouped = p_bits_msb.view(B, C//2, -1, half_Q)
        p_lsb_grouped = p_bits_lsb.view(B, C//2, -1, half_Q)

        # 3. MSB와 LSB 결합 (Concatenate) -> (B, C/2, num_symbols, Q)
        # 이렇게 해야 하나의 심볼에 두 채널의 정보가 섞입니다.
        p_grouped = torch.cat([p_msb_grouped, p_lsb_grouped], dim=-1)
        
        # 4. Symbol Prob
        # p_exp:   (B, C, S, 1, Q) - 예측한 비트 확률
        # bin_exp: (1, 1, 1, M, Q) - Constellation 포인트별 비트 패턴 (d_k)
        p_exp = p_grouped.unsqueeze(3)
        bin_exp = self.bin_map.view(1, 1, 1, self.M, Q)
        
        # log pi_k = sum( d_kj * log(p_j) + (1-d_kj) * log(1-p_j) )
        # log_probs: (B, C, S, M)
        term1 = bin_exp * torch.log(p_exp + 1e-10)           # d=1인 경우
        term2 = (1 - bin_exp) * torch.log(1 - p_exp + 1e-10) # d=0인 경우
        log_probs = torch.sum(term1 + term2, dim=-1)
        
        # 5. Gumbel Softmax & Modulation
        is_training = self.training
        soft_onehot = F.gumbel_softmax(log_probs, tau=self.temp, hard=not is_training, dim=-1) # (B, C, S, M) -> v_{l,n} (Soft)
        complex_symbols = torch.matmul(soft_onehot, self.get_normalized_constellation())
        
        return complex_symbols, p_bits
    
class DigitalDemodulator(nn.Module):
    """
    Calculates P(b=1|y) using Sigmoid(LLR).
    Input: 
      - received: (B, S, 2) 
      - constellation: (M, 2)
      - bin_map: (M, Q)
      - sigma: Noise Standard Deviation
    """
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.bits_per_symbol = int(np.log2(M)) # Q

    def forward(self, received, constellation, bin_map, sigma=1.0):
            # 1. Distance
            dist_sq = (received.unsqueeze(2) - constellation.view(1, 1, self.M, 2)).pow(2).sum(-1)
            
            # 2. Exponents (Log Prob)
            sigma_sq = torch.square(sigma) if torch.is_tensor(sigma) else sigma**2
            snr_coeff = 1.0 / (2 * sigma_sq + 1e-9)
            exponents = -snr_coeff * dist_sq

            # 3. Bit Grouping
            bin_map_exp = bin_map.view(1, 1, self.M, self.bits_per_symbol)
            exponents_exp = exponents.unsqueeze(-1) # (B, S, M, 1)
            inf_mask = torch.tensor(-1e9, device=received.device)
            
            # C_1: bit 1
            logits_1 = torch.where(bin_map_exp == 1, exponents_exp, inf_mask)
            # C_0: bit 0
            logits_0 = torch.where(bin_map_exp == 0, exponents_exp, inf_mask)
            
            # 4. LLR
            # ln( Sum_{c in C1} P(y|c) ), ln( Sum_{c in C0} P(y|c) )
            log_sum_1 = torch.logsumexp(logits_1, dim=2) # (B, S, Q)
            log_sum_0 = torch.logsumexp(logits_0, dim=2) # (B, S, Q)
            
            # 5. Sigmoid (Prob P(1|y))
            return torch.sigmoid(log_sum_1 - log_sum_0)

class BitToFeatureMapper(nn.Module):
    """
    Inverse of FeatureToBitMapper.
    Input: (B, C, Target) -> Output: (B, L, C)
    """
    def __init__(self, target_bits_per_channel, out_seq_len, out_dim):
        super().__init__()      
        self.target_bits = target_bits_per_channel
        self.out_seq_len = out_seq_len
        self.size = int(np.sqrt(out_seq_len))
        
        # 1. Linear Projection: Target Bits -> Intermediate Spatial
        mid_channels = 4
        flat_dim = mid_channels * self.size * self.size
        self.proj = nn.Linear(target_bits_per_channel, flat_dim)
        
        # 2. Transpose Conv (or just Conv) to mix features
        # (B*C, mid, H, W) -> (B*C, 1, H, W)
        self.deconv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 3, 1, 1)
        )
        
    def forward(self, x):
        # x: (B, C, Target)
        B, C, T = x.shape
        # 1. Flatten
        x = x.view(B*C, -1)
        # 2. Linear projection
        x = self.proj(x)

        # 3. Reshape for Conv
        x = x.view(B*C, -1, self.size, self.size)

        # 4. Conv
        x = self.deconv(x)
        # 5. Reshape back (B*C, 1, H, W) -> (B, C, L) -> (B, L, C)
        return x.view(B, C, self.out_seq_len).transpose(1, 2)