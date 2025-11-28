from net.decoder import create_decoder
from net.encoder import create_encoder
from net.channel import Channel
from net.modules import DigitalMapper, DigitalDemodulator, BitToFeatureMapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# [1] Classification Head Definition
class ClassificationHead(nn.Module):
    """
    Simple Classification Head for CrossEntropy Loss.
    Input: Feature Map (B, L_spatial, C)
    Process: Global Average Pooling -> Linear -> Class Logits
    """
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        # AdaptiveAvgPool1d operates on (B, C, L), so input needs transpose.
        # Global Average Pooling removes spatial dimension L.
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (B, L_spatial, C)
        # Permute to (B, C, L_spatial) for pooling
        x = x.transpose(1, 2)
        x = self.gap(x).flatten(1) # (B, C)
        return self.fc(x)

class E2E_SwinJSCC(nn.Module):
    """
    End-to-End SwinJSCC with Digital Modulation and Progressive Transmission.
    
    This model integrates a Swin Transformer-based Encoder/Decoder with a learnable 
    Digital Modulation Mapper. It supports progressive transmission where features 
    are transmitted in packets, allowing the receiver to reconstruct the image 
    incrementally.

    Args Explanation:
    -----------------
    args.C (int): 
        - Bits per Spatial Token. 
        - Determines how many bits are allocated per single feature token in the spatial grid.
        - Example: 16 -> Each spatial token (pixel in feature map) is converted to 16 bits.

    args.M (int):
        - Modulation Order (e.g., 16 for 16QAM, 64 for 64QAM).
        - Determines the number of bits per symbol (Q = log2(M)).
        - Example: M=16 -> Q=4 bits per symbol.

    config.encoder_kwargs['embed_dims'][-1]:
        - Total Channel Number (C_total).
        - Total number of feature channels extracted by the Encoder (e.g., 192).
        - This determines the depth of the feature map.

    self.channels_per_packet (Hardcoded to 2):
        - Number of feature channels transmitted in a single packet.
        - The total number of transmission steps (L) is C_total / channels_per_packet.
    """
    def __init__(self, args, config):
        super(E2E_SwinJSCC, self).__init__()
        self.config = config
        
        # [Config Update] Pass num_classes to Decoder config
        # Determine number of classes based on dataset (e.g., CIFAR10=10, ImageNet=1000)
        num_classes = 1000 # ImageNet default
        if args.trainset == 'CIFAR10': num_classes = 10
        # Update config dictionary (add key if missing)
        config.decoder_kwargs['num_classes'] = num_classes

        # 1. Initialization based on Config & Logger
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        
        # Create Encoder and Decoder from configs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        
        # Log network configuration
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info(f"Encoder: {encoder_kwargs}")
            config.logger.info(f"Decoder: {decoder_kwargs}")

        # [3] Resolution & Downsample Factor Calculation
        self.H = self.W = 0 # Initialize for resolution change detection
        
        # Calculate Downsample Factor from Swin Config
        # Formula: Patch_Size * 2^(Num_Stages - 1)
        depths = encoder_kwargs['depths']
        patch_size = encoder_kwargs.get('patch_size', 2)
        downsample_factor = patch_size * (2 ** (len(depths) - 1))
        
        # Store log2 value for Decoder resolution update (e.g., 16 -> 4)
        self.downsample_ratio = int(np.log2(downsample_factor))
        
        # Calculate Feature Map Shape (Fixed for Mapper Initialization)
        # Assumes input image size from config is fixed for training
        img_H, img_W = config.image_dims[1], config.image_dims[2]
        feat_H = img_H // downsample_factor
        feat_W = img_W // downsample_factor
        feature_map_shape = (feat_H, feat_W)
        
        if config.logger is not None:
            config.logger.info(f"Feature Map Shape: {feature_map_shape}, Downsample Factor: {downsample_factor}")

        # Dimensions & Muxing Setup
        self.total_channels = encoder_kwargs['embed_dims'][-1] # C_total (e.g., 192)
        self.channels_per_packet = 2 # Fixed Muxing Ratio: 2 channels per packet
        
        # Calculate Total Transmission Steps (L)
        assert self.total_channels % self.channels_per_packet == 0, \
            "Total channels must be divisible by 2."
        self.L = self.total_channels // self.channels_per_packet
        
        self.bits_per_token = int(args.C) # User defined bits per spatial token
        
        # [Modules] Initialize Digital Mapper, Demodulator, and Inverse Mapper
        # Mapper: Features -> Bits -> Symbols
        self.mapper = DigitalMapper(
            in_dim=self.total_channels, 
            feature_map_shape=(feat_H, feat_W),
            M=args.M, 
            target_bits_per_channel=self.bits_per_token
        )
        
        # Demodulator: Symbols -> Probabilities (Soft Bits)
        self.demodulator = DigitalDemodulator(M=args.M)
        
        # Inverse Mapper: Probabilities -> Features
        seq_len = feat_H * feat_W
        self.inv_mapper = BitToFeatureMapper(
            target_bits_per_channel=self.bits_per_token,
            out_seq_len=seq_len,
            out_dim=self.total_channels 
        )
        
        # Channel and Losses
        self.channel = Channel(args, config)
        self.distortion_loss = nn.MSELoss() 
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input_image, snr, label=None):
        """
        Forward pass of the E2E system.
        
        Steps:
        1. Encoder extracts features from input image.
        2. Mapper converts features to digital symbols.
        3. Symbols pass through wireless channel (with noise).
        4. Receiver demodulates symbols to bit probabilities.
        5. Inverse Mapper reconstructs features from probabilities.
        6. Progressive Loop: Reconstructs image packet-by-packet.
        """
        B, _, H, W = input_image.shape
        
        # [3] Dynamic Resolution Update (Swin Transformer Requirement)
        # Update masks if input resolution changes during inference
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            # Decoder receives downsampled resolution
            ds_H = H // (2 ** self.downsample_ratio)
            ds_W = W // (2 ** self.downsample_ratio)
            self.decoder.update_resolution(ds_H, ds_W)
            self.H = H
            self.W = W

        # 1. Transmitter
        x_feat_all = self.encoder(input_image, model='E2E')
        symbols_all, p_bits_all = self.mapper(x_feat_all)
        
        # 2. Channel Processing (Vectorized)
        B, C_total, S_sub, _ = symbols_all.shape
        
        # Mapper output: (B, C, S, 2) -> Last dim is [Real, Imag]
        # We need to group all Reals and all Imags separately for the channel module.
        # Permute to (B, 2, C, S)
        symbols_permuted = symbols_all.permute(0, 3, 1, 2).contiguous()
        
        # Flatten for Channel: (B, -1) -> [Real_Block, Imag_Block]
        symbols_flat = symbols_permuted.view(B, -1)
        
        # Wireless Channel Pass (One-shot)
        noisy_flat = self.channel(symbols_flat, snr, avg_pwr=True)
        
        # Reshape back: (B, 2, -1) -> Split Real and Imag
        noisy_separated = noisy_flat.view(B, 2, -1)
        
        # Permute back to (B, -1, 2) for Demodulator
        # (B, Total_Symbols, 2) where Total_Symbols = Total_C * S_sub
        rx_input = noisy_separated.permute(0, 2, 1).contiguous()
        
        # Sigma Calculation for LLR
        sigma = torch.tensor(np.sqrt(1.0 / (2 * 10**(snr/10))), device=input_image.device).float()
        
        # Shared Constellation from Encoder
        constellation = self.mapper.get_normalized_constellation()
        bin_map = self.mapper.bin_map
        
        # Demodulation (Probabilities): (B, Total_Symbols, Q)
        # Output is probability P(b=1|y) in range [0, 1]
        prob_flat = self.demodulator(rx_input, constellation, bin_map, sigma)
        
        # -------------------------------------------------------
        # 3. Inverse Mapping (UEP Decoding 적용) - [Dynamic Version]
        # -------------------------------------------------------
        
        # [수정] 하드코딩 제거하고 멤버 변수에서 가져오기
        Q = self.mapper.bits_per_symbol          # 예: 4 (16QAM)
        C_original = self.total_channels         # 예: 256 (또는 320, 192 등)
        C_combined = C_original // 2             # 예: 128 (UEP로 합쳐진 채널 수)
        
        # S_uep: 합쳐진 한 쌍(Pair)이 차지하는 심볼 개수
        # (채널당 비트 * 2개 채널) / 심볼당 비트
        # 예: (16 * 2) // 4 = 8 심볼
        # [주의] bits_per_token * 2가 Q로 나누어 떨어져야 함 (보통 16, 4라 괜찮음)
        S_uep = (self.bits_per_token * 2) // Q 

        # 1. (B, C_combined, S, Q) 형태로 Reshape
        #    view의 -1 부분은 자동으로 S_uep 크기가 됩니다.
        prob_grouped = prob_flat.view(B, C_combined, -1, Q)
        
        # 2. Q(비트)를 MSB와 LSB로 분리
        #    절반은 짝수 채널(MSB), 절반은 홀수 채널(LSB)
        half_Q = Q // 2
        prob_msb = prob_grouped[:, :, :, :half_Q] # (B, C/2, S, Q/2)
        prob_lsb = prob_grouped[:, :, :, half_Q:] # (B, C/2, S, Q/2)
        
        # 3. 채널 다시 인터리빙 (짝수, 홀수 순서대로 끼워넣기)
        #    dim=2에 쌓으면 (B, C_combined, 2, S, Q/2)가 됨
        prob_interleaved = torch.stack([prob_msb, prob_lsb], dim=2)
        
        # 4. Flatten하여 원래 채널 구조 및 비트 시퀀스로 복원
        #    Target Shape: (B, C_original, bits_per_token)
        #    예: (B, 256, 16)
        prob_reshaped = prob_interleaved.view(B, C_original, self.bits_per_token)
        
        # (B, Spatial_L, Total_C) - All features recovered at once
        recovered_features_all = self.inv_mapper(prob_reshaped)
        
        # -------------------------------------------------------
        # 4. Progressive Loop (Reconstruction & Task)
        # -------------------------------------------------------
        results = {
            'mse': [],
            'ce': [],
            'recon_img': []
        }
        
        # Feature Accumulator (Zero Initialized)
        feature_buffer = torch.zeros_like(x_feat_all)
        
        for l in range(self.L):
            c_start = l * self.channels_per_packet
            c_end = (l + 1) * self.channels_per_packet
            
            # Slice current packet's features from recovered buffer
            packet_features = recovered_features_all[:, :, c_start:c_end]
            
            # Accumulate features (fill in the buffer)
            feature_buffer[:, :, c_start:c_end] = packet_features
            
            # --- Decoder Input Preparation ---
            # Decoding (Reconstruction & Classification)
            decoder_input = feature_buffer.clone()
            recon_img, logits = self.decoder(decoder_input)

            # Loss Calculation
            loss_mse = self.distortion_loss(recon_img, input_image)
            
            loss_ce = torch.tensor(0.0, device=input_image.device)
            if label is not None and logits is not None:
                loss_ce = self.ce_loss(logits, label)
            
            results['mse'].append(loss_mse)
            results['ce'].append(loss_ce)
            results['recon_img'].append(recon_img)

        return results, p_bits_all