import torch.optim as optim
from net.network import E2E_SwinJSCC
from data.datasets import get_loader
from utils import *
from loss.distortion import Distortion, MS_SSIM
from torch.utils.checkpoint import checkpoint
import os
import torch
import torch.nn as nn
import argparse
import time
import numpy as np
from datetime import datetime

# --- 1. Arguments Configuration ---
# =============================================================================
# [User Guide & Arguments Explanation]
#
# 1. args.C (Bits per Spatial Token) [Default: 12]:
#    - Meaning: Number of bits allocated per single feature token.
#    - Changed default to 12 as requested.
#
# 2. args.beta (Beta for Info-Max) [Default: 1.0]:
#    - Meaning: Balancing weight for MSE in the Information formula.
#    - Formula: INFO = -CrossEntropy - (beta * MSE)
#    - Control: Higher beta emphasizes Reconstruction (MSE), Lower beta emphasizes Classification.
#
# 3. args.sample_num (Monte-Carlo Sampling R) [Default: 1]:
#    - Meaning: Number of VAE samples (R) to approximate the expectation.
#    - Implementation: Input batch is repeated R times to calculate average loss.
# =============================================================================

# --- Arguments Configuration ---
parser = argparse.ArgumentParser(description='E2E SwinJSCC for Classification & Recon')

# [Mode]
parser.add_argument('--training', action='store_true', help='Flag to start training.')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs).')

# [Dataset Settings]
parser.add_argument('--trainset', type=str, default='CIFAR10', choices=['CIFAR10', 'ImageNet'], 
                    help='Dataset for training.')
# Note: Test set is automatically determined based on trainset (Validation set).

# [Communication & Model Settings]
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], 
                    help='Wireless channel model.')
parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'], 
                    help='Size of Swin Transformer backbone.')

# [Key Parameters]
parser.add_argument('--C', type=str, default='12', 
                    help='Bits per spatial token (Default: 12).')
parser.add_argument('--M', type=int, default=16, 
                    help='QAM Modulation Order (16 or 64).')
parser.add_argument('--multiple-snr', type=str, default='10', 
                    help='Training SNR (dB). e.g., "10" or "0,10,20"')

# [Info-Max Parameters]
parser.add_argument('--beta', type=float, default=1.0, 
                    help='Weight for MSE term in Info calculation (Info = -CE - beta*MSE).')
parser.add_argument('--sample_num', type=int, default=1, 
                    help='Number of Monte-Carlo samples (R) for expectation approximation.')

args = parser.parse_args()

# --- Global Variables for ALM ---
rho = 1.0       
gamma = 1.2     
lambda_l = None 

# Parse SNR list
if isinstance(args.multiple_snr, str): snr_list = [int(s) for s in args.multiple_snr.split(',')]
else: snr_list = [int(args.multiple_snr)]

# --- 2. Configuration Class ---
class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = False

    # [Path Configuration]
    # Default server paths
    BASE_ROOT = "/data4/hongsik/E2E_SWIN"
    
    # [Path Settings]
    base_save_path = "/data4/hongsik/E2E_SWIN"
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = f'{base_save_path}/history/{filename}'
    log = workdir + f'/Log_{filename}.log'
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    normalize = False
    learning_rate = 0.0001
    tot_epoch = 1000000

    print_step = 1

    # [Data Path Resolution]
    # Use arguments if provided, else use default server paths
    if args.trainset == 'CIFAR10':
        save_model_freq = 1 # Validate every 5 epochs
        image_dims = (3, 32, 32)
        
        train_data_dir = "/data4/hongsik/E2E_SWIN/data/CIFAR10/"
        test_data_dir = "/data4/hongsik/E2E_SWIN/data/CIFAR10/"
        
        batch_size = 32 * torch.cuda.device_count()
        downsample = 2 
        
        common_kwargs = dict(
            img_size=(32, 32), 
            patch_size=2, 
            in_chans=3, 
            window_size=2, 
            mlp_ratio=4., 
            qkv_bias=True, qk_scale=None, 
            norm_layer=nn.LayerNorm, 
            patch_norm=True,
            model='E2E',  # 필수 인자 추가
            C = None        # 필수 인자 추가 (Encoder 출력 채널 유지)
        )
        #encoder_kwargs = dict(embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], **common_kwargs)
        #decoder_kwargs = dict(embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], **common_kwargs)    
        encoder_kwargs = dict(embed_dims=[64, 128], depths=[2, 4], num_heads=[4, 8], **common_kwargs)
        decoder_kwargs = dict(embed_dims=[128, 64], depths=[4, 2], num_heads=[8, 4], **common_kwargs)   
    elif args.trainset == 'ImageNet':
        save_model_freq = 5
        image_dims = (3, 256, 256)

        train_data_dir = "/data4/hongsik/data/ImageNet/train" 
        test_data_dir = "/data4/hongsik/data/ImageNet/val"
        
        batch_size = 32 * torch.cuda.device_count() 
        downsample = 4 
        
        common_kwargs = dict(
            img_size=(256, 256), 
            patch_size=2, 
            in_chans=3, 
            window_size=8, 
            mlp_ratio=4., 
            qkv_bias=True, 
            qk_scale=None, 
            norm_layer=nn.LayerNorm, 
            patch_norm=True,
            model='E2E',  # 필수 인자 추가
            C = None      # 필수 인자 추가
        )

        if args.model_size == 'small':
            encoder_kwargs = dict(embed_dims=[64, 128, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], **common_kwargs)
            decoder_kwargs = dict(embed_dims=[320, 256, 128, 64], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], **common_kwargs)
        elif args.model_size == 'base':
            encoder_kwargs = dict(embed_dims=[64, 128, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], **common_kwargs)
            decoder_kwargs = dict(embed_dims=[320, 256, 128, 64], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], **common_kwargs)
        elif args.model_size == 'large':
            encoder_kwargs = dict(embed_dims=[64, 128, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], **common_kwargs)
            decoder_kwargs = dict(embed_dims=[320, 256, 128, 64], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], **common_kwargs)

# Loss Setup: Reconstruction metric is implicit (MSE), but MS-SSIM is useful for logging
CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()

# --- 3. Training Loop ---
def train_one_epoch(args):
    """
    Train one epoch with Monte-Carlo Sampling (R) and Beta-Weighted Info.
    Formula: Info = E[log p(y|z) - beta * ||x - x_hat||^2]
             Info ~= -CrossEntropy - beta * MSE
    """
    global rho, lambda_l, global_step, optimizer_backbone, optimizer_const
    
    net.train()
    elapsed, losses = AverageMeter(), AverageMeter()
    psnrs = AverageMeter()
    ce_losses = AverageMeter()
    
    # Retrieve L (Total Steps) from Network
    if isinstance(net, nn.DataParallel): L = net.module.L
    else: L = net.L
        
    # Initialize Dual Variables
    if lambda_l is None: lambda_l = torch.zeros(L).cuda()
    alpha = torch.full((L,), 1.0 / L).cuda()

    # Reset Accumulators for Epoch-wise updates
    optimizer_const.zero_grad()
    h_accumulator = torch.zeros(L).cuda()
    num_batches = 0

    for batch_idx, (input_img, label) in enumerate(train_loader):
        torch.cuda.empty_cache()
        start_time = time.time()
        global_step += 1
        num_batches += 1
        
        input_img, label = input_img.cuda(), label.cuda()
        current_snr = np.random.choice(snr_list)

        # 1. Forward Pass
        # results = {
        #'mse': [loss_step_1, loss_step_2, ..., loss_step_L],       # 각 단계의 MSE Loss (Tensor)
        #'ce':  [loss_step_1, loss_step_2, ..., loss_step_L],       # 각 단계의 CrossEntropy Loss (Tensor)
        #'recon_img': [img_step_1, img_step_2, ..., img_step_L]     # 각 단계의 복원 이미지 (Tensor)
        #}
        results, _ = net(input_img, current_snr, label)
        
        # 2. Extract and Average Losses (for Multi-GPU)
        mse_list = [m.mean() for m in results['mse']]
        ce_list = [c.mean() for c in results['ce']]

        # 3. ALM Constraints Calculation
        # Info = -CrossEntropy - beta * MSE
        # (Maximizing Info -> Minimizing -(Info))
        # Note: CE is positive (NLL), so -CE is Log-Likelihood (negative).
        # We want to MAXIMIZE (LogLikelihood - beta*MSE).
        
        info_vals = []
        for c, m in zip(ce_list, mse_list):
            info_val = -c - (args.beta * m) # Info formula
            info_vals.append(info_val)
            
        info_stack = torch.stack(info_vals) # (L,)
        info_L = info_stack[-1]
        
        # (Constraints calculation logic remains same)
        denom = info_L.abs() + 1e-9
        h_vec = []
        for l in range(L):
            curr_info = info_stack[l]
            prev_info = info_stack[l-1] if l > 0 else torch.tensor(0.0).cuda()
            h_val = ((curr_info - prev_info) / denom) - alpha[l]
            h_vec.append(h_val)
        h_stack = torch.stack(h_vec)
        
        h_accumulator += h_stack.detach()
        
        # 4. Total Loss Combination
        # Loss = -Info_L + Penalty + Lagrange
        loss_main = -info_L 
        
        penalty = (rho / 2) * torch.sum(h_stack ** 2)
        lagrange = torch.sum(lambda_l * h_stack)
        
        total_loss = loss_main + penalty + lagrange
        
        # 5. Optimization (Backbone Only)
        # Backbone is updated every batch.
        # Constellation gradients are accumulated.
        optimizer_backbone.zero_grad()
        total_loss.backward() 
        optimizer_backbone.step() 
        
        # Logging
        elapsed.update(time.time() - start_time)
        losses.update(total_loss.item())
        ce_losses.update(ce_list[-1].item())
        
        if mse_list[-1].item() > 0:
            cur_psnr = 10 * np.log10(255.**2 / mse_list[-1].item())
            psnrs.update(cur_psnr)

        if (global_step % config.print_step) == 0:
            log = (f'Step {global_step} | Total Loss {losses.val:.4f} | '
                   f'CE {ce_losses.val:.4f} | PSNR {psnrs.val:.2f} | Rho {rho:.2f}')
            logger.info(log)

    # --- End of Epoch Updates ---
    # 1. Update Constellation (Using accumulated average gradients)
    for param in optimizer_const.param_groups[0]['params']:
        if param.grad is not None:
            param.grad /= num_batches
    optimizer_const.step()
    
    # 2. Update Dual Variables (Lambda) & Penalty (Rho)
    h_avg = h_accumulator / num_batches
    with torch.no_grad():
        lambda_l += rho * h_avg
        rho = min(rho * gamma, 100.0)

    logger.info(f"Epoch End: Rho updated to {rho:.2f}")

# --- 4. Validation / Test Function ---
def validate(loader):
    config.isTrain = False
    net.eval()
    
    psnrs = AverageMeter()
    ce_losses = AverageMeter()
    
    results_psnr = np.zeros(len(snr_list))
    results_ce = np.zeros(len(snr_list))
    
    for i, SNR in enumerate(snr_list):
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, list) or isinstance(batch, tuple):
                    input_img, label = batch[0], batch[1]
                else:
                    input_img = batch
                    label = None
                
                input_img = input_img.cuda()
                if label is not None: label = label.cuda()
                
                # Forward Pass
                results, _ = net(input_img, SNR, label)
                
                # Metrics (Final Step)
                final_mse = results['mse'][-1]
                final_ce = results['ce'][-1]
                
                if final_mse.ndim > 0: final_mse = final_mse.mean()
                if final_ce.ndim > 0: final_ce = final_ce.mean()
                
                # Reconstruction Metric
                if final_mse.item() > 0:
                    cur_psnr = 10 * np.log10(255.**2 / final_mse.item())
                    psnrs.update(cur_psnr)
                
                # Classification Metric
                if label is not None:
                    ce_losses.update(final_ce.item())
        
        results_psnr[i] = psnrs.avg
        results_ce[i] = ce_losses.avg
        psnrs.clear(); ce_losses.clear()

    print(f"Validation PSNR: {results_psnr.tolist()}")
    print(f"Validation CE Loss: {results_ce.tolist()}")
    
    # Return average PSNR for Early Stopping criterion
    return np.mean(results_psnr)

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    seed_torch(config.seed)
    logger = logger_configuration(config, save_log=True)
    logger.info("Initializing E2E SwinJSCC Framework...")
    logger.info(config.__dict__)
    
    # 1. Model Initialization
    net = E2E_SwinJSCC(args, config).cuda()
    
    # Multi-GPU Setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
        raw_net = net.module
    else:
        raw_net = net

    # 2. Optimizer Setup (Separated)
    # Constellation Parameters
    const_params = [raw_net.mapper.constellation_param]
    const_param_ids = list(map(id, const_params))
    
    # Backbone Parameters
    backbone_params = filter(lambda p: id(p) not in const_param_ids, net.parameters())
    
    optimizer_backbone = optim.Adam(backbone_params, lr=config.learning_rate)
    optimizer_const = optim.Adam(const_params, lr=config.learning_rate) 
    
    # 3. Data Loaders
    # Note: get_loader returns (train, test). We use 'test_loader' as validation during training.
    train_loader, val_loader = get_loader(args, config)
    global_step = 0
    
    # Early Stopping Vars
    best_psnr = -1e9
    epochs_no_improve = 0
    
    # 4. Training Loop
    if args.training:
        for epoch in range(config.tot_epoch):
            logger.info(f"Start Epoch {epoch}")
            train_one_epoch(args)
            
            # Validation
            if (epoch + 1) % config.save_model_freq == 0:
                logger.info("Running Validation...")
                avg_psnr = validate(val_loader)
                
                # Save Best Model Logic
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    epochs_no_improve = 0
                    save_path = config.models + f'/{config.filename}_best.model'
                    
                    # Unwrap DataParallel before saving
                    state_dict = raw_net.state_dict()
                    torch.save(state_dict, save_path)
                    logger.info(f"New Best Model Saved! Avg PSNR: {best_psnr:.4f}")
                else:
                    epochs_no_improve += config.save_model_freq
                    logger.info(f"No improvement. Counter: {epochs_no_improve}/{args.patience}")
                
                # Early Stopping
                if epochs_no_improve >= args.patience:
                    logger.info("Early Stopping Triggered. Training Finished.")
                    break
    else:
        logger.info("Running Test Mode...")
        validate(val_loader)

# python main.py --training --trainset CIFAR10 --C 16 --M 16 --multiple-snr 10