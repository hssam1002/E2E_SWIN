from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import torch
import numpy as np

# Number of workers for data loading
NUM_DATASET_WORKERS = 8

def get_loader(args, config):
    """
    Returns train and test data loaders based on args.trainset.
    Supported: 'CIFAR10', 'ImageNet'
    """
    
    # -----------------------------------------------------------
    # 1. CIFAR-10 Dataset
    # -----------------------------------------------------------
    if args.trainset == 'CIFAR10':
        # Transforms (Standard augmentation for CIFAR)
        # Norm is usually optional for E2E JSCC, but can be used.
        # Here we use basic toTensor (0~1 range) which fits most JSCC works.
        # If config.norm is True, add Normalization.
        
        if config.norm:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])

        # Download=True allows automatic download if not exists
        train_dataset = datasets.CIFAR10(root=config.train_data_dir,
                                         train=True,
                                         transform=transform_train,
                                         download=True)

        test_dataset = datasets.CIFAR10(root=config.test_data_dir,
                                        train=False,
                                        transform=transform_test,
                                        download=True)

    # -----------------------------------------------------------
    # 2. ImageNet-1k Dataset
    # -----------------------------------------------------------
    elif args.trainset == 'ImageNet':
        # Standard ImageNet Transforms
        # Train: RandomResizedCrop(256) -> RandomHorizontalFlip -> ToTensor
        # Test: Resize(256) -> CenterCrop(256) -> ToTensor
        # (Using 256x256 as per config)
        
        img_size = 256
        
        if config.norm:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            normalize = transforms.Normalize(mean, std)
        else:
            normalize = transforms.Lambda(lambda x: x) # Identity

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize(img_size + 32), # Resize slightly larger then crop
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        # ImageFolder requires structure: root/class_x/image.jpg
        train_dataset = datasets.ImageFolder(root=config.train_data_dir, 
                                             transform=transform_train)
        
        test_dataset = datasets.ImageFolder(root=config.test_data_dir, 
                                            transform=transform_test)

    else:
        raise NotImplementedError(f"Dataset {args.trainset} not supported.")

    # -----------------------------------------------------------
    # DataLoader Creation
    # -----------------------------------------------------------
    
    # Deterministic worker seeding
    def worker_init_fn_seed(worker_id):
        seed = 10 + worker_id
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=NUM_DATASET_WORKERS,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn_seed,
                              drop_last=True)

    # Test batch size can be larger
    test_batch_size = config.batch_size * 2 if args.trainset == 'CIFAR10' else config.batch_size
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=NUM_DATASET_WORKERS,
                             pin_memory=True)

    return train_loader, test_loader