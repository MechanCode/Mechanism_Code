"""seed_utils.py

Utility functions for setting random seeds to ensure reproducibility across experiments.

This module provides a unified way to set seeds for:
- Python's random module
- NumPy's random generator
- PyTorch (CPU and CUDA)
- CUDNN deterministic behavior
- DataLoader worker processes

Usage:
    from utils.seed_utils import set_seed, get_worker_init_fn, get_dataloader_generator
    
    # Set global seeds before creating model/data
    set_seed(42)
    
    # Use in DataLoader for consistent data loading
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        generator=get_dataloader_generator(42),
        worker_init_fn=get_worker_init_fn(42),
        ...
    )
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across all random number generators.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU random generator
    - PyTorch CUDA random generator (all devices)
    - CUDNN deterministic behavior (optional)
    
    Parameters:
        seed: The random seed to use.
        deterministic: If True, enables CUDNN deterministic mode for exact
                      reproducibility (may reduce performance). Default True.
    
    Note:
        For exact reproducibility, ensure this function is called before:
        - Creating any models (weight initialization uses random numbers)
        - Creating any data loaders (shuffling uses random numbers)
        - Any operations that use random numbers
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    if deterministic:
        # Enable deterministic algorithms in CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8, use deterministic algorithms globally
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                # Some operations may not have deterministic implementations
                pass
    else:
        # Allow CUDNN to find optimal algorithms (faster but non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_worker_init_fn(base_seed: int):
    """Create a worker initialization function for DataLoader workers.
    
    Each worker process needs its own unique but reproducible seed.
    This function returns a callable that can be passed to DataLoader's
    worker_init_fn parameter.
    
    Parameters:
        base_seed: The base seed from which worker seeds are derived.
    
    Returns:
        A function that initializes worker random states.
    
    Example:
        loader = DataLoader(dataset, num_workers=4, 
                           worker_init_fn=get_worker_init_fn(42))
    """
    def worker_init_fn(worker_id):
        # Each worker gets a unique seed derived from base_seed and worker_id
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        # PyTorch's internal RNG for each worker
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def get_dataloader_generator(seed: int) -> torch.Generator:
    """Create a PyTorch Generator with a fixed seed for DataLoader.
    
    This generator is used by DataLoader for shuffling and sampling.
    Using a fixed generator ensures reproducible data order.
    
    Parameters:
        seed: The random seed for the generator.
    
    Returns:
        A torch.Generator object initialized with the given seed.
    
    Example:
        loader = DataLoader(dataset, shuffle=True,
                           generator=get_dataloader_generator(42))
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
