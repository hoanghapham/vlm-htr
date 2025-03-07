import numpy as np


def gen_split_indices(total_samples: int, seed: int = 42):

    np.random.seed(seed)

    train_indices = np.random.choice(range(total_samples), size=int(0.7 * total_samples), replace=False)
    
    val_indices = np.random.choice(
        [idx for idx in range(total_samples) if idx not in train_indices], 
        size=int(0.15 * total_samples), replace=False
    )
    
    test_inidices = np.random.choice(
        [idx for idx in range(total_samples) if idx not in np.concatenate([train_indices, val_indices])], 
        size = total_samples - len(train_indices) - len(val_indices),
        replace=False
    )

    return train_indices, val_indices, test_inidices