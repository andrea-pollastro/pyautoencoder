from __future__ import annotations

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union, Tuple, Callable
import numpy as np
import wget

class DSprite(Dataset):
    """PyTorch dataset wrapper for the dSprites factorized shapes dataset.

    Source:
        Matthey et al., "dSprites: Disentanglement test sprites."
        Original files hosted by DeepMind on GitHub: https://github.com/google-deepmind/dsprites-dataset

    Files:
        - dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

    Contents (loaded into memory on init):
        - X                (torch.Tensor[int8]): Binary images, shape [N, 64, 64].
        - latents_values  (torch.Tensor[float64]): Continuous latent values per sample,
                                                    shape [N, 6].
        - latents_classes (torch.Tensor[int64]): Discrete latent indices per sample,
                                                 shape [N, 6].

    Latent factor order (size):
        [color (1), shape (3), scale (6), orientation (40), posX (32), posY (32)]

    Notes:
        - Images are binary (0/1) stored as int8; most models will want them converted
          to float and possibly normalized. Provide a `transform` to handle this.
        - All arrays are fully loaded into CPU memory at construction for fast access.
        - Set `download=True` (default) to fetch the NPZ if missing at `root`.
    """
    
    _NPZ_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/"
    _NPZ_FILENAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

    def __init__(self, 
                 root: Optional[Union[str, Path]] = None,
                 transform: Optional[Callable] = None,
                 download: bool = True):
        """Initialize the dSprites dataset.

        Args:
            root (str | pathlib.Path | None): Directory to store/find the NPZ file.
                Defaults to "./data/dSprites" when None.
            transform (Callable | None): Optional transform applied to each image.
            download (bool): If True and the dataset file is not present at `root`,
                it will be downloaded from the official GitHub URL.
        """
        # Assign default if no root is provided
        if root is None:
            root = Path('data') / 'dSprites'
        elif isinstance(root, str):
            root = Path(root)
        
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.filepath = self.root / DSprite._NPZ_FILENAME
            
        if download and not self.filepath.exists():
            url = DSprite._NPZ_URL + DSprite._NPZ_FILENAME
            print(f'Downloading dSprites from {url}')
            wget.download(url, out=str(self.filepath))
            print('\nDownload completed')
        
        data = np.load(self.filepath, allow_pickle=True)
        self.X = torch.as_tensor(data['imgs'], dtype=torch.int8).unsqueeze(1)
        self.latents_values = torch.as_tensor(data['latents_values'], dtype=torch.float64)
        self.latents_classes = torch.as_tensor(data['latents_classes'], dtype=torch.int64)
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.X[idx]
        lv = self.latents_values[idx]
        lc = self.latents_classes[idx]
        if self.transform:
            img = self.transform(img)
        return img, lv, lc
