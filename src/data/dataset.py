"""The one PyTorch Dataset used by every model, task and command."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CUBClassificationDataset(Dataset):
    """A view of a Hugging Face CUB split, selected by ORIGINAL row indices.

    Images are decoded lazily, one at a time, when the DataLoader asks for them.
    Each item is a dict with the transformed image, the integer label (0..199) and
    the original row index, so every prediction can be traced back to its image.
    """

    def __init__(self, hf_dataset, indices=None, transform=None):
        # Only image + label are needed; dropping the long caption column speeds up row access.
        if hasattr(hf_dataset, 'select_columns'):
            hf_dataset = hf_dataset.select_columns([c for c in ('image', 'label') if c in hf_dataset.column_names])
        self.data = hf_dataset
        self.indices = list(range(len(hf_dataset))) if indices is None else [int(i) for i in indices]
        self.transform = transform
        if any(i < 0 or i >= len(hf_dataset) for i in self.indices):
            raise ValueError('Row index outside the dataset')

    def __len__(self):
        return len(self.indices)

    def raw(self, position: int):
        """Original PIL image and label for a position in this view (no transform)."""
        item = self.data[self.indices[position]]
        return item['image'], int(item['label'])

    def __getitem__(self, position):
        row = self.indices[position]
        item = self.data[row]
        image = item['image'] if self.transform is None else self.transform(item['image'])
        return {'image': image,
                'label': torch.tensor(int(item['label']), dtype=torch.long),
                'row_index': torch.tensor(row, dtype=torch.long)}
