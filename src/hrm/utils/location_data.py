"""
Location prediction data utilities.
Handles loading and batching of trajectory data for next location prediction.
"""

import torch
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class LocationDataset:
    """Dataset for next location prediction."""
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 50,
        use_features: bool = True,
    ):
        """
        Initialize location dataset.
        
        Args:
            data_path: Path to pickle file with trajectory data
            max_seq_len: Maximum sequence length
            use_features: Whether to use additional features (weekday, time, duration, etc.)
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.use_features = use_features
        
        # Load data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        
        # Compute vocabulary size
        all_locations = []
        for sample in self.data:
            all_locations.extend(sample['X'].tolist())
            all_locations.append(sample['Y'])
        
        unique_locs = set(all_locations)
        self.vocab_size = max(unique_locs) + 1  # +1 for 0-based indexing
        self.num_locations = len(unique_locs)
        
        print(f"Vocabulary size: {self.vocab_size} (unique locations: {self.num_locations})")
        
        # Feature dimensions
        if self.use_features:
            sample = self.data[0]
            self.num_users = max([s['user_X'].max() for s in self.data]) + 1
            self.num_weekdays = 7
            self.max_start_min = 24 * 60  # minutes in a day
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.data[idx]
    
    def collate_fn(
        self,
        batch: List[Dict[str, np.ndarray]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of samples
            device: Target device
            
        Returns:
            Tuple of (inputs, targets, features_dict)
            - inputs: [batch_size, seq_len] location IDs
            - targets: [batch_size] target location IDs
            - features_dict: Optional dict with additional features
        """
        batch_size = len(batch)
        
        # Prepare tensors
        max_len = min(max(len(s['X']) for s in batch), self.max_seq_len)
        
        inputs = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        features_dict = None
        if self.use_features:
            features_dict = {
                'user': torch.zeros((batch_size, max_len), dtype=torch.long, device=device),
                'weekday': torch.zeros((batch_size, max_len), dtype=torch.long, device=device),
                'start_min': torch.zeros((batch_size, max_len), dtype=torch.float32, device=device),
                'duration': torch.zeros((batch_size, max_len), dtype=torch.float32, device=device),
                'diff': torch.zeros((batch_size, max_len), dtype=torch.long, device=device),
                'mask': torch.zeros((batch_size, max_len), dtype=torch.bool, device=device),
            }
        
        # Fill tensors
        for i, sample in enumerate(batch):
            seq = sample['X']
            # Take last max_len items
            if len(seq) > max_len:
                seq = seq[-max_len:]
                if self.use_features:
                    user_seq = sample['user_X'][-max_len:]
                    weekday_seq = sample['weekday_X'][-max_len:]
                    start_min_seq = sample['start_min_X'][-max_len:]
                    dur_seq = sample['dur_X'][-max_len:]
                    diff_seq = sample['diff'][-max_len:]
            else:
                if self.use_features:
                    user_seq = sample['user_X']
                    weekday_seq = sample['weekday_X']
                    start_min_seq = sample['start_min_X']
                    dur_seq = sample['dur_X']
                    diff_seq = sample['diff']
            
            seq_len = len(seq)
            inputs[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            targets[i] = sample['Y']
            
            if self.use_features:
                features_dict['user'][i, :seq_len] = torch.tensor(user_seq, dtype=torch.long)
                features_dict['weekday'][i, :seq_len] = torch.tensor(weekday_seq, dtype=torch.long)
                features_dict['start_min'][i, :seq_len] = torch.tensor(start_min_seq, dtype=torch.float32)
                features_dict['duration'][i, :seq_len] = torch.tensor(dur_seq, dtype=torch.float32)
                features_dict['diff'][i, :seq_len] = torch.tensor(diff_seq, dtype=torch.long)
                features_dict['mask'][i, :seq_len] = True
        
        return inputs, targets, features_dict


class LocationDataLoader:
    """DataLoader for location prediction."""
    
    def __init__(
        self,
        dataset: LocationDataset,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.dataset.collate_fn(batch, self.device)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
