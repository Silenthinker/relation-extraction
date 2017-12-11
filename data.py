import torch
from torch.utils.data import Dataset


class DDI2013Dataset(Dataset):
    """
    Dataset Class for relation extraction

    args: 
        data_tensor (ins_num, seq_length): words 
        target_tensor (ins_num, 1): targets
    """
    
    target_map = {t:i for i, t in enumerate(['null', 'advise', 'effect', 'mechanism', 'int'])}
    
    def __init__(self, data_tensor, target_tensor, position_tensor, indices, mask):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.position_tensor = position_tensor
        self.indices = indices
        self.mask = mask

    def __getitem__(self, index):
        return {'feature': self.data_tensor[index], 
                'position': self.position_tensor[index], 
                'target': self.target_tensor[index], 
                'index': self.indices[index],
                'mask': self.mask[index]}

    def __len__(self):
        return self.data_tensor.size(0)
    
    def collate(self, samples):
        """
        merges a list of samples to form a mini-batch
        """
        def merge(key):
            return torch.stack([s[key] for s in samples], dim=0)
        
        keys = ['feature', 'position', 'target', 'index', 'mask']
        res = {k:merge(k) for k in keys}
        
        return res