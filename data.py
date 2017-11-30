from torch.utils.data import Dataset


class REDataset(Dataset):
    """
    Dataset Class for relation extraction

    args: 
        data_tensor (ins_num, seq_length): words 
        target_tensor (ins_num, 1): targets
    """
    def __init__(self, data_tensor, target_tensor, position_tensor, indices):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.position_tensor = position_tensor
        self.indices = indices

    def __getitem__(self, index):
        return (self.data_tensor[index], self.position_tensor[index]), self.target_tensor[index], self.indices[index]

    def __len__(self):
        return self.data_tensor.size(0)