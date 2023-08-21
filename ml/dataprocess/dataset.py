from torch import tensor
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        return tensor(text), tensor(label)
