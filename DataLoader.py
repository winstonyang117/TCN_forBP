import torch

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
   
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
   
    def __len__(self):
        return len(self.data)