from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Tokenize here.
        return self.data[index]

    def __len__(self):
        return len(self.data)