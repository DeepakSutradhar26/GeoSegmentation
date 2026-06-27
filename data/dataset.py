from torch.utils.data import Dataset

class GeoDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        aerial_image_path = self.data_paths[idx][0]
        binary_mask_path = self.data_paths[idx][1]
        pass