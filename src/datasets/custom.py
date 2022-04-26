from torch.utils.data import DataLoader, Dataset
from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import torch
import numpy as np



import torchvision.transforms as transforms


class Custom_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [-8.5016, 7.0120]

        # Custom dataset preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[0]],
                                                             [min_max[1] - min_max[0]])])

    
        self.train_set = Custom(root = self.root, transform=transform)
        self.test_set=None

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""

        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size,
                        shuffle=shuffle_train, num_workers=num_workers)
        # test_dataLoader = DataLoader(self.test_set, batch_size=batch_size,
        #                 shuffle=shuffle_test, num_workers=num_workers)

        test_dataLoader = None
        print("***********in Loader***************")
        return train_dataLoader, test_dataLoader

class Custom(Dataset):
    """FLARACC dataset."""

    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(root)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample=self.data[idx]
        sample = np.array([sample])
        sample = sample.astype(np.float32)
        if self.transform:
            sample = self.transform(sample).reshape(1,184,608)

        return sample, [0], [0]
