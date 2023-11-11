from torch.utils.data import Dataset
from config import BASE_DIR
from PIL import Image
import os


class Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(BASE_DIR, f"data/processed/images/{self.df['image'][idx]}.tif")

        image = Image.open(path)
        image = self.transforms(image)

        label = self.df['label'][idx]

        return image, label
