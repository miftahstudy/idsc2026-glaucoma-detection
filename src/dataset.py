import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        required_cols = ["Image Name", "Label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan pada dataframe")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["Image Name"]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")

        image = Image.open(img_path).convert("RGB")

        label_str = str(row["Label"]).strip()
        label = 1.0 if label_str == "GON+" else 0.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)