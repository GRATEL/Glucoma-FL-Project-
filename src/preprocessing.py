import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class RefugeDataset(Dataset):
    def __init__(self, split_dir, transform, image_subfolder="Images_Cropped"):
        index_path = os.path.join(split_dir, "index.json")
        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.entries = list(self.index.values())
        self.image_dir = os.path.join(split_dir, image_subfolder)
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.image_dir, entry["ImgName"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = entry["Label"]
        return img, label