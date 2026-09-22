from preprocessing import RefugeDataset, base_transform
from torch.utils.data import DataLoader

train_dataset = RefugeDataset("data/raw/REFUGE/train", base_transform)
print("Dataset size:", len(train_dataset))

loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

images, labels = next(iter(loader))
print("Batch shape:", images.shape)
print("Labels:", labels)