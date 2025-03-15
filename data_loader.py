import torch
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir="../data", batch_sizes=(64, 32, 16)):
    """Tạo DataLoader cho tập train, validation và test."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    # Tải dataset
    train_val_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    # Chia train/val
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    for images, labels in train_loader:
        print('Sample batch - images shape:', images.shape)
        print('Sample batch - labels shape:', labels.shape)
        break
