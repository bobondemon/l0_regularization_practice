import torch
import torch.utils.data as data
from torch.utils.data import dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl


class CIFAR10Dataset(dataset.Dataset):
    def __init__(self, root, types):
        super().__init__()
        if types not in ["train", "val", "test"]:
            raise TypeError(f"Only `train`, `val`, or `test` are allowed in argument `types`")
        self._types = types
        self._root = root
        mean_and_std = self._cal_mean_and_std_by_train_data()
        val_and_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_and_std)])
        # For training, we add some augmentation. Networks are too powerful and would overfit.
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(*mean_and_std),
            ]
        )
        # Loading the training dataset. We need to split it into a training and validation part
        # We need to do a little trick because the validation set should not use the augmentation.
        self._data = None
        if types == "train":
            train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
            pl.seed_everything(42)
            self._data, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
        elif types == "val":
            val_dataset = CIFAR10(root=root, train=True, transform=val_and_test_transform, download=True)
            pl.seed_everything(42)
            _, self._data = torch.utils.data.random_split(val_dataset, [45000, 5000])
        else:
            self._data = CIFAR10(root=root, train=False, transform=val_and_test_transform, download=True)

    def _cal_mean_and_std_by_train_data(self):
        raw_train_dataset = CIFAR10(root=self._root, train=True, download=True)
        means = (raw_train_dataset.data / 255.0).mean(axis=(0, 1, 2))
        stds = (raw_train_dataset.data / 255.0).std(axis=(0, 1, 2))
        return means, stds

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def types(self):
        return self._types


class CIFAR10TrainDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=128, num_workers=4, loader_kwargs={}):
        if not isinstance(dataset, CIFAR10Dataset):
            raise ValueError("dataset only be type `CIFAR10Dataset`")
        if not dataset.types == "train":
            raise ValueError("dataset only be type `train`")
        # overwrite anyway
        loader_kwargs["shuffle"] = True
        loader_kwargs["drop_last"] = True
        loader_kwargs["pin_memory"] = True
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, **loader_kwargs)


class CIFAR10ValTestDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=128, num_workers=4, loader_kwargs={}):
        if not isinstance(dataset, CIFAR10Dataset):
            raise ValueError("dataset only be type `CIFAR10Dataset`")
        if not (dataset.types == "val" or dataset.types == "test"):
            raise ValueError("dataset only be type `val` or `test`")
        # overwrite anyway
        loader_kwargs["shuffle"] = False
        loader_kwargs["drop_last"] = False
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, **loader_kwargs)


if __name__ == "__main__":
    import torchvision
    from PIL import Image
    import matplotlib.pyplot as plt

    NUM_IMAGES = 24
    data_root_path = "./data"
    loaded_dataset = CIFAR10Dataset(data_root_path, "train")
    dataset_iter = iter(loaded_dataset)
    # test_dataset = CIFAR10Dataset(data_root_path, 'test')
    train_images = [next(dataset_iter)[0] for idx in range(NUM_IMAGES)]

    img_grid = torchvision.utils.make_grid(torch.stack(train_images, dim=0), nrow=8, normalize=True, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title(f"First {NUM_IMAGES} examples on CIFAR10")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
