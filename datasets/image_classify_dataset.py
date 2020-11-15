from torch.utils.data import dataset
from torch.utils.data import dataloader, random_split
from datasets.common.transforms import get_image_transform
import torchvision
from torchvision import transforms


class Cifar10(dataset.Dataset):
    def __init__(self, file_root, transform=transforms.ToTensor(), target_transform=None):
        super(Cifar10, self).__init__()
        self.root_path = file_root
        self._dataset = torchvision.datasets.CIFAR10(root=file_root, transform=transform)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img, target = self._dataset.__getitem__(item)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.transform(self._dataset.classes[target])
        return img, target

    def __len__(self):
        return self._dataset.__len__()

    def get_class_map(self):
        return self._dataset.classes

    @classmethod
    def collate_fn(cls, data):
        images, targets = zip(*data)
        feed_dict = {
            "images": images,
            "targets": targets,
        }
        return feed_dict


def get_classify_data(config):
    transform = get_image_transform()
    if 'cifar10' == config.data.name:
        train_dataset = Cifar10(file_root=config.data.image_root_path, transform=transform)
        train_size = int((1 - config.train.valid_percent) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        train_dataset = train_dataset.dataset
        valid_dataset = valid_dataset.dataset
    else:
        raise Exception(f'Not support data {config.data.name}')
    train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=config.train.batch_size,
                                         num_workers=config.train.num_workers,
                                         shuffle=config.train.shuffle, collate_fn=train_dataset.collate_fn)
    valid_loader = dataloader.DataLoader(dataset=valid_dataset, batch_size=config.train.batch_size,
                                         num_workers=config.train.num_workers,
                                         shuffle=config.train.shuffle, collate_fn=valid_dataset.collate_fn)
    return train_dataset, train_loader, valid_dataset, valid_loader


if __name__ == '__main__':
    root_path = '/home/ubuntu/likun/image_data'