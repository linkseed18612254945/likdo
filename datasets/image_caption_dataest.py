import torch
from torchvision import transforms
from torch.utils.data import dataset
from utils import io
from datasets.common.vocab import Vocab, build_vocab
import os
import random
from PIL import Image
from datasets.common.transforms import get_image_transform
from torch.utils.data import dataloader, random_split
import nltk
from utils.logger import get_logger


logger = get_logger(__file__)

class FlickrDataset(dataset.Dataset):
    @classmethod
    def get_captions_from_json(cls, path):
        dataset = io.load_json(path)['images']
        captions = []
        for i, d in enumerate(dataset):
            captions += [str(x['raw']) for x in d['sentences']]
        return captions

    @classmethod
    def build_annotation(cls, caption_json, caption_per_image):
        annotations = dict()
        img_index = 0
        for img_info in caption_json['images']:
            if caption_per_image is not None:
                use_sentences = random.choices(img_info['sentences'], k=caption_per_image)
            else:
                use_sentences = img_info['sentences']
            for sentence in use_sentences:
                tokens = sentence.get('tokens') or nltk.tokenize.word_tokenize(str(sentence['raw']).lower())
                annotations[img_index] = {'filename': img_info['filename'], 'raw': sentence['raw'],
                                          'tokens': tokens}
                img_index += 1
        return annotations

    def __init__(self, img_root, caption_json, vocab_json,
                 transform=transforms.ToTensor(), caption_per_image=None):
        super(FlickrDataset, self).__init__()

        # initialize args
        self.img_root = img_root
        if isinstance(caption_json, str):
            self.caption_json = io.load_json(caption_json)
        else:
            self.caption_json = caption_json
        self.dataset_name = self.caption_json['dataset']
        self.vocab = Vocab.from_json(vocab_json)
        self.transform = transform

        self.annotations = FlickrDataset.build_annotation(self.caption_json, caption_per_image)

    def __getitem__(self, index):
        # Image load and transform
        img_name = self.annotations[index]['filename']
        image = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        if image.size[0] < 225 or image.size[1] < 255:
            image = image.resize((256, 256), Image.ANTIALIAS)
        if self.transform is not None:
            image = self.transform(image)

        # question sentence numericalize
        tokens = self.annotations[index].get('tokens')
        caption = self.vocab.map_sequence(tokens)

        return image, caption

    def __len__(self):
        return len(self.annotations.keys())

    @classmethod
    def collate_fn(cls, data):
        """Creates mini-batch tensors from the list of tuples (image, caption).

        We should build custom collate_fn rather than using default collate_fn,
        because merging caption (including padding) is not supported in default.

        Args:
            data: list of tuple (image, caption).
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            cap = torch.LongTensor(cap)
            end = lengths[i]
            targets[i, :end] = cap[:end]
        feed_dict = {
            "images": images,
            "captions": targets,
            "lengths": lengths
        }
        return feed_dict

def get_image_caption_data(config):
    transform = get_image_transform()
    if 'flickr' in config.data.name:
        train_dataset = FlickrDataset(img_root=config.data.image_root_path,
                                      caption_json=config.data.train_caption_path,
                                      vocab_json=config.data.vocab_path, transform=transform)
        if config.data.valid_caption_path is not None:
            valid_dataset = FlickrDataset(img_root=config.data.image_root_path,
                                          caption_json=config.data.valid_caption_path,
                                          vocab_json=config.data.vocab_path, transform=transform)
        else:
            train_size = int((1 - config.train.valid_percent) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
            train_dataset = train_dataset.dataset
            valid_dataset = valid_dataset.dataset
    else:
        raise Exception(f'Not support data {config.data.name}')
    train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                                         shuffle=config.train.shuffle, collate_fn=train_dataset.collate_fn)
    valid_loader = dataloader.DataLoader(dataset=valid_dataset, batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                                         shuffle=config.train.shuffle, collate_fn=valid_dataset.collate_fn)
    return train_dataset, train_loader, valid_dataset, valid_loader


if __name__ == '__main__':
    img_root = '/home/ubuntu/likun/image_data/flickr8k-images'
    caption_path = '/home/ubuntu/likun/image_data/caption/dataset_flickr8k.json'
    vocab_path = '/home/ubuntu/likun/image_data/vocab/flickr8k_vocab.json'

    # build vocab
    build_vocab(file_paths=(caption_path,), compile_functions=(FlickrDataset.get_captions_from_json,), vocab_path=vocab_path)

    # build dataset
    # trans = get_image_transform()
    # ImageCaptionDataset(img_root=img_root, caption_json=caption_path, vocab_json=vocab_path, transform=trans)