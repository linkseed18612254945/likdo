import torch
from torchvision import transforms
from torch.utils.data import dataset
from utils import io, container
from datasets.vocab import Vocab
import os
import random
import nltk
from PIL import Image


def flickr_annotation(caption_json, caption_per_image):
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

def coco_annotation(caption_json):
    pass


class ImageCaptionDataset(dataset.Dataset):
    def __init__(self, img_root, caption_json, vocab_json, annotation_func=flickr_annotation,
                 transform=transforms.ToTensor(), caption_per_image=None):
        super(ImageCaptionDataset, self).__init__()

        # initialize args
        self.img_root = img_root
        if isinstance(caption_json, str):
            self.caption_json = io.load_json(caption_json)
        else:
            self.caption_json = caption_json
        self.dataset_name = self.caption_json['dataset']
        self.annotations = annotation_func(self.caption_json, caption_per_image)
        self.vocab = Vocab.from_json(vocab_json)
        self.transform = transform

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


def collate_fn(data):
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
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths