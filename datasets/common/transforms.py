import torchvision
import torch
from datasets.common.vocab import Vocab

class GloveVectorTransform(object):
    def __init__(self, glove_file_path):
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            glove_vectors = f.read().splitlines()
        self.glove_vectors = {l.split()[0]: l.split()[1:] for l in glove_vectors}

    def __call__(self, word):
        return torch.FloatTensor(list(map(float, self.glove_vectors.get(word))))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_image_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
    return transform

def get_text_static_transform(static_vector_path, glove_vocab):
    transform = GloveVectorTransform(static_vector_path, glove_vocab)
    return transform


