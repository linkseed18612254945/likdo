import os

import torch
from PIL import Image
import json
from torchvision import transforms
from torch.utils.data import Dataset
from torchtext.data import Field
import nltk

class VQADataset(Dataset):
    def __init__(self, img_root, question_json, scene_json=None, transform=transforms.ToTensor(), sample_num=None):
        super(VQADataset, self).__init__()
        self.img_root = img_root
        self.transform = transform
        if isinstance(question_json, str):
            with open(question_json, 'r') as f:
                self.question_json = json.load(f)
        else:
            self.question_json = question_json

        self.scene_json = None
        if scene_json is not None and isinstance(scene_json, str):
            with open(scene_json, 'r') as f:
                self.scene_json = json.load(f)
        else:
            self.scene_json = scene_json

        self.dataset_info = self.question_json['info']
        self.annotations = dict()
        data_index = 0
        questions_tokens = []
        answers_tokens = []
        sample_num = len(self.question_json['questions']) if sample_num is None else sample_num
        for question in self.question_json['questions'][:sample_num]:
            question_tokens = question.get('tokens') or nltk.tokenize.word_tokenize(str(question['question']).lower())
            question_tokens = [w for w in question_tokens if isinstance(w, str)]
            question['tokens'] = question_tokens
            questions_tokens.append(question_tokens)
            if 'answer' in question:
                answer_tokens = nltk.tokenize.word_tokenize(str(question['answer']).lower())
                answers_tokens.append(answer_tokens)
            self.annotations[data_index] = question
            data_index += 1
        self.question_field = Field(init_token='<bos>', eos_token='<eos>')
        self.question_field.build_vocab(questions_tokens)
        self.answer_field = Field()
        self.answer_field.build_vocab(answers_tokens)

    def __getitem__(self, index):
        image_name = self.annotations[index]['image_filename']
        image = Image.open(os.path.join(self.img_root, image_name)).convert('RGB')
        if image.size[0] < 225 or image.size[1] < 255:
            image = image.resize((256, 256), Image.ANTIALIAS)
        if self.transform is not None:
            image = self.transform(image)
        tokens = self.annotations[index]['tokens']
        question = self.question_field.numericalize(self.question_field.pad([tokens])).squeeze()
        answer = self.annotations[index]['answer']
        answer = torch.LongTensor([self.answer_field.vocab.stoi[answer]])
        return image, question, answer

    def __len__(self):
        return len(self.annotations)

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
    images, questions, answers = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    answers = torch.stack(answers, 0).squeeze()

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(question) for question in questions]
    targets = torch.zeros(len(questions), max(lengths)).long()
    for i, q in enumerate(questions):
        end = lengths[i]
        targets[i, :end] = q[:end]
    return images, targets, lengths, answers

if __name__ == '__main__':
    img_root = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/val'
    question_json = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
    dataset = VQADataset(img_root=img_root, question_json=question_json)