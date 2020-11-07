from PIL import Image
import os

from torch.utils.data import dataloader

from datasets.common.transforms import get_image_transform
from utils import io
from datasets.common.vocab import Vocab, build_vocab
from torch.utils.data.dataset import Dataset, random_split
import torch

class ClevrDataset(Dataset):
    @classmethod
    def get_questions_from_json(cls, json_file):
        question_json = io.load_json(json_file)
        questions = []
        for question in question_json['questions']:
            questions.append(question['question'])
        return questions

    @classmethod
    def get_answers_from_json(cls, json_file):
        question_json = io.load_json(json_file)
        answers = []
        for question in question_json['questions']:
            if len(question['answer'].split()) > 1:
                raise Exception('Answer is not a single word')
            answers.append(question['answer'])
        return answers

    @classmethod
    def build_annotation(cls, question_json, data_nums, question_per_image=None):
        annotations = dict()
        data_nums = len(question_json['questions']) if data_nums == 'all' else data_nums
        use_nums = 0
        for img_info in question_json['questions']:
            annotations[img_info['question_index']] = img_info
            if use_nums >= data_nums:
                break
            use_nums += 1
        return annotations

    def __init__(self, image_root_path, question_json_path, question_vocab_path, answer_vocab_path,
                 transform, data_nums='all', question_per_image=None):
        super(ClevrDataset, self).__init__()
        self.image_root_path = image_root_path
        if isinstance(question_json_path, str):
            self.question_json = io.load_json(question_json_path)
        else:
            self.question_json = question_json_path
        self.dataset_info = self.question_json['info']
        self.question_vocab = Vocab.from_json(question_vocab_path)
        self.answer_vocab = Vocab.from_json(answer_vocab_path)
        self.transform = transform

        self.annotations = ClevrDataset.build_annotation(self.question_json, data_nums, question_per_image)

    def __getitem__(self, index):
        img_name = self.annotations[index]['image_filename']
        image = Image.open(os.path.join(self.image_root_path, img_name)).convert('RGB')
        if image.size[0] < 225 or image.size[1] < 255:
            image = image.resize((256, 256), Image.ANTIALIAS)
        if self.transform is not None:
            image = self.transform(image)
        question = self.annotations[index].get('question')
        question = self.question_vocab.map_sequence(question)
        answer = self.annotations[index].get('answer')
        answer = self.answer_vocab.map_sequence(answer, add_bos=False, add_eos=False)
        return image, question, answer

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
        images, questions, answers = zip(*data)
        answers = torch.LongTensor([a[0] for a in answers])
        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        question_lengths = [len(qu) for qu in questions]
        question_targets = torch.zeros(len(questions), max(question_lengths)).long()
        for i, qu in enumerate(questions):
            qu = torch.LongTensor(qu)
            end = question_lengths[i]
            question_targets[i, :end] = qu[:end]
        feed_dict = {
            "images": images,
            "questions": question_targets,
            "question_lengths": question_lengths,
            "answers": answers,
        }
        return feed_dict


def get_vqa_dataset(config):
    transform = get_image_transform()
    if 'clevr' in config.data.name:
        train_dataset = ClevrDataset(image_root_path=config.data.train_image_root_path,
                                     question_json_path=config.data.train_question_path,
                                     question_vocab_path=config.data.question_vocab_path,
                                     answer_vocab_path=config.data.answer_vocab_path,
                                     transform=transform, data_nums=config.data.train_data_nums)
        if config.data.valid_question_path is not None:
            valid_dataset = ClevrDataset(image_root_path=config.data.valid_image_root_path,
                                         question_json_path=config.data.valid_question_path,
                                         question_vocab_path=config.data.question_vocab_path,
                                         answer_vocab_path=config.data.answer_vocab_path,
                                         transform=transform, data_nums=config.data.valid_data_nums)
        else:
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
    # build vocab
    clevr_json_files = ('/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json',
                        '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_val_questions.json',
                        '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_test_questions.json')
    q_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_question_vocab.json'
    a_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_answer_vocab.json'
    build_vocab(file_paths=clevr_json_files, compile_functions=(ClevrDataset.get_questions_from_json, ) * 3, vocab_path=q_vocab_path)
    build_vocab(file_paths=clevr_json_files, compile_functions=(ClevrDataset.get_answers_from_json, ) * 3, vocab_path=a_vocab_path)