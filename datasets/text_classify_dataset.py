import nltk
from torch.utils.data import dataset
from utils import io
from datasets.common.vocab import Vocab, build_vocab
from datasets.common.transforms import *
from torch.utils.data import dataloader, random_split
from utils.logger import get_logger
from utils.text_process import pad_sentences_batch
import tqdm
import pandas as pd
logger = get_logger(__file__)


class News20(dataset.Dataset):
    @classmethod
    def get_text_from_csv(cls, path, extra_param):
        df = io.load_csv(path, without_header=False)
        return df['text'].tolist()

    @classmethod
    def get_class_name_from_csv(cls, path, extra_param):
        df = io.load_csv(path, without_header=False)
        res = []
        for label in df['label'].tolist():
            res.extend(label.split('.'))
        return res

    def __init__(self, text_path, text_vocab_json, label_vocab_json, text_preprocess=None,
                 text_transformer=None, label_transformer=None, label_name_transformer=None,
                 data_nums='all', train_labels='all'):
        super(News20, self).__init__()

        # initialize args
        self.text_path = text_path
        self.vocab = Vocab.from_json(text_vocab_json)
        self.label_vocab = Vocab.from_json(label_vocab_json)
        self.text_preprocess = text_preprocess
        self.data_nums = data_nums
        self.train_labels = train_labels
        self.num_labels = len(self.label_vocab.idx2word)

        self.annotations = self.get_annotations()

        self.text_transformer = text_transformer
        self.label_transformer = label_transformer
        self.label_name_transformer = label_name_transformer

    def get_annotations(self):
        annotations = dict()
        df = io.load_csv(self.text_path, without_header=False, shuffle=True)
        data_nums = df.shape[0] if self.data_nums == 'all' else self.data_nums
        for index, row in tqdm.tqdm(df.iterrows(), desc='Build dataset annotations', total=data_nums):
            if self.train_labels != 'all' and row['label'] not in self.train_labels:
                continue

            text = row['text']
            if self.text_preprocess is not None:
                text = self.text_preprocess(text)
            text = nltk.tokenize.word_tokenize(text.lower())
            annotations[index] = {'label': self.label_vocab.word2idx[row['label']], 'label_name': row['label'],
                                  'text': text}
            if index >= data_nums:
                break
        return annotations

    def __getitem__(self, index):
        if self.text_transformer is not None:
            text = self.text_transformer((self.annotations[index]['text']))
        else:
            text = self.vocab.map_sequence(self.annotations[index]['text'])

        if self.label_transformer is not None:
            label = self.label_transformer([self.annotations[index]['label']])
        else:
            label = self.annotations[index]['label']

        if self.label_name_transformer is not None:
            label_name = self.label_name_transformer((self.annotations[index]['label_name']))
        else:
            label_name = self.vocab.map_sequence(self.annotations[index]['label_name'])

        return text, label, label_name

    def __len__(self):
        return len(self.annotations.keys())

    @classmethod
    def collate_fn(cls, data):
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[0]), reverse=True)
        texts, labels, label_names = zip(*data)
        if isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, 0).squeeze()
        text_targets, text_lengths = pad_sentences_batch(texts)
        label_name_targets, label_name_lengths = pad_sentences_batch(label_names)

        feed_dict = {
            "texts": text_targets,
            "text_lengths": text_lengths,
            "labels": labels,
            "label_names": label_name_targets,
            "is_static_vector": False
        }
        return feed_dict


class DBpediaConcept(dataset.Dataset):
    @classmethod
    def get_text_from_csv(cls, path, extra_param):
        df = io.load_csv(path, without_header=True)
        return df[2].tolist()

    @classmethod
    def get_item_from_csv(cls, path, extra_param):
        df = io.load_csv(path, without_header=True)
        return df[1].tolist()

    @classmethod
    def get_class_name_from_csv(cls, path, extra_param):
        classes = io.load_txt(path)
        return classes

    def __init__(self, text_path, classes_path, vocab_json, text_transformer=None,
                 item_transformer=None, label_transformer=None, label_name_transformer=None,
                 data_nums='all', train_labels='all'):
        super(DBpediaConcept, self).__init__()

        # initialize args
        self.text_path = text_path
        self.vocab = Vocab.from_json(vocab_json)
        self.idx2classes = io.load_txt(classes_path)
        self.data_nums = data_nums
        self.num_labels = len(self.idx2classes)
        self.train_labels = train_labels
        self.annotations = self.get_annotations()

        self.text_transformer = text_transformer
        self.item_transformer = item_transformer
        self.label_transformer = label_transformer
        self.label_name_transformer = label_name_transformer

    def get_annotations(self):
        annotations = dict()
        df = io.load_csv(self.text_path, without_header=True, shuffle=True)
        data_nums = df.shape[0] if self.data_nums == 'all' else self.data_nums
        for index, row in tqdm.tqdm(df.iterrows(), desc='Build dataset annotations', total=data_nums):
            # tokens = nltk.tokenize.word_tokenize(row[2].lower())
            if self.train_labels != 'all' and self.idx2classes[row[0] - 1] not in self.train_labels:
                continue
            annotations[index] = {'label': row[0] - 1, 'label_name': self.idx2classes[row[0] - 1],
                                  'text': row[2].lower(),  'item': row[1]}
            if index >= data_nums:
                break
        return annotations

    def __getitem__(self, index):
        if self.text_transformer is not None:
            text = self.text_transformer((self.annotations[index]['text']))
        else:
            text = self.vocab.map_sequence(self.annotations[index]['text'])

        if self.item_transformer is not None:
            item = self.item_transformer((self.annotations[index]['item']))
        else:
            item = self.vocab.map_sequence(self.annotations[index]['item'])

        if self.label_transformer is not None:
            label = self.label_transformer([self.annotations[index]['label']])
        else:
            label = self.annotations[index]['label']

        if self.label_name_transformer is not None:
            label_name = self.label_name_transformer((self.annotations[index]['label_name']))
        else:
            label_name = self.vocab.map_sequence(self.annotations[index]['label_name'])

        return text, item, label, label_name

    def __len__(self):
        return len(self.annotations.keys())

    @classmethod
    def collate_fn(cls, data):
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[0]), reverse=True)
        texts, items, labels, label_names = zip(*data)
        if isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, 0).squeeze()
        text_targets, text_lengths = pad_sentences_batch(texts)
        item_targets, items_lengths = pad_sentences_batch(items)
        label_name_targets, label_name_lengths = pad_sentences_batch(label_names)

        feed_dict = {
            "texts": text_targets,
            "text_lengths": text_lengths,
            "items": item_targets,
            "labels": labels,
            "label_names": label_name_targets,
            "is_static_vector": False
        }
        return feed_dict

def get_text_classify_data(config):
    bert_transform = BertTokenizerTransformer(config.model.bert_base_path)
    longtensor_transform = torch.LongTensor
    if 'dbpedia' in config.data.name:
        train_dataset = DBpediaConcept(text_path=config.data.train_text_path,
                                       classes_path=config.data.classes_path,
                                       vocab_json=config.data.vocab_path,
                                       text_transformer=bert_transform, label_transformer=longtensor_transform,
                                       data_nums=config.train.train_data_nums)
        if config.data.valid_text_path is not None:
            valid_dataset = DBpediaConcept(text_path=config.data.valid_text_path,
                                           classes_path=config.data.classes_path,
                                           vocab_json=config.data.vocab_path,
                                           text_transformer=bert_transform, label_transformer=longtensor_transform,
                                           data_nums=config.train.valid_data_nums)
        else:
            train_size = int((1 - config.train.valid_percent) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
            train_dataset = train_dataset.dataset
            valid_dataset = valid_dataset.dataset
    elif 'news20' in config.data.name:
        train_dataset = News20(text_path=config.data.train_text_path,
                               text_vocab_json=config.data.vocab_path,
                               label_vocab_json=config.data.label_vocab_path,
                               text_transformer=bert_transform, label_transformer=longtensor_transform,
                               label_name_transformer=None, data_nums=config.train.train_data_nums)
        if config.data.valid_text_path is not None:
            valid_dataset = News20(text_path=config.data.valid_text_path,
                                   text_vocab_json=config.data.vocab_path,
                                   label_vocab_json=config.data.label_vocab_path,
                                   text_transformer=bert_transform, label_transformer=longtensor_transform,
                                   label_name_transformer=None, data_nums=config.train.train_data_nums)
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
    data_path = '/home/ubuntu/likun/nlp_data/text_classify/20news-18828/train_flat.csv'
    classes_path = '/home/ubuntu/likun/nlp_data/text_classify/20news-18828/train_flat.csv'
    text_vocab_path = '/home/ubuntu/likun/vocab/news20_vocab.json'
    label_vocab_path = '/home/ubuntu/likun/vocab/news20_classes.json'

    #
    # # build vocab
    # build_vocab(file_paths=(classes_path,),
    #             compile_functions=(News20.get_class_name_from_csv,),
    #             extra_param=None, vocab_path=vocab_path, add_special_token=False)

    # build dataset
    train_dataset = News20(text_path=data_path,
                           text_vocab_json=text_vocab_path,
                           label_vocab_json=label_vocab_path,
                           text_transformer=None, label_transformer=None,
                           label_name_transformer=None, data_nums=5000)
