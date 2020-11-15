import torch
from torch.utils.data import dataset
from utils import io
from datasets.common.vocab import Vocab, build_vocab
from datasets.common.transforms import get_image_transform
from torch.utils.data import dataloader, random_split
import nltk
from utils.logger import get_logger
from utils.text_process import pad_sentences_batch


logger = get_logger(__file__)

class DBpediaCencept(dataset.Dataset):
    @classmethod
    def get_description_from_csv(cls, path, extra_param):
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

    def __init__(self, text_path, classes_path, vocab_json, desc_transformer=None,
                 item_transformer=None, label_transformer=None, data_nums='all'):
        super(DBpediaCencept, self).__init__()

        # initialize args
        self.text_path = text_path
        self.vocab = Vocab.from_json(vocab_json)
        self.idx2classes = {i + 1: k for i, k in enumerate(io.load_txt(classes_path))}
        self.data_nums = data_nums
        self.class_nums = len(self.idx2classes)

        self.desc_transformer = desc_transformer
        self.item_transformer = item_transformer
        self.label_transformer = label_transformer

    @property
    def annotations(self):
        annotations = dict()
        df = io.load_csv(self.text_path, without_header=True)
        data_nums = df.shape[0] if self.data_nums == 'all' else self.data_nums
        for index, row in df.iterrows():
            tokens = nltk.tokenize.word_tokenize(row[2]).lower()
            annotations[index] = {'label': row[0], 'label_name': self.idx2classes[row[0]],
                                  'desc': tokens,  'item': row[1]}
            if index >= data_nums:
                break
        return annotations

    def __getitem__(self, index):
        if self.desc_transformer is not None:
            desc = self.desc_transformer((self.annotations[index]['desc']))
        else:
            desc = self.vocab.map_sequence(self.annotations[index]['desc'])

        if self.item_transformer is not None:
            item = self.item_transformer((self.annotations[index]['item']))
        else:
            item = self.vocab.map_sequence(self.annotations[index]['item'])

        if self.label_transformer is not None:
            label_name = self.label_transformer((self.annotations[index]['label_name']))
        else:
            label_name = self.vocab.map_sequence(self.annotations[index]['label_name'])

        return desc, item, self.annotations[index]['label'], label_name

    def __len__(self):
        return len(self.annotations.keys())

    @classmethod
    def collate_fn(cls, data):
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[0]), reverse=True)
        descs, items, labels, label_names = zip(*data)
        desc_targets, desc_lengths = pad_sentences_batch(descs)
        item_targets, items_lengths = pad_sentences_batch(items)
        label_name_targets, label_name_lengths = pad_sentences_batch(label_names)

        feed_dict = {
            "descs": desc_targets,
            "desc_lengths": desc_lengths,
            "items": item_targets,
            "label": labels,
            "label_name": label_name_targets,
            "is_static_vector": False
        }
        return feed_dict

def get_text_classify_data(config):
    transform = get_image_transform()
    if 'dbpedia' in config.data.name:
        train_dataset = DBpediaCencept(text_path=config.data.train_text_path,
                                       classes_path=config.data.classes_path,
                                       vocab_json=config.data.vocab_path,
                                       desc_transformer=None, item_transformer=None, label_transformer=None,
                                       data_nums=config.train.train_data_nums)
        if config.data.valid_caption_path is not None:
            valid_dataset = DBpediaCencept(text_path=config.data.valid_text_path,
                                           classes_path=config.data.classes_path,
                                           vocab_json=config.data.vocab_path,
                                           desc_transformer=None, item_transformer=None, label_transformer=None,
                                           data_nums=config.train.valid_data_nums)
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
    dbpedia_data_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'
    classes_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/classes.txt'
    vocab_path = '/home/ubuntu/likun/vocab/dbpedia_vocab.json'

    # build vocab
    build_vocab(file_paths=(dbpedia_data_path, dbpedia_data_path, classes_path),
                compile_functions=(DBpediaCencept.get_description_from_csv,DBpediaCencept.get_item_from_csv, DBpediaCencept.get_class_name_from_csv),
                extra_param=None, vocab_path=vocab_path)

